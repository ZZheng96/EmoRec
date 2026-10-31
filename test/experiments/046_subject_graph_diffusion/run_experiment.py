import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
PRED = HERE / "predictions"
CACHE = HERE / "cache" / "graph_features.npz"


def load_cache(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def build_cache():
    src = load_cache((HERE / CONFIG["spectral_cache"]).resolve())
    names = np.asarray(src["feature_names"])
    mask = np.asarray([str(n).startswith("log_bandpower_") for n in names])
    if mask.sum() != 150:
        raise RuntimeError("001 cache does not contain exactly 150 log-bandpower features")
    X_train = src["X_train"][:, mask].astype(np.float64)
    X_public = src["X_public"][:, mask].astype(np.float64)
    # Subject-local graph: cosine kNN, deterministic tie-breaking by distance/index.
    def diffuse(X, ids):
        out = np.empty((len(X), X.shape[1] * 2), dtype=X.dtype)
        for subject in np.unique(ids):
            rows = np.flatnonzero(ids == subject)
            z = X[rows]
            norms = np.linalg.norm(z, axis=1, keepdims=True)
            sim = (z @ z.T) / np.maximum(norms @ norms.T, 1e-12)
            np.fill_diagonal(sim, -np.inf)
            k = min(CONFIG["graph"]["k_neighbors"], len(rows) - 1)
            neigh = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            weights = np.take_along_axis(sim, neigh, axis=1)
            weights = np.maximum(weights, 0.0)
            denom = weights.sum(axis=1, keepdims=True)
            weights = np.divide(weights, denom, out=np.full_like(weights, 1.0 / k), where=denom > 1e-12)
            neighbor_mean = (weights[..., None] * z[neigh]).sum(axis=1)
            out[rows] = np.concatenate([z, (1 - CONFIG["graph"]["alpha"]) * z + CONFIG["graph"]["alpha"] * neighbor_mean], axis=1)
        return out
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X_control=X_train, X_diffusion=diffuse(X_train, src["subject_ids"]), X_public_control=X_public, X_public_diffusion=diffuse(X_public, src["public_user_ids"]), y_train=src["y_train"], subject_ids=src["subject_ids"], health_groups=src["health_groups"], original_trial_ids=src["original_trial_ids"], public_user_ids=src["public_user_ids"], public_trial_ids=src["public_trial_ids"])


def model(seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**CONFIG["hgb"], random_state=seed))


def folds_for(subjects, groups, seed):
    from sklearn.model_selection import StratifiedKFold
    unique = np.unique(subjects); ug = np.asarray([groups[np.flatnonzero(subjects == s)[0]] for s in unique])
    assignment = {}
    for f, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(unique, ug)):
        for s in unique[valid]: assignment[s] = f
    return np.asarray([assignment[s] for s in subjects], dtype=int)


def assert_subject_isolation(subjects, folds):
    for f in range(5):
        assert not (set(subjects[folds == f]) & set(subjects[folds != f]))


def metrics(y, p):
    pred = p >= .5; tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": .5, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def summarize(y, p, ref, folds):
    subjects = []
    for s in ref.subject_id.unique():
        m = ref.subject_id.to_numpy() == s; subjects.append({"subject_id": s, "health_group": ref.loc[m, "health_group"].iloc[0], **metrics(y[m], p[m])})
    groups = {}
    for g in ("HC", "DEP"):
        m = ref.health_group.to_numpy() == g; rows = [r for r in subjects if r["health_group"] == g]
        groups[g] = {"n_subjects": len(rows), "pooled": metrics(y[m], p[m]), "subject_auc_mean": float(np.mean([r["auc"] for r in rows]))}
    return {"overall": metrics(y, p), "folds": [{"fold": f, "n_subjects": int(ref.loc[folds == f, "subject_id"].nunique()), "n_samples": int((folds == f).sum()), **metrics(y[folds == f], p[folds == f])} for f in range(5)], "subject_auc_mean": float(np.mean([r["auc"] for r in subjects])), "health_groups": groups, "subjects": subjects}


def evaluate(name, X, XP, y, ref, folds, users, trials, seed):
    p = np.full(len(y), np.nan)
    for f in range(5):
        train, valid = folds != f, folds == f; fit = model(seed + f); fit.fit(X[train], y[train]); p[valid] = fit.predict_proba(X[valid])[:, 1]
    if not np.isfinite(p).all(): raise RuntimeError("incomplete OOF")
    PRED.mkdir(parents=True, exist_ok=True); ref.assign(prob_positive=p).to_csv(PRED / f"{name}_oof.csv", index=False)
    fit = model(seed); fit.fit(X, y); public = fit.predict_proba(XP)[:, 1]
    submission = pd.DataFrame({"user_id": users, "trial_id": trials, "prob. of positive emotion": public}); validate_submission(submission, [f"P_test{i}" for i in range(1, 11)]); submission.to_csv(PRED / f"{name}_public.csv", index=False)
    return summarize(y, p, ref, folds)


def main():
    if not CACHE.exists(): build_cache()
    d = load_cache(CACHE); keep = d["subject_ids"] != CONFIG["excluded_subject"]
    subjects, groups, y = d["subject_ids"][keep], d["health_groups"][keep], d["y_train"][keep]
    if len(y) != CONFIG["expected_oof_rows"]: raise RuntimeError("unexpected OOF rows")
    ref = pd.DataFrame({"subject_id": subjects, "health_group": groups, "original_trial_id": d["original_trial_ids"][keep], "y_true": y})
    outer = pd.read_csv((HERE / CONFIG["outer_reference"]).resolve()); folds = outer.fold.to_numpy(int)
    if not np.array_equal(outer.subject_id, subjects): raise RuntimeError("outer reference alignment mismatch")
    ref["fold"] = folds; assert_subject_isolation(subjects, folds)
    sx = normalize_by_subject(d["X_control"][keep], subjects, "zscore"); dx = normalize_by_subject(d["X_diffusion"][keep], subjects, "zscore")
    sp = normalize_by_subject(d["X_public_control"], d["public_user_ids"], "zscore"); dp = normalize_by_subject(d["X_public_diffusion"], d["public_user_ids"], "zscore")
    runs = {}; summary = []
    with threadpool_limits(limits=1):
        for name, X, XP in (("spectral_control", sx, sp), ("graph_diffusion", dx, dp)):
            r = evaluate(name, X, XP, y, ref, folds, d["public_user_ids"], d["public_trial_ids"], 2026); runs[name] = r
            summary.append({"run_name": name, "n_features": X.shape[1], "oof_auc": r["overall"]["auc"], "oof_f1": r["overall"]["f1"], "oof_balanced_accuracy": r["overall"]["balanced_accuracy"], "subject_auc_mean": r["subject_auc_mean"], "hc_auc": r["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": r["health_groups"]["DEP"]["pooled"]["auc"]})
        repeated = {}
        for seed in CONFIG["repeat_seeds"]:
            rf = folds_for(subjects, groups, seed); assert_subject_isolation(subjects, rf)
            for name, X, XP in (("spectral_control", sx, sp), ("graph_diffusion", dx, dp)):
                r = evaluate(f"{name}_seed{seed}", X, XP, y, ref, rf, d["public_user_ids"], d["public_trial_ids"], seed); repeated[f"{name}_seed{seed}"] = r
    runs.update(repeated)
    provenance = {"spectral_cache_sha256": hashlib.sha256(((HERE / CONFIG["spectral_cache"]).resolve()).read_bytes()).hexdigest(), "graph_cache_sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest(), "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
    result = {"experiment_config": CONFIG, "provenance": provenance, "protocol": {"subjectwise_unlabeled_zscore": True, "subject_local_cosine_knn": True, "cross_subject_edges": False, "labels_or_order_in_features": False, "fixed_002_outer_folds": True, "independent_repeats": True, "public_predictions_only": True}, "runs": runs}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    (HERE / "input_metadata.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)

if __name__ == "__main__": main()
