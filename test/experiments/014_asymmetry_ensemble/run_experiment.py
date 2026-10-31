import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emo_rec.features import ASYMMETRY_PAIRS, CHANNEL_NAMES, FREQUENCY_BANDS
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
CACHE = (EXPERIMENT_DIR / "../001_spectral_logpower/cache/features.npz").resolve()
BASELINE_DIR = (EXPERIMENT_DIR / "../002_subject_normalization").resolve()
AUX_DIRS = [(EXPERIMENT_DIR / "../004_labram_frozen").resolve(), (EXPERIMENT_DIR / "../006_frequency_graph").resolve()]


def load_cache():
    with np.load(CACHE) as cache:
        return {k: cache[k] for k in cache.files}


def folds(subjects, health, seed):
    unique = np.unique(subjects)
    labels = np.array([health[np.flatnonzero(subjects == s)[0]] for s in unique])
    splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
    result = {}
    for fold, (_, val) in enumerate(splitter.split(unique, labels)):
        for subject in unique[val]:
            result[subject] = fold
    return np.array([result[s] for s in subjects], dtype=int)


def asymmetry(X, names):
    names = list(names)
    band_count = len(FREQUENCY_BANDS)
    expected = len(CHANNEL_NAMES) * band_count
    base = X[:, :expected]
    channel_index = {channel: i for i, channel in enumerate(CHANNEL_NAMES)}
    differences = np.stack(
        [base[:, channel_index[left] * band_count:channel_index[left] * band_count + band_count] - base[:, channel_index[right] * band_count:channel_index[right] * band_count + band_count] for left, right in ASYMMETRY_PAIRS], axis=1
    ).reshape(len(X), -1)
    asym_names = np.array([f"asymmetry_{left}_{right}_{band}" for left, right in ASYMMETRY_PAIRS for band in FREQUENCY_BANDS])
    return np.concatenate([base, differences], axis=1), np.concatenate([np.array(names[:expected]), asym_names])


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, p)), "binary_f1": float(f1_score(y, pred, zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": 0.5, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def evaluate(name, X, X_public, data, fold_ids, config):
    y = data["y_train"]
    oof = np.full(len(y), np.nan)
    for fold in range(5):
        train, val = fold_ids != fold, fold_ids == fold
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=config["random_seed"]))
        model.fit(X[train], y[train])
        oof[val] = model.predict_proba(X[val])[:, 1]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=config["random_seed"]))
    model.fit(X, y)
    public_p = model.predict_proba(X_public)[:, 1]
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    frame = pd.DataFrame({"subject_id": data["subject_ids"], "health_group": data["health_groups"], "original_trial_id": data["original_trial_ids"], "fold": fold_ids, "y_true": y, "prob_positive": oof})
    frame.to_csv(PREDICTIONS_DIR / f"{name}_oof.csv", index=False)
    public = pd.DataFrame({"user_id": data["public_user_ids"], "trial_id": data["public_trial_ids"], "prob. of positive emotion": public_p})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
    public.to_csv(PREDICTIONS_DIR / f"{name}_public.csv", index=False)
    health = {}
    for group in ("HC", "DEP"):
        mask = data["health_groups"] == group
        health[group] = {"n_samples": int(mask.sum()), "pooled": metrics(y[mask], oof[mask])}
    return {"n_subjects": int(len(np.unique(data["subject_ids"]))), "n_samples": int(len(y)), "n_features": int(X.shape[1]), "overall": metrics(y, oof), "health_groups": health, "folds": [{"fold": f, **metrics(y[fold_ids == f], oof[fold_ids == f])} for f in range(5)], "oof": oof, "public": public_p}


def load_prediction(path, expected, y=None):
    frame = pd.read_csv(path)
    for col in ("subject_id", "original_trial_id", "fold"):
        if col in expected and not np.array_equal(frame[col].to_numpy(), expected[col]):
            raise RuntimeError(f"Prediction alignment mismatch: {path.name} {col}")
    if y is not None and not np.array_equal(frame["y_true"].to_numpy(), y):
        raise RuntimeError(f"Prediction labels mismatch: {path}")
    return frame["prob_positive"].to_numpy()


def main():
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    data = load_cache()
    keep = data["subject_ids"] != config["excluded_subject"]
    data = {k: v[keep] if k in ("X_train", "y_train", "subject_ids", "health_groups", "original_trial_ids") else v for k, v in data.items()}
    if len(np.unique(data["subject_ids"])) != 69:
        raise RuntimeError("Expected exactly 69 subjects after HC43 exclusion")
    fold_ids = folds(data["subject_ids"], data["health_groups"], config["random_seed"])
    names = data["feature_names"]
    X_combined, combined_names = asymmetry(data["X_train"], names)
    X_public, _ = asymmetry(data["X_public"], names)
    results = {"experiment_config": config, "protocol": {"subject_exclusion": "HC43", "n_subjects": 69, "seed": 2026, "folds": "subject-wise stratified 5-fold", "classifier_scaler": "fit on each training fold only", "fusion_weights_fixed": True, "public_labels_used": False}, "runs": {}}
    for normalization in ("raw", "subject_zscore"):
        X = normalize_by_subject(X_combined, data["subject_ids"], "raw" if normalization == "raw" else "zscore")
        X_public_n = normalize_by_subject(X_public, data["public_user_ids"], "raw" if normalization == "raw" else "zscore")
        result = evaluate(f"combined_{normalization}", X, X_public_n, data, fold_ids, config)
        results["runs"][f"combined_{normalization}"] = {k: v for k, v in result.items() if k not in ("oof", "public")}
    expected = {k: data[k] for k in ("subject_ids", "original_trial_ids")}
    baseline = load_prediction(BASELINE_DIR / "predictions/subject_zscore_exclude_hc43_oof.csv", expected, data["y_train"])
    aux = [load_prediction(directory / "predictions" / filename, expected, data["y_train"]) for directory, filename in zip(AUX_DIRS, ("labram_frozen_subject_zscore_oof.csv", "frequency_graph_zscore_oof.csv"))]
    fused = 0.9 * baseline + 0.1 * np.mean(aux, axis=0)
    fused_public_parts = [pd.read_csv(BASELINE_DIR / "predictions/subject_zscore_exclude_hc43_public.csv")["prob. of positive emotion"].to_numpy()]
    fused_public_parts += [pd.read_csv(directory / "predictions" / filename)["prob. of positive emotion"].to_numpy() for directory, filename in zip(AUX_DIRS, ("labram_frozen_subject_zscore_public.csv", "frequency_graph_zscore_public.csv"))]
    fused_public = 0.9 * fused_public_parts[0] + 0.1 * np.mean(fused_public_parts[1:], axis=0)
    fused_frame = pd.DataFrame({"subject_id": data["subject_ids"], "health_group": data["health_groups"], "original_trial_id": data["original_trial_ids"], "fold": fold_ids, "y_true": data["y_train"], "prob_positive": fused})
    fused_frame.to_csv(PREDICTIONS_DIR / "fixed_fusion_oof.csv", index=False)
    public = pd.DataFrame({"user_id": data["public_user_ids"], "trial_id": data["public_trial_ids"], "prob. of positive emotion": fused_public})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
    public.to_csv(PREDICTIONS_DIR / "fixed_fusion_public.csv", index=False)
    results["runs"]["fixed_fusion_0.9_0.1"] = {"n_subjects": 69, "n_samples": len(data["y_train"]), "n_features": None, "overall": metrics(data["y_train"], fused), "health_groups": {g: {"n_samples": int((data["health_groups"] == g).sum()), "pooled": metrics(data["y_train"][data["health_groups"] == g], fused[data["health_groups"] == g])} for g in ("HC", "DEP")}, "weights": {"002_subject_zscore": 0.9, "004_006_auxiliary_mean": 0.1}, "auxiliary_sources": ["004_labram_frozen", "006_frequency_graph"]}
    rows = [{"run_name": n, "oof_auc": r["overall"]["roc_auc"], "oof_binary_f1": r["overall"]["binary_f1"], "hc_auc": r["health_groups"]["HC"]["pooled"]["roc_auc"], "dep_auc": r["health_groups"]["DEP"]["pooled"]["roc_auc"]} for n, r in results["runs"].items()]
    results["baseline_comparison"] = {"auc_threshold": config["baseline_auc"], "f1_threshold": config["baseline_f1"], "runs_exceeding_auc": [r["run_name"] for r in rows if r["oof_auc"] > config["baseline_auc"]], "runs_exceeding_f1": [r["run_name"] for r in rows if r["oof_binary_f1"] > config["baseline_f1"]]}
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
