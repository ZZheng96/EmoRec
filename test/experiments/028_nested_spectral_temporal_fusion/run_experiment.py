import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "predictions"
OOF_COLUMNS = ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"]


def load_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_cache_alignment(a, b):
    for key in ("subject_ids", "health_groups", "original_trial_ids", "y_train", "public_user_ids", "public_trial_ids"):
        if not np.array_equal(a[key], b[key]):
            raise RuntimeError(f"cache alignment mismatch: {key}")


def load_outer_reference(path, reference):
    frame = pd.read_csv(path)
    if list(frame.columns) != OOF_COLUMNS:
        raise RuntimeError("unexpected outer reference columns")
    for key in ("subject_id", "health_group", "original_trial_id", "y_true"):
        if not np.array_equal(frame[key].to_numpy(), reference[key].to_numpy()):
            raise RuntimeError(f"outer metadata mismatch: {key}")
    return frame["fold"].to_numpy(dtype=int)


def metrics(y, p):
    pred = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": 0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def model(config, seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=seed))


def select_weight(y, spectral, temporal, subjects, weights, seed, inner_folds=3):
    unique = np.unique(subjects)
    labels = np.asarray([y[np.flatnonzero(subjects == s)[0]] for s in unique])
    splitter = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    ps, pt = np.full(len(y), np.nan), np.full(len(y), np.nan)
    inner_assignment = np.full(len(y), -1, dtype=int)
    for inner, (tr, va) in enumerate(splitter.split(unique, labels)):
        train_subjects, valid_subjects = unique[tr], unique[va]
        train = np.isin(subjects, train_subjects)
        valid = np.isin(subjects, valid_subjects)
        ms, mt = model(CONFIG, seed), model(CONFIG, seed)
        ms.fit(spectral[train], y[train]); mt.fit(temporal[train], y[train])
        ps[valid] = ms.predict_proba(spectral[valid])[:, 1]
        pt[valid] = mt.predict_proba(temporal[valid])[:, 1]
        inner_assignment[valid] = inner
    if not np.isfinite(ps).all() or not np.isfinite(pt).all():
        raise RuntimeError("incomplete inner OOF")
    scores = []
    for w in weights:
        p = w * ps + (1 - w) * pt
        m = metrics(y, p)
        scores.append({"weight": float(w), "auc": m["auc"], "f1": m["f1"]})
    chosen = max(scores, key=lambda r: (r["auc"], r["f1"], r["weight"]))
    return chosen["weight"], scores, ps, pt, inner_assignment


def evaluate_single(name, X, X_public, y, ref, folds, config):
    p = np.full(len(y), np.nan); fold_rows = []
    for fold in range(5):
        tr, va = folds != fold, folds == fold
        fitted = model(config, 2026 + fold); fitted.fit(X[tr], y[tr]); p[va] = fitted.predict_proba(X_public if False else X[va])[:, 1]
        fold_rows.append({"fold": fold, "n_train_subjects": int(ref.loc[tr, "subject_id"].nunique()), "n_validation_subjects": int(ref.loc[va, "subject_id"].nunique()), **metrics(y[va], p[va])})
    write_oof(name, ref, p)
    fitted = model(config, 2026); fitted.fit(X, y)
    public = pd.DataFrame({"user_id": ref.attrs["public_user_ids"], "trial_id": ref.attrs["public_trial_ids"], "prob. of positive emotion": fitted.predict_proba(X_public)[:, 1]})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)]); public.to_csv(OUT / f"{name}_public.csv", index=False)
    return summarize(y, p, ref, fold_rows)


def write_oof(name, ref, p):
    out = ref.copy(); out["prob_positive"] = p; out.to_csv(OUT / f"{name}_oof.csv", index=False)


def summarize(y, p, ref, fold_rows, extra=None):
    subject_rows = []
    for s in np.unique(ref["subject_id"]):
        mask = ref["subject_id"].to_numpy() == s
        subject_rows.append({"subject_id": s, "health_group": ref.loc[mask, "health_group"].iloc[0], **metrics(y[mask], p[mask])})
    groups = {}
    for g in ("HC", "DEP"):
        mask = ref["health_group"].to_numpy() == g; rows = [r for r in subject_rows if r["health_group"] == g]
        groups[g] = {"n_subjects": len(rows), "pooled": metrics(y[mask], p[mask]), "subject_auc_mean": float(np.mean([r["auc"] for r in rows]))}
    result = {"overall": metrics(y, p), "folds": fold_rows, "health_groups": groups, "subject_auc_mean": float(np.mean([r["auc"] for r in subject_rows])), "subjects": subject_rows}
    if extra: result.update(extra)
    return result


def main():
    global CONFIG
    CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    spectral = load_cache((HERE / CONFIG["spectral_cache"]).resolve()); temporal = load_cache((HERE / CONFIG["temporal_cache"]).resolve())
    assert_cache_alignment(spectral, temporal)
    keep = spectral["subject_ids"] != CONFIG["excluded_subject"]
    subjects, groups, y = spectral["subject_ids"][keep], spectral["health_groups"][keep], spectral["y_train"][keep]
    ref = pd.DataFrame({"subject_id": subjects, "health_group": groups, "original_trial_id": spectral["original_trial_ids"][keep], "fold": -1, "y_true": y})
    folds = load_outer_reference((HERE / CONFIG["outer_reference"]).resolve(), ref); ref["fold"] = folds
    if len(y) != 2760 or len(np.unique(subjects)) != 69 or np.any(np.isin(["HC43"], subjects)): raise RuntimeError("unexpected filtered data")
    ref.attrs["public_user_ids"], ref.attrs["public_trial_ids"] = spectral["public_user_ids"], spectral["public_trial_ids"]
    sx = normalize_by_subject(spectral["X_train"][keep, :150], subjects, "zscore"); tx = normalize_by_subject(temporal["X_train"][keep], subjects, "zscore")
    sp = normalize_by_subject(spectral["X_public"][:, :150], spectral["public_user_ids"], "zscore"); tp = normalize_by_subject(temporal["X_public"], temporal["public_user_ids"], "zscore")
    runs = {}; summary = []; selection_rows = []; inner_rows = []; outer_s, outer_t = np.full(len(y), np.nan), np.full(len(y), np.nan); fusion = np.full(len(y), np.nan); fixed = np.full(len(y), np.nan)
    with threadpool_limits(limits=1):
        for fold in range(5):
            tr, va = folds != fold, folds == fold
            w, scores, inner_s, inner_t, inner_assignment = select_weight(y[tr], sx[tr], tx[tr], subjects[tr], CONFIG["weights"], 2026 + fold)
            train_indices = np.flatnonzero(tr)
            for local_i, global_i in enumerate(train_indices):
                inner_rows.append({"outer_fold": fold, "inner_fold": int(inner_assignment[local_i]), "subject_id": subjects[global_i], "health_group": groups[global_i], "original_trial_id": int(spectral["original_trial_ids"][keep][global_i]), "y_true": int(y[global_i]), "spectral_prob": float(inner_s[local_i]), "temporal_prob": float(inner_t[local_i]), "selected_weight": float(w), "fusion_prob": float(w * inner_s[local_i] + (1 - w) * inner_t[local_i])})
            selection_rows.append({"outer_fold": fold, "selected_weight": w, "inner_seed": 2026 + fold, "inner_subjects": int(len(np.unique(subjects[tr]))), "inner_oof_rows": int(tr.sum()), "candidate_scores": json.dumps(scores, separators=(",", ":"))})
            ms, mt = model(CONFIG, 2026 + fold), model(CONFIG, 2026 + fold); ms.fit(sx[tr], y[tr]); mt.fit(tx[tr], y[tr])
            outer_s[va] = ms.predict_proba(sx[va])[:, 1]; outer_t[va] = mt.predict_proba(tx[va])[:, 1]; fusion[va] = w * outer_s[va] + (1 - w) * outer_t[va]; fixed[va] = .5 * outer_s[va] + .5 * outer_t[va]
            print(f"outer fold {fold + 1}/5 selected w={w:.2f}", flush=True)
    for name, p in (("spectral_control", outer_s), ("temporal_control", outer_t), ("nested_fusion", fusion), ("fixed_05_05", fixed)):
        write_oof(name, ref, p)
        fold_rows = [{"fold": f, **metrics(y[folds == f], p[folds == f])} for f in range(5)]
        runs[name] = summarize(y, p, ref, fold_rows, {"n_features": 150 if name == "spectral_control" else 450 if name == "temporal_control" else 2, "selected_weights": [r["selected_weight"] for r in selection_rows] if name == "nested_fusion" else None})
        summary.append({"run_name": name, "auc": runs[name]["overall"]["auc"], "f1": runs[name]["overall"]["f1"], "balanced_accuracy": runs[name]["overall"]["balanced_accuracy"], "subject_auc_mean": runs[name]["subject_auc_mean"], "hc_auc": runs[name]["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": runs[name]["health_groups"]["DEP"]["pooled"]["auc"]})
    mean_w = float(np.mean([r["selected_weight"] for r in selection_rows]))
    for name, X, XP, p in (("spectral_control", sx, sp, None), ("temporal_control", tx, tp, None), ("nested_fusion", np.concatenate([sx, tx], 1), np.concatenate([sp, tp], 1), mean_w), ("fixed_05_05", np.concatenate([sx, tx], 1), np.concatenate([sp, tp], 1), .5)):
        if p is None:
            fitted = model(CONFIG, 2026); fitted.fit(X, y); public_p = fitted.predict_proba(XP)[:, 1]
        else:
            a, b = model(CONFIG, 2026), model(CONFIG, 2026); a.fit(sx, y); b.fit(tx, y); public_p = p * a.predict_proba(sp)[:, 1] + (1 - p) * b.predict_proba(tp)[:, 1]
        pd.DataFrame({"user_id": spectral["public_user_ids"], "trial_id": spectral["public_trial_ids"], "prob. of positive emotion": public_p}).to_csv(OUT / f"{name}_public.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(OUT / "nested_weight_selection_oof.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(OUT / "nested_inner_oof.csv", index=False)
    result = {"experiment_config": CONFIG, "provenance": {"spectral_cache_sha256": sha256((HERE / CONFIG["spectral_cache"]).resolve()), "temporal_cache_sha256": sha256((HERE / CONFIG["temporal_cache"]).resolve()), "outer_reference_sha256": sha256((HERE / CONFIG["outer_reference"]).resolve()), "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "threads": 1}, "protocol": {"cache_alignment_checked": True, "outer_folds_reused_from_002": True, "inner_subject_isolation": True, "no_existing_oof_used": True, "public_mean_selected_weight": mean_w}, "runs": runs}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    report = "# 实验028：严格被试嵌套的 spectral-HGB 与 temporal-HGB 概率融合\n\n严格复用001前150维log_bandpower、009的450维时序缓存、002固定5个外层被试折并排除HC43。两输入均按被试无标签zscore；HGB固定参数为max_iter=200、learning_rate=.03、max_leaf_nodes=15、min_samples_leaf=20、l2=1、early_stopping=False，外层seed=2026+fold，单线程。每个外层训练集内重新做3折被试级StratifiedKFold，两个HGB产生严格inner OOF，仅按AUC、F1、spectral优先选择w。外层验证只预测一次并保存四类结果。\n\n## 结果\n\n|run|AUC|F1|balanced accuracy|subject macro AUC|HC AUC|DEP AUC|\n|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"|{r['run_name']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['balanced_accuracy']:.6f}|{r['subject_auc_mean']:.6f}|{r['hc_auc']:.6f}|{r['dep_auc']:.6f}|" for r in summary) + f"\n\n各外层选择权重：{[r['selected_weight'] for r in selection_rows]}；公开集使用完整训练集重训，nested_fusion采用上述权重均值{mean_w:.4f}。\n\n## 解释边界\n\n这些是固定外层折上的模型选择结果，不是嵌套外层性能的无偏估计，不能宣称无偏；外层折复用也意味着比较存在选择偏差。权重选择未读取既有OOF概率，所有inner概率均在本次训练中重新产生。详细折、被试、HC/DEP及provenance见metrics.json，候选权重与inner OOF选择记录见predictions/nested_weight_selection_oof.csv。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8"); print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
