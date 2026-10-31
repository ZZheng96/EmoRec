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
OUT = HERE / "predictions"
OOF_COLUMNS = ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"]
CHANNEL_NAMES = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def load_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_cache_alignment(a, b):
    for key in ("subject_ids", "health_groups", "original_trial_ids", "y_train", "public_user_ids", "public_trial_ids"):
        if not np.array_equal(a[key], b[key]):
            raise RuntimeError(f"cache alignment mismatch: {key}")


def roi_indices(config):
    lookup = {name: i for i, name in enumerate(CHANNEL_NAMES)}
    rois = {}
    used = []
    for roi, names in config["roi_channels"].items():
        if len(set(names)) != len(names) or any(name not in lookup for name in names):
            raise ValueError(f"invalid or duplicate channels in {roi}")
        indices = [lookup[name] for name in names]
        if set(used) & set(indices):
            raise ValueError("ROI channel overlap")
        used.extend(indices); rois[roi] = indices
    if sorted(used) != list(range(30)):
        raise ValueError("ROI definitions do not cover all 30 channels")
    return rois


def build_roi_features(x, config):
    rois = roi_indices(config)
    a = np.asarray(x, dtype=float).reshape(len(x), 30, 5)
    chunks, names = [], []
    for roi, indices in rois.items():
        part = a[:, indices, :]
        for stat in ("mean", "std", "median"):
            values = getattr(np, stat)(part, axis=1)
            chunks.append(values)
            names.extend(f"roi_{roi}_{stat}_{band}" for band in BANDS)
    return np.concatenate(chunks, axis=1), names


def build_robust_features(x, subjects, trials, config):
    x = np.asarray(x, dtype=float)
    out = np.empty((len(x), 300), dtype=float)
    names = [f"trial_mad_{i:03d}" for i in range(150)] + [f"trial_signed_{i:03d}" for i in range(150)]
    for subject in np.unique(subjects):
        for trial in np.unique(trials[subjects == subject]):
            ix = np.flatnonzero((subjects == subject) & (trials == trial))
            if len(ix) != 5:
                raise ValueError("every subject/trial must have five contiguous windows")
            # Median and all deviations are derived solely from this trial's 5 spectral rows.
            values = x[ix, :150]
            center = np.median(values, axis=0)
            mad = np.median(np.abs(values - center), axis=0)
            signed = np.mean(values - center, axis=0)
            absolute = np.mean(np.abs(values - center), axis=0)
            out[ix] = np.repeat(np.concatenate([absolute, signed])[None, :], 5, axis=0)
    return out, names


def model(config, seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=seed))


def metrics(y, p):
    pred = p >= .5; tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": .5, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def write_oof(name, ref, p):
    out = ref.copy(); out["prob_positive"] = p; out.to_csv(OUT / f"{name}_oof.csv", index=False)


def summarize(y, p, ref, n_features):
    subjects = []
    for s in np.unique(ref.subject_id):
        mask = ref.subject_id.to_numpy() == s
        subjects.append({"subject_id": s, "health_group": ref.loc[mask, "health_group"].iloc[0], **metrics(y[mask], p[mask])})
    groups = {}
    for g in ("HC", "DEP"):
        mask = ref.health_group.to_numpy() == g; rows = [r for r in subjects if r["health_group"] == g]
        groups[g] = {"n_subjects": len(rows), "pooled": metrics(y[mask], p[mask]), "subject_auc_mean": float(np.mean([r["auc"] for r in rows]))}
    return {"n_features": n_features, "overall": metrics(y, p), "folds": [{"fold": f, **metrics(y[ref.fold == f], p[ref.fold == f])} for f in range(5)], "subject_auc_mean": float(np.mean([r["auc"] for r in subjects])), "health_groups": groups, "subjects": subjects}


def evaluate_single(name, X, XP, y, ref, n_features):
    p = np.full(len(y), np.nan)
    for fold in range(5):
        tr, va = ref.fold.to_numpy() != fold, ref.fold.to_numpy() == fold
        fitted = model(CONFIG, 2026 + fold); fitted.fit(X[tr], y[tr]); p[va] = fitted.predict_proba(X[va])[:, 1]
    write_oof(name, ref, p)
    fitted = model(CONFIG, 2026); fitted.fit(X, y)
    public = pd.DataFrame({"user_id": ref.attrs["public_user_ids"], "trial_id": ref.attrs["public_trial_ids"], "prob. of positive emotion": fitted.predict_proba(XP)[:, 1]})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)]); public.to_csv(OUT / f"{name}_public.csv", index=False)
    return p, summarize(y, p, ref, n_features)


def main():
    global CONFIG
    CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8")); OUT.mkdir(parents=True, exist_ok=True)
    spectral = load_cache((HERE / CONFIG["spectral_cache"]).resolve()); temporal = load_cache((HERE / CONFIG["temporal_cache"]).resolve()); assert_cache_alignment(spectral, temporal)
    keep = spectral["subject_ids"] != CONFIG["excluded_subject"]
    subjects, groups, y = spectral["subject_ids"][keep], spectral["health_groups"][keep], spectral["y_train"][keep]
    ref = pd.DataFrame({"subject_id": subjects, "health_group": groups, "original_trial_id": spectral["original_trial_ids"][keep], "fold": -1, "y_true": y})
    outer = pd.read_csv((HERE / CONFIG["outer_reference"]).resolve())
    if list(outer.columns) != OOF_COLUMNS or not np.array_equal(outer["subject_id"], ref.subject_id) or not np.array_equal(outer["original_trial_id"], ref.original_trial_id): raise RuntimeError("outer reference mismatch")
    ref["fold"] = outer["fold"].to_numpy(int); ref.attrs["public_user_ids"] = spectral["public_user_ids"]; ref.attrs["public_trial_ids"] = spectral["public_trial_ids"]
    if len(y) != 2760 or len(np.unique(subjects)) != 69: raise RuntimeError("unexpected filtered data")
    sx = normalize_by_subject(spectral["X_train"][keep, :150], subjects, "zscore"); sp = normalize_by_subject(spectral["X_public"][:, :150], spectral["public_user_ids"], "zscore")
    tx = normalize_by_subject(temporal["X_train"][keep], subjects, "zscore"); tp = normalize_by_subject(temporal["X_public"], temporal["public_user_ids"], "zscore")
    roi, roi_names = build_roi_features(sx, CONFIG); roi_p, _ = build_roi_features(sp, CONFIG)
    trial_windows = np.empty(len(subjects), dtype=int)
    for subject in np.unique(subjects):
        subject_rows = np.flatnonzero(subjects == subject)
        trial_windows[subject_rows] = np.arange(len(subject_rows)) // 5
    robust, robust_names = build_robust_features(sx, subjects, trial_windows, CONFIG)
    # Public cache has one 10-second row per trial, so the five-window trial deviation is unavailable.
    robust_p = np.zeros((len(sp), 300), dtype=float)

    runs, summary, preds = {}, [], {}
    with threadpool_limits(limits=1):
        for name, X, XP, n in [("spectral_control", sx, sp, 150), ("roi_static", roi, roi_p, len(roi_names)), ("robust_trial_deviation", robust, robust_p, 300), ("temporal_control", tx, tp, 450)]:
            p, result = evaluate_single(name, X, XP, y, ref, n); preds[name] = p; runs[name] = result
        preds["fusion_roi_temporal"] = .5 * preds["roi_static"] + .5 * preds["temporal_control"]
        write_oof("fusion_roi_temporal", ref, preds["fusion_roi_temporal"]); runs["fusion_roi_temporal"] = summarize(y, preds["fusion_roi_temporal"], ref, len(roi_names) + 450)
        a = model(CONFIG, 2026); b = model(CONFIG, 2026); a.fit(roi, y); b.fit(tx, y)
        public = pd.DataFrame({"user_id": spectral["public_user_ids"], "trial_id": spectral["public_trial_ids"], "prob. of positive emotion": .5 * a.predict_proba(roi_p)[:, 1] + .5 * b.predict_proba(tp)[:, 1]}); validate_submission(public, [f"P_test{i}" for i in range(1, 11)]); public.to_csv(OUT / "fusion_roi_temporal_public.csv", index=False)
    for name, result in runs.items():
        summary.append({"run_name": name, "n_features": result["n_features"], "auc": result["overall"]["auc"], "f1": result["overall"]["f1"], "balanced_accuracy": result["overall"]["balanced_accuracy"], "subject_auc_mean": result["subject_auc_mean"], "hc_auc": result["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": result["health_groups"]["DEP"]["pooled"]["auc"]})
    result = {"experiment_config": CONFIG, "provenance": {"spectral_cache_sha256": sha256((HERE / CONFIG["spectral_cache"]).resolve()), "temporal_cache_sha256": sha256((HERE / CONFIG["temporal_cache"]).resolve()), "outer_reference_sha256": sha256((HERE / CONFIG["outer_reference"]).resolve()), "channel_names_source": "src/emo_rec/features.py constant reproduced and checked", "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "threads": 1}, "protocol": {"cache_alignment_checked": True, "outer_folds_reused_from_002": True, "no_labels_in_feature_construction": True, "trial_groups_five_rows": True, "roi_expected_dimensions_note": "The four requested anatomical ROIs yield 4*5*3=60, not 75; no fifth ROI was invented.", "all_conditions_same_2760_rows": True}, "feature_names": {"roi_static": roi_names, "robust_trial_deviation": robust_names}, "runs": runs}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    report = "# 实验030：ROI频谱汇聚与trial内稳健偏差\n\n复用001的150维频谱缓存、009的450维temporal缓存和002固定五折，排除HC43；所有特征构建不使用标签，并按被试分别zscore。ROI通道严格采用项目`CHANNEL_NAMES`常量，四个请求ROI覆盖30通道且无重叠。注意：四个ROI×5频带×3统计量实际为60维，任务文字中的75维与给定四ROI划分不一致，本实验未虚构第五ROI。稳健偏差为每个被试每个原trial连续5行相对trial中位数的mean absolute deviation和median-centered signed deviation，各300维并逐窗重复。\n\n## 结果\n\n" + pd.DataFrame(summary).to_string(index=False) + "\n\nfusion_roi_temporal固定0.5/0.5，不根据OOF选权重。详细指标、折、被试诊断和哈希见metrics.json，公开集与2760行OOF预测见predictions。\n\n## 限制\n\n外层折沿用002，结果适合本项目内条件比较而非独立泛化估计；稳健特征按要求汇聚为trial常数后重复到五个窗口，未将逐窗原始偏差纳入模型；public trial分组依赖缓存稳定的8行/用户顺序。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8"); print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
