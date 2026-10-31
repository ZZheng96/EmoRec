import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emo_rec.data import load_public_test_segments, load_train_segments
from emo_rec.features import extract_spectral_features
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[2]
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
OOF_COLUMNS = ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"]
META_KEYS = ("subject_ids", "health_groups", "original_trial_ids", "y_train")


def load_cache(path):
    with np.load(path) as cache:
        return {key: cache[key] for key in cache.files}


def raw_multiresolution_features(segments):
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (30, 2500):
        raise ValueError("Expected 10-second segments with shape (n, 30, 2500)")
    windows = segments.reshape(len(segments), 30, 5, 500).transpose(0, 2, 1, 3).reshape(-1, 30, 500)
    features, names = extract_spectral_features(windows)
    mask = np.asarray([name.startswith("log_bandpower_") for name in names])
    return features[:, mask].reshape(len(segments), 750), np.asarray([f"window{w}_{n}" for w in range(5) for n in np.asarray(names)[mask]])


def build_feature_cache(path):
    train_x, labels, subjects, groups, trials = [], [], [], [], []
    feature_names = None
    files = sorted((ROOT / "数据" / "训练集").glob("*/*.mat"))
    for index, file in enumerate(files, 1):
        segments, y, trial_ids = load_train_segments(file)
        x, feature_names = raw_multiresolution_features(segments)
        train_x.append(x); labels.append(y); subjects.extend([file.stem] * len(x))
        groups.extend(["DEP" if file.parent.name == "抑郁症患者" else "HC"] * len(x)); trials.append(trial_ids)
        print(f"Features: {index}/{len(files)} {file.stem}", flush=True)
    public_x, public_users, public_trials = [], [], []
    files = sorted((ROOT / "数据" / "公开测试集").glob("*.mat"), key=lambda p: int(p.stem.replace("P_test", "")))
    for file in files:
        x, _ = raw_multiresolution_features(load_public_test_segments(file))
        public_x.append(x); public_users.extend([file.stem] * len(x)); public_trials.extend(np.arange(1, 9))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X_train=np.concatenate(train_x), y_train=np.concatenate(labels), subject_ids=np.asarray(subjects),
                        health_groups=np.asarray(groups), original_trial_ids=np.concatenate(trials), feature_names=feature_names,
                        X_public=np.concatenate(public_x), public_user_ids=np.asarray(public_users), public_trial_ids=np.asarray(public_trials))


def assert_cache_alignment(*caches):
    for key in META_KEYS + ("public_user_ids", "public_trial_ids"):
        if not all(np.array_equal(caches[0][key], cache[key]) for cache in caches[1:]):
            raise RuntimeError(f"cache alignment mismatch: {key}")


def load_outer_reference(path, reference):
    frame = pd.read_csv(path)
    if list(frame.columns) != OOF_COLUMNS:
        raise RuntimeError("Unexpected outer reference columns")
    for key in ("subject_id", "health_group", "original_trial_id", "y_true"):
        if not np.array_equal(frame[key].to_numpy(), reference[key].to_numpy()):
            raise RuntimeError(f"outer metadata alignment mismatch: {key}")
    return frame["fold"].to_numpy(dtype=int)


def assert_subject_isolation(subjects, folds):
    for fold in range(5):
        if set(subjects[folds != fold]) & set(subjects[folds == fold]):
            raise RuntimeError(f"subject leakage in fold {fold}")


def model(config, seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=seed))


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, p)), "binary_f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": 0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def subject_macro_auc(y, p, subjects):
    values = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        if len(np.unique(y[mask])) == 2:
            values.append(roc_auc_score(y[mask], p[mask]))
    return float(np.mean(values)) if values else float("nan")


def evaluate(name, X, X_public, reference, folds, config):
    y = reference.y_true.to_numpy(); subjects = reference.subject_id.to_numpy(); oof = np.full(len(y), np.nan); fold_rows = []
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        fitted = model(config, config["random_seed"] + fold); fitted.fit(X[train], y[train]); oof[valid] = fitted.predict_proba(X[valid])[:, 1]
        row = {"fold": fold, "overall": metrics(y[valid], oof[valid]), "subject_macro_auc": subject_macro_auc(y[valid], oof[valid], subjects[valid]), "n_train_subjects": int(reference.loc[train, "subject_id"].nunique()), "n_validation_subjects": int(reference.loc[valid, "subject_id"].nunique())}
        for group in ("HC", "DEP"):
            mask = valid & (reference.health_group.to_numpy() == group); row[group] = metrics(y[mask], oof[mask])
        fold_rows.append(row)
    if not np.isfinite(oof).all(): raise RuntimeError(f"incomplete OOF predictions for {name}")
    output = reference.copy(); output["prob_positive"] = oof; output.to_csv(PREDICTIONS_DIR / f"{name}_oof.csv", index=False)
    fitted = model(config, config["random_seed"]); fitted.fit(X, y)
    public = pd.DataFrame({"user_id": reference.attrs["public_user_ids"], "trial_id": reference.attrs["public_trial_ids"], "prob. of positive emotion": fitted.predict_proba(X_public)[:, 1]})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)]); public.to_csv(PREDICTIONS_DIR / f"{name}_public.csv", index=False)
    groups = {g: metrics(y[reference.health_group.to_numpy() == g], oof[reference.health_group.to_numpy() == g]) for g in ("HC", "DEP")}
    return {"n_features": int(X.shape[1]), "overall": metrics(y, oof), "subject_macro_auc": subject_macro_auc(y, oof, subjects), "groups": groups, "folds": fold_rows}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--rebuild-features", action="store_true"); args = parser.parse_args()
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8")); PREDICTIONS_DIR.mkdir(exist_ok=True)
    spectral = load_cache(EXPERIMENT_DIR / config["spectral_cache"]); temporal = load_cache(EXPERIMENT_DIR / config["temporal_cache"])
    multi_path = EXPERIMENT_DIR / config["multiresolution_cache"]
    if args.rebuild_features or not multi_path.exists(): build_feature_cache(multi_path)
    multi = load_cache(multi_path); assert_cache_alignment(spectral, temporal, multi)
    if spectral["X_train"].shape[1] < 150 or temporal["X_train"].shape[1] != 450 or multi["X_train"].shape[1] != 750: raise RuntimeError("unexpected feature dimensions")
    keep = spectral["subject_ids"] != config["excluded_subject"]
    reference = pd.DataFrame({"subject_id": spectral["subject_ids"][keep], "health_group": spectral["health_groups"][keep], "original_trial_id": spectral["original_trial_ids"][keep], "fold": -1, "y_true": spectral["y_train"][keep]})
    if len(reference) != config["expected_train_rows"] or reference.subject_id.nunique() != config["expected_subjects"]: raise RuntimeError("unexpected HC43 exclusion result")
    folds = load_outer_reference(EXPERIMENT_DIR / config["outer_reference"], reference); reference["fold"] = folds; assert_subject_isolation(reference.subject_id.to_numpy(), folds)
    reference.attrs["public_user_ids"] = spectral["public_user_ids"]; reference.attrs["public_trial_ids"] = spectral["public_trial_ids"]
    subjects = reference.subject_id.to_numpy(); users = spectral["public_user_ids"]
    sx = normalize_by_subject(spectral["X_train"][keep, :150], subjects, "zscore", 1e-12); sp = normalize_by_subject(spectral["X_public"][:, :150], users, "zscore", 1e-12)
    tx = normalize_by_subject(temporal["X_train"][keep], subjects, "zscore", 1e-12); tp = normalize_by_subject(temporal["X_public"], users, "zscore", 1e-12)
    mx = normalize_by_subject(multi["X_train"][keep], subjects, "zscore", 1e-12); mp = normalize_by_subject(multi["X_public"], users, "zscore", 1e-12)
    conditions = {"spectral150": (sx, sp), "temporal450": (tx, tp), "multiresolution750_raw": (mx, mp), "concat900": (np.concatenate([sx, mx], 1), np.concatenate([sp, mp], 1))}
    runs = {name: evaluate(name, x, p, reference, folds, config) for name, (x, p) in conditions.items()}
    summary = [{"run_name": n, "n_features": r["n_features"], "oof_auc": r["overall"]["roc_auc"], "oof_binary_f1": r["overall"]["binary_f1"], "subject_macro_auc": r["subject_macro_auc"], "hc_auc": r["groups"]["HC"]["roc_auc"], "dep_auc": r["groups"]["DEP"]["roc_auc"]} for n, r in runs.items()]
    result = {"experiment_config": config, "protocol": {"cache_alignment_checked": True, "excluded_subject": "HC43", "outer_folds_reused_from_002": True, "subject_isolation_checked": True, "no_tuning_fusion_or_validation_labels": True, "normalization": "subject-wise unlabeled zscore"}, "runs": runs}
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); pd.DataFrame(summary).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    report = "# 实验023：多分辨率短时频谱 HGB\n\n本实验从原始EEG每个10秒段切分5个2秒子窗，逐窗调用现有 `extract_spectral_features`，仅保留150维log-bandpower并拼接为750维。控制组复用001频谱和009时序缓存；先核验全部元数据逐行一致，再排除HC43并复用002固定5折。所有条件均采用按被试无标签zscore、StandardScaler及实验016/017固定HGB参数，不调参、不使用验证标签。\n\n## 实际结果\n\n|条件|OOF AUC|OOF F1|被试宏AUC|HC AUC|DEP AUC|\n|---|---:|---:|---:|---:|---:|\n" + "\n".join(f"|{r['run_name']}|{r['oof_auc']:.6f}|{r['oof_binary_f1']:.6f}|{r['subject_macro_auc']:.6f}|{r['hc_auc']:.6f}|{r['dep_auc']:.6f}|" for r in summary) + "\n\n每组输出2760行OOF和80行公开集预测；每折指标详见metrics.json。局限：固定折复用可能带来模型选择偏差；公开集无标签，不能计算泛化指标；仅比较单一固定HGB协议。\n"
    (EXPERIMENT_DIR / "实验报告.md").write_text(report, encoding="utf-8"); print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__": main()
