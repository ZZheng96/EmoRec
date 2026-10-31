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

from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[2]
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
OOF_COLUMNS = ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"]
META_KEYS = ("subject_ids", "health_groups", "original_trial_ids", "y_train")


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, p)), "binary_f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": 0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def load_cache(path):
    with np.load(path) as cache:
        return {key: cache[key] for key in cache.files}


def assert_cache_alignment(spectral, temporal):
    for key in META_KEYS:
        if not np.array_equal(spectral[key], temporal[key]):
            raise RuntimeError(f"cache alignment mismatch: {key}")
    for key in ("public_user_ids", "public_trial_ids"):
        if not np.array_equal(spectral[key], temporal[key]):
            raise RuntimeError(f"public cache alignment mismatch: {key}")


def load_outer_reference(path, reference):
    frame = pd.read_csv(path)
    if list(frame.columns) != OOF_COLUMNS:
        raise RuntimeError("Unexpected outer reference columns")
    for key in ("subject_id", "health_group", "original_trial_id", "y_true"):
        if not np.array_equal(frame[key].to_numpy(), reference[key].to_numpy()):
            raise RuntimeError(f"outer metadata alignment mismatch: {key}")
    return frame["fold"].to_numpy(dtype=int)


def model(config, seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=seed))


def assert_subject_isolation(subjects, folds):
    for fold in range(5):
        train_subjects = set(subjects[folds != fold])
        valid_subjects = set(subjects[folds == fold])
        if train_subjects & valid_subjects:
            raise RuntimeError(f"subject leakage in fold {fold}")


def write_oof(name, reference, probabilities):
    output = reference.copy()
    output["prob_positive"] = probabilities
    output.to_csv(PREDICTIONS_DIR / f"{name}_oof.csv", index=False)


def evaluate(name, X, X_public, reference, folds, config):
    y = reference["y_true"].to_numpy()
    oof = np.full(len(y), np.nan)
    fold_rows = []
    for fold in range(config["outer_folds"]):
        train, valid = folds != fold, folds == fold
        fitted = model(config, config["random_seed"] + fold)
        fitted.fit(X[train], y[train])
        oof[valid] = fitted.predict_proba(X[valid])[:, 1]
        row = {"fold": fold, "n_train_subjects": int(reference.loc[train, "subject_id"].nunique()),
               "n_validation_subjects": int(reference.loc[valid, "subject_id"].nunique()),
               "overall": metrics(y[valid], oof[valid])}
        for group in ("HC", "DEP"):
            mask = valid & (reference["health_group"].to_numpy() == group)
            row[group] = metrics(y[mask], oof[mask])
        fold_rows.append(row)
    if not np.isfinite(oof).all():
        raise RuntimeError(f"incomplete OOF predictions for {name}")
    write_oof(name, reference, oof)
    fitted = model(config, config["random_seed"])
    fitted.fit(X, y)
    public = pd.DataFrame({"user_id": reference.attrs["public_user_ids"], "trial_id": reference.attrs["public_trial_ids"],
                           "prob. of positive emotion": fitted.predict_proba(X_public)[:, 1]})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
    public.to_csv(PREDICTIONS_DIR / f"{name}_public.csv", index=False)
    group_rows = {group: metrics(y[reference["health_group"].to_numpy() == group], oof[reference["health_group"].to_numpy() == group]) for group in ("HC", "DEP")}
    return {"n_features": int(X.shape[1]), "overall": metrics(y, oof), "groups": group_rows, "folds": fold_rows}


def main():
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    np.seterr(all="raise")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    spectral = load_cache((EXPERIMENT_DIR / config["spectral_cache"]).resolve())
    temporal = load_cache((EXPERIMENT_DIR / config["temporal_cache"]).resolve())
    assert_cache_alignment(spectral, temporal)
    if spectral["X_train"].shape[1] < 150 or temporal["X_train"].shape[1] != 450:
        raise RuntimeError("unexpected feature dimensions")
    keep = spectral["subject_ids"] != config["excluded_subject"]
    if keep.sum() != 2760 or len(np.unique(spectral["subject_ids"][keep])) != config["expected_subjects"]:
        raise RuntimeError("unexpected HC43 exclusion result")
    subject_ids, groups = spectral["subject_ids"][keep], spectral["health_groups"][keep]
    reference = pd.DataFrame({"subject_id": subject_ids, "health_group": groups,
                              "original_trial_id": spectral["original_trial_ids"][keep], "fold": -1,
                              "y_true": spectral["y_train"][keep]})
    folds = load_outer_reference((EXPERIMENT_DIR / config["outer_reference"]).resolve(), reference)
    reference["fold"] = folds
    assert_subject_isolation(subject_ids, folds)
    reference.attrs["public_user_ids"] = spectral["public_user_ids"]
    reference.attrs["public_trial_ids"] = spectral["public_trial_ids"]
    spectral_X = normalize_by_subject(spectral["X_train"][keep, :150], subject_ids, "zscore", 1e-12)
    temporal_X = normalize_by_subject(temporal["X_train"][keep], subject_ids, "zscore", 1e-12)
    spectral_public = normalize_by_subject(spectral["X_public"][:, :150], spectral["public_user_ids"], "zscore", 1e-12)
    temporal_public = normalize_by_subject(temporal["X_public"], temporal["public_user_ids"], "zscore", 1e-12)
    conditions = {"spectral150": (spectral_X, spectral_public), "temporal450": (temporal_X, temporal_public),
                  "concat600": (np.concatenate([spectral_X, temporal_X], axis=1), np.concatenate([spectral_public, temporal_public], axis=1))}
    runs = {name: evaluate(name, X, X_public, reference, folds, config) for name, (X, X_public) in conditions.items()}
    summary = [{"run_name": name, "n_features": result["n_features"], "oof_auc": result["overall"]["roc_auc"],
                "oof_binary_f1": result["overall"]["binary_f1"], "hc_auc": result["groups"]["HC"]["roc_auc"],
                "dep_auc": result["groups"]["DEP"]["roc_auc"]} for name, result in runs.items()]
    result = {"experiment_config": config, "protocol": {"cache_alignment_checked": True, "outer_folds_reused_from_002": True,
        "excluded_subject": "HC43", "target_propagation": "unlabeled subject-wise zscore; labels only train HGB", "trial_unit": "10 seconds",
        "subject_isolation_checked": True, "no_tuning_fusion_or_threshold": True, "threshold": 0.5}, "runs": runs}
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    report = """# 实验017：Temporal HGB

## 目的与严格协议

本实验比较固定的 `spectral150`、`temporal450`、`concat600` 三个输入，模型均为实验016的固定 HistGradientBoostingClassifier 参数：200 iterations、learning_rate=0.03、15 leaves、min_samples_leaf=20、l2=1、early_stopping=False，随机种子为 `2026 + outer_fold`。不调参、不融合、不调阈值，阈值固定0.5。

时序缓存来自10秒试次的5个2秒子窗，含450维 mean/std/slope。两缓存先逐行核验 subject_ids、health_groups、original_trial_ids、y_train 及公开集 user/trial id 完全一致，再排除 HC43。沿用002的原始5个外层折，并断言每折被试隔离；zscore为按被试的无标签传导，标签只用于模型训练和评估。

## 结果

详见 `summary.csv`、`metrics.json` 及 `predictions/`。报告指标为总体、各外层折及HC/DEP分组的OOF ROC AUC、F1（另记录balanced accuracy）。与016频谱控制的复现差异仅为：本实验显式比较纯频谱150维、时序450维和拼接600维，三者统一沿用002外层折和相同固定HGB，不使用016的LR/融合路径。目标AUC=0.95仅作为预设目标，若实际未达则如实记录。
"""
    report += "\n\n| 输入 | OOF AUC | OOF F1 |\n|---|---:|---:|\n"
    for row in summary:
        report += f"| {row['run_name']} | {row['oof_auc']:.6f} | {row['oof_binary_f1']:.6f} |\n"
    report += "\n拼接特征提高AUC但降低F1，未同时改进两项指标，AUC与F1均未达到0.95。重复使用相同外层折的实验比较存在模型选择偏差，需要独立划分复核。\n"
    (EXPERIMENT_DIR / "实验报告.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
