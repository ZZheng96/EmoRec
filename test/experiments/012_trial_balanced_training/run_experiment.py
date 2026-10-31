from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
CACHE_PATH = (EXPERIMENT_DIR.parent / "001_spectral_logpower" / "cache" / "features.npz").resolve()
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
SEGMENTS_PER_TRIAL = 5
SEED = 2026
N_SPLITS = 5
THRESHOLD = 0.5


def load_cache() -> dict[str, np.ndarray]:
    with np.load(CACHE_PATH) as cache:
        return {key: cache[key] for key in cache.files}


def make_subject_folds(subject_ids: np.ndarray, health_groups: np.ndarray) -> np.ndarray:
    subjects = np.unique(subject_ids)
    subject_health = np.asarray([health_groups[np.flatnonzero(subject_ids == s)[0]] for s in subjects])
    splitter = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    fold_by_subject = {}
    for fold, (_, valid) in enumerate(splitter.split(subjects, subject_health)):
        for subject in subjects[valid]:
            fold_by_subject[subject] = fold
    return np.asarray([fold_by_subject[s] for s in subject_ids], dtype=np.int64)


def select_log_bandpower(X: np.ndarray, names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray([str(name).startswith("log_bandpower_") for name in names])
    return X[:, mask], names[mask]


def make_model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, penalty="l2", max_iter=2000, solver="lbfgs", random_state=SEED),
    )


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "binary_f1": float(f1_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "threshold": THRESHOLD, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def trial_group_columns() -> list[str]:
    return ["subject_id", "health_group", "original_trial_id", "y_true"]


def aggregate_trials(oof: pd.DataFrame) -> pd.DataFrame:
    groups = oof.groupby(trial_group_columns(), sort=False)
    sizes = groups.size()
    if not (sizes == SEGMENTS_PER_TRIAL).all():
        raise ValueError(f"Each trial must have exactly 5 segments: {sizes.value_counts().to_dict()}")
    if (groups["fold"].nunique() != 1).any() or (groups["y_true"].nunique() != 1).any():
        raise ValueError("Trial labels and folds must be consistent")
    result = groups.agg(
        fold=("fold", "first"), n_segments=("prob_positive", "size"),
        prob_positive=("prob_positive", "mean"),
    ).reset_index()
    return result[[*trial_group_columns(), "fold", "n_segments", "prob_positive"]]


def run_condition(name: str, X: np.ndarray, X_public: np.ndarray, data: dict[str, np.ndarray], folds: np.ndarray, weight: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    y = data["y_train"]
    oof_probability = np.full(len(y), np.nan)
    for fold in range(N_SPLITS):
        train_mask = folds != fold
        valid_mask = folds == fold
        model = make_model()
        model.fit(X[train_mask], y[train_mask], logisticregression__sample_weight=np.full(train_mask.sum(), weight))
        oof_probability[valid_mask] = model.predict_proba(X[valid_mask])[:, 1]
    if not np.isfinite(oof_probability).all():
        raise RuntimeError(f"{name}: incomplete OOF")
    oof = pd.DataFrame({
        "subject_id": data["subject_ids"], "health_group": data["health_groups"],
        "original_trial_id": data["original_trial_ids"], "fold": folds,
        "y_true": y, "prob_positive": oof_probability,
    })
    trial = aggregate_trials(oof)
    final_model = make_model()
    final_model.fit(X, y, logisticregression__sample_weight=np.full(len(y), weight))
    public_probability = final_model.predict_proba(X_public)[:, 1]
    public = pd.DataFrame({
        "user_id": data["public_user_ids"], "trial_id": data["public_trial_ids"],
        "prob. of positive emotion": public_probability,
    })
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
    metrics = {
        "weight_per_segment": weight,
        "n_subjects": int(np.unique(data["subject_ids"]).size), "n_segments": int(len(y)),
        "n_trials": int(len(trial)), "segment_level": classification_metrics(y, oof_probability),
        "trial_level_diagnostic": classification_metrics(trial["y_true"].to_numpy(), trial["prob_positive"].to_numpy()),
        "folds": [classification_metrics(y[folds == f], oof_probability[folds == f]) for f in range(N_SPLITS)],
    }
    return metrics, oof, trial, public


def write_report(result: dict) -> None:
    a = result["conditions"]["segment_weight_baseline"]
    b = result["conditions"]["trial_balanced"]
    lines = [
        "# 实验 012：trial-balanced 训练",
        "", "> 日期：2026-10-31", "> 状态：已完成；未修改实验001/002", "",
        "## 1. 协议",
        "",
        "复用实验001缓存中的150维 `log_bandpower`，排除HC43后保留69名被试；折分沿用实验002：先在全部70名被试上按健康组分层、seed=2026生成5折，再删除HC43。被试z-score使用每名被试全部无标签段统计，分类器的 `StandardScaler` 每折只在训练折拟合。",
        "",
        "A为现有segment等权基线，每个训练segment权重1；B为trial-balanced训练，每个原始50秒trial的5个segment权重1/5。由于sklearn正则化目标包含样本权重总量，显式保留1/5并非简单忽略参数。C仅将B的OOF概率按5段均值聚合为训练折/OOF trial级诊断；验证和公开预测仍为10秒粒度。",
        "",
        "## 2. 结果",
        "",
        "| 条件 | segment权重 | 片段AUC | 片段F1 | 片段平衡准确率 | trial诊断AUC | trial诊断F1 | trial诊断平衡准确率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| A segment-weight baseline | {a['weight_per_segment']:.5f} | {a['segment_level']['roc_auc']:.6f} | {a['segment_level']['binary_f1']:.6f} | {a['segment_level']['balanced_accuracy']:.6f} | {a['trial_level_diagnostic']['roc_auc']:.6f} | {a['trial_level_diagnostic']['binary_f1']:.6f} | {a['trial_level_diagnostic']['balanced_accuracy']:.6f} |",
        f"| B trial-balanced | {b['weight_per_segment']:.5f} | {b['segment_level']['roc_auc']:.6f} | {b['segment_level']['binary_f1']:.6f} | {b['segment_level']['balanced_accuracy']:.6f} | {b['trial_level_diagnostic']['roc_auc']:.6f} | {b['trial_level_diagnostic']['binary_f1']:.6f} | {b['trial_level_diagnostic']['balanced_accuracy']:.6f} |",
        "",
        "trial诊断只用于说明B在原始trial单位的表现，不替代官方10秒段评价。",
        "",
        "## 3. 公开集与产物",
        "",
        "公开集保留80行连续概率（10名用户、每人8行），没有5段聚合，也没有改变官方提交格式。`predictions/*_oof.csv`为完整片段级OOF；`trial_diagnostic_oof.csv`为B的5段均值诊断；`*_public.csv`为10秒粒度公开预测。",
    ]
    (EXPERIMENT_DIR / "实验报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raw_data = load_cache()
    raw_folds = make_subject_folds(raw_data["subject_ids"], raw_data["health_groups"])
    keep = raw_data["subject_ids"] != "HC43"
    data = {key: value[keep] if key in {"X_train", "y_train", "subject_ids", "health_groups", "original_trial_ids"} else value for key, value in raw_data.items()}
    folds = raw_folds[keep]
    X_all, names = select_log_bandpower(data["X_train"], data["feature_names"])
    X_public, _ = select_log_bandpower(data["X_public"], data["feature_names"])
    X = normalize_by_subject(X_all, data["subject_ids"], "zscore")
    X_public = normalize_by_subject(X_public, data["public_user_ids"], "zscore")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    conditions = {}
    for name, weight in (("segment_weight_baseline", 1.0), ("trial_balanced", 0.2)):
        metrics, oof, trial, public = run_condition(name, X, X_public, data, folds, weight)
        oof.to_csv(PREDICTIONS_DIR / f"{name}_oof.csv", index=False)
        public.to_csv(PREDICTIONS_DIR / f"{name}_public.csv", index=False)
        if name == "trial_balanced":
            trial.to_csv(PREDICTIONS_DIR / "trial_diagnostic_oof.csv", index=False)
        conditions[name] = metrics
    result = {"experiment_config": {"seed": SEED, "n_splits": N_SPLITS, "n_features": int(X.shape[1]), "excluded_subject": "HC43", "segments_per_trial": SEGMENTS_PER_TRIAL, "normalization": "subject zscore all unlabeled segments", "public_rows": 80}, "conditions": conditions}
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
