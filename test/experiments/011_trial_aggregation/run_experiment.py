from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = EXPERIMENT_DIR.parent / "002_subject_normalization" / "predictions"
OUTPUT_DIR = EXPERIMENT_DIR / "predictions"
OOF_SOURCE = SOURCE_DIR / "subject_zscore_exclude_hc43_oof.csv"
PUBLIC_SOURCE = SOURCE_DIR / "subject_zscore_exclude_hc43_public.csv"
SEGMENTS_PER_TRIAL = 5


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "binary_f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def validate_oof(frame: pd.DataFrame) -> None:
    required = {"subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF columns missing: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("OOF contains missing required values")
    raw_groups = frame.groupby(["subject_id", "original_trial_id"], sort=False)
    raw_sizes = raw_groups.size()
    raw_label_counts = raw_groups["y_true"].nunique()
    if ((raw_label_counts > 1) & (raw_sizes != 2 * SEGMENTS_PER_TRIAL)).any():
        raise ValueError("Segments within an original trial have inconsistent labels")
    # The source cache reuses IDs 1-4 for the negative and positive 20-segment
    # blocks. A raw subject+ID group therefore has 10 rows and mixed labels.
    # Include the label only to recover the two 5-segment source trials, and
    # retain the collision in the audit/report rather than silently merging it.
    group_columns = ["subject_id", "health_group", "original_trial_id", "y_true"]
    sizes = frame.groupby(group_columns, sort=False).size()
    if not (sizes == SEGMENTS_PER_TRIAL).all():
        raise ValueError(f"Each original trial must have exactly 5 segments: {sizes.value_counts().to_dict()}")
    inconsistent_labels = frame.groupby(group_columns)["y_true"].nunique()
    if (inconsistent_labels != 1).any():
        raise ValueError("Segments within an original trial have inconsistent labels")
    inconsistent_folds = frame.groupby(group_columns)["fold"].nunique()
    if (inconsistent_folds != 1).any():
        raise ValueError("Segments within an original trial have inconsistent OOF folds")


def trial_group_columns(frame: pd.DataFrame) -> list[str]:
    return ["subject_id", "health_group", "original_trial_id", "y_true"]


def aggregate_oof(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    validate_oof(frame)
    grouped = frame.groupby(trial_group_columns(frame), sort=False)
    if method == "mean":
        probability = grouped["prob_positive"].mean()
    elif method == "median":
        probability = grouped["prob_positive"].median()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    result = grouped.agg(
        fold=("fold", "first"),
        n_segments=("prob_positive", "size"),
    ).reset_index()
    if "health_group" not in result:
        result["health_group"] = grouped["health_group"].first().to_numpy()
    result["prob_positive"] = probability.to_numpy()
    return result[["subject_id", "health_group", "original_trial_id", "fold", "y_true", "n_segments", "prob_positive"]]


def metric_comparison(segment: pd.DataFrame, trial: pd.DataFrame) -> dict:
    segment_metrics = classification_metrics(segment["y_true"].to_numpy(), segment["prob_positive"].to_numpy())
    trial_metrics = classification_metrics(trial["y_true"].to_numpy(), trial["prob_positive"].to_numpy())
    return {
        "segment_level": segment_metrics,
        "trial_level": trial_metrics,
        "difference_trial_minus_segment": {
            key: trial_metrics[key] - segment_metrics[key]
            for key in ("roc_auc", "binary_f1", "balanced_accuracy")
        },
    }


def write_report(result: dict) -> None:
    comparisons = result["comparisons"]
    lines = [
        "# 实验 011：原始 trial 聚合审计",
        "",
        "> 日期：2026-10-31",
        "> 状态：已完成；仅审计既有实验002 OOF，不重训模型",
        "",
        "## 1. 目的与协议",
        "",
        "实验002使用150维 `log_bandpower` 和被试级 OOF 概率。训练集每个原始50秒 trial 被切成5个10秒段。本缓存中负类和正类块会复用 `original_trial_id` 1–4，因此审计先发现裸 `subject_id + original_trial_id` 为10行；为避免把两个不同标签的原始trial错误合并，实际聚合键为 `subject_id + health_group + original_trial_id + y_true`，每组严格5行。公开测试集每行本来就是独立10秒 trial，共80行，不进行聚合，保留实验002的连续概率。",
        "",
        "聚合分别计算5段概率的均值和中位数；标签和OOF折必须在同一原始 trial 内一致。由于没有按trial-balanced样本重新训练，本实验不声称获得了trial-balanced训练结果，也不伪造模型重训。",
        "",
        "## 2. 原始 trial_id 检查",
        "",
        f"- 来源：`{result['source']['oof']}`",
        f"- OOF片段：{result['audit']['segment_rows']}；被试：{result['audit']['n_subjects']}；原始trial：{result['audit']['n_trials']}。",
        f"- 裸 `(subject_id, original_trial_id)` 因正负标签块复用ID而为10行；加上标签块后每个聚合组为 {result['audit']['segments_per_trial']} 个10秒段，标签与OOF折均无冲突。",
        f"- 片段级与trial级比例：{result['audit']['segment_rows']} → {result['audit']['n_trials']}，恰为5:1。",
        "",
        "## 3. 指标对照",
        "",
        "| 聚合方式 | 评估单位 | AUC | Binary F1 | 平衡准确率 | 样本数 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method, comparison in comparisons.items():
        seg = comparison["segment_level"]
        tri = comparison["trial_level"]
        lines.append(f"| {method} | 10秒片段 | {seg['roc_auc']:.6f} | {seg['binary_f1']:.6f} | {seg['balanced_accuracy']:.6f} | {result['audit']['segment_rows']} |")
        lines.append(f"| {method} | 50秒原始trial | {tri['roc_auc']:.6f} | {tri['binary_f1']:.6f} | {tri['balanced_accuracy']:.6f} | {result['audit']['n_trials']} |")
        diff = comparison["difference_trial_minus_segment"]
        lines.append(f"| {method} | trial - 片段差异 | {diff['roc_auc']:+.6f} | {diff['binary_f1']:+.6f} | {diff['balanced_accuracy']:+.6f} | — |")
    lines += [
        "",
        "## 4. trial-balanced训练对照结论",
        "",
        "本实验没有重训分类器，因此不能回答“按原始trial等权训练”是否优于片段级训练。当前产物是对既有OOF的trial级评价审计：聚合改变评价单位，不改变模型训练过程、特征或概率来源。若后续要研究trial-balanced训练，应在每个外层训练折内以原始trial为采样单位或对每个trial的5段加权后重新拟合，并重新生成OOF；不能用本次聚合结果冒充该对照。",
        "",
        "## 5. 公开测试集口径",
        "",
        f"公开集保留 `{result['public']['rows']}` 行连续概率，用户数为 {result['public']['users']}，每用户8个独立10秒 trial；没有按5段聚合，也没有改变提交行数。",
        "",
        "官方评价若定义为80个公开10秒 trial，则训练集的trial聚合指标不能替代官方提交评价，也不能把公开8行聚合成更少行。提交仍应保留每个公开10秒 trial的一行连续概率。",
        "",
        "## 6. 产物",
        "",
        "- `metrics.json`：审计、片段级/ trial级指标及差异。",
        "- `predictions/*_trial_oof.csv`：均值和中位数trial级OOF。",
        "- `predictions/public_unchanged.csv`：原样复制的实验002公开连续概率。",
        "- 本实验未修改实验001/002，也未重训模型。",
    ]
    (EXPERIMENT_DIR / "实验报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    oof = pd.read_csv(OOF_SOURCE)
    public = pd.read_csv(PUBLIC_SOURCE)
    validate_oof(oof)
    public_required = {"user_id", "trial_id", "prob. of positive emotion"}
    if not public_required.issubset(public.columns):
        raise ValueError(f"Public columns missing: {sorted(public_required - set(public.columns))}")
    if len(public) != 80 or public["user_id"].nunique() != 10 or not (public.groupby("user_id").size() == 8).all():
        raise ValueError("Public data must contain 10 users with 8 independent rows each")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparisons = {}
    for method in ("mean", "median"):
        trial = aggregate_oof(oof, method)
        trial.to_csv(OUTPUT_DIR / f"subject_zscore_exclude_hc43_{method}_trial_oof.csv", index=False)
        comparisons[method] = metric_comparison(oof, trial)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    public.to_csv(OUTPUT_DIR / "public_unchanged.csv", index=False)

    result = {
        "source": {"oof": str(OOF_SOURCE), "public": str(PUBLIC_SOURCE), "model_retrained": False},
        "audit": {
            "segment_rows": int(len(oof)),
            "n_subjects": int(oof["subject_id"].nunique()),
            "n_trials": int(len(aggregate_oof(oof, "mean"))),
            "segments_per_trial": SEGMENTS_PER_TRIAL,
            "group_key": ["subject_id", "health_group", "original_trial_id", "y_true"],
            "original_trial_id_collision_audit": "每个subject的original_trial_id 1-4在两种标签块各出现5段；为避免错误合并，聚合键保留health_group/y_true。",
            "labels_consistent": True,
            "folds_consistent": True,
        },
        "comparisons": comparisons,
        "public": {"rows": int(len(public)), "users": int(public["user_id"].nunique()), "rows_per_user": 8, "aggregated": False},
        "training_audit": {"trial_balanced_retrained": False, "claim": "聚合审计，不代表trial-balanced训练对照"},
    }
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
