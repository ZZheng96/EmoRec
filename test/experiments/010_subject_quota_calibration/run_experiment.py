from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = EXPERIMENT_DIR.parent / "002_subject_normalization" / "predictions"
OUTPUT_DIR = EXPERIMENT_DIR / "predictions"
OOF_SOURCE = SOURCE_DIR / "subject_zscore_exclude_hc43_oof.csv"
PUBLIC_SOURCE = SOURCE_DIR / "subject_zscore_exclude_hc43_public.csv"

SAMPLES_PER_SUBJECT = 40
POSITIVE_QUOTA = 20
PUBLIC_TRIALS_PER_USER = 8
PUBLIC_POSITIVE_QUOTA = 4


def apply_quota(probabilities: np.ndarray, quota: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or quota < 0 or quota > len(probabilities):
        raise ValueError("quota must be between 0 and the number of probabilities")
    order = np.argsort(-probabilities, kind="mergesort")
    predictions = np.zeros(len(probabilities), dtype=np.int64)
    predictions[order[:quota]] = 1
    return predictions


def quota_by_group(frame: pd.DataFrame, group_column: str, quota: int) -> np.ndarray:
    predictions = np.zeros(len(frame), dtype=np.int64)
    for _, group in frame.groupby(group_column, sort=False):
        predictions[group.index.to_numpy()] = apply_quota(group["prob_positive"].to_numpy(), quota)
    return predictions


def metrics(y_true: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "binary_f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_count": int(predictions.sum()),
    }


def check_groups(frame: pd.DataFrame, group_column: str, expected_size: int) -> None:
    sizes = frame.groupby(group_column, sort=False).size()
    if not (sizes == expected_size).all():
        raise ValueError(f"{group_column} groups must all contain {expected_size} rows; got {sizes.to_dict()}")


def main() -> None:
    oof = pd.read_csv(OOF_SOURCE)
    public = pd.read_csv(PUBLIC_SOURCE)
    required_oof = {"subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"}
    required_public = {"user_id", "trial_id", "prob. of positive emotion"}
    if not required_oof.issubset(oof.columns) or not required_public.issubset(public.columns):
        raise ValueError("source prediction columns are incomplete")
    check_groups(oof, "subject_id", SAMPLES_PER_SUBJECT)
    check_groups(public.rename(columns={"prob. of positive emotion": "prob_positive"}), "user_id", PUBLIC_TRIALS_PER_USER)

    oof["quota_prediction"] = quota_by_group(oof, "subject_id", POSITIVE_QUOTA)
    public = public.rename(columns={"prob. of positive emotion": "prob_positive"})
    public["quota_prediction"] = quota_by_group(public, "user_id", PUBLIC_POSITIVE_QUOTA)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(OUTPUT_DIR / "subject_zscore_exclude_hc43_quota_oof.csv", index=False)
    public.to_csv(OUTPUT_DIR / "subject_zscore_exclude_hc43_quota_public.csv", index=False)

    # A binary submission candidate is emitted only for audit; preserving rank probabilities is preferred.
    binary_public = public[["user_id", "trial_id", "quota_prediction"]].rename(
        columns={"quota_prediction": "prob. of positive emotion"}
    )
    binary_public.to_csv(OUTPUT_DIR / "subject_zscore_exclude_hc43_quota_public_binary.csv", index=False)

    original_oof_predictions = (oof["prob_positive"].to_numpy() >= 0.5).astype(np.int64)
    calibrated_metrics = metrics(
        oof["y_true"].to_numpy(), oof["prob_positive"].to_numpy(), oof["quota_prediction"].to_numpy()
    )
    original_metrics = metrics(
        oof["y_true"].to_numpy(), oof["prob_positive"].to_numpy(), original_oof_predictions
    )
    result = {
        "source": {
            "oof": str(OOF_SOURCE),
            "public": str(PUBLIC_SOURCE),
            "model_retrained": False,
        },
        "prior": {
            "oof_samples_per_subject": SAMPLES_PER_SUBJECT,
            "oof_positive_quota": POSITIVE_QUOTA,
            "public_trials_per_user": PUBLIC_TRIALS_PER_USER,
            "public_positive_quota": PUBLIC_POSITIVE_QUOTA,
            "labels_used_for_quota": False,
        },
        "oof": {
            "before_threshold_0_5": original_metrics,
            "after_quota": calibrated_metrics,
            "auc_change": calibrated_metrics["roc_auc"] - original_metrics["roc_auc"],
            "f1_change": calibrated_metrics["binary_f1"] - original_metrics["binary_f1"],
            "auc_uses_original_probabilities": True,
        },
        "public": {
            "n_users": int(public["user_id"].nunique()),
            "n_rows": int(len(public)),
            "positive_count": int(public["quota_prediction"].sum()),
            "binary_submission_is_rank_destroying": True,
        },
    }
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
