import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
BASE_DIR = EXPERIMENT_DIR.parent / "002_subject_normalization"
spec = importlib.util.spec_from_file_location("experiment002_protocol", BASE_DIR / "run_experiment.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def coral_training_alignment(X, subject_ids, shrinkage=0.5):
    """Align training subject covariances to a fixed identity target after zscore."""
    if not 0 < shrinkage <= 1:
        raise ValueError("shrinkage must be in (0, 1]")
    aligned = np.empty_like(X, dtype=float)
    for subject in np.unique(subject_ids):
        mask = subject_ids == subject
        centered = X[mask] - X[mask].mean(axis=0)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        covariance = (1 - shrinkage) * covariance + shrinkage * np.eye(X.shape[1])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        inverse_sqrt = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
        aligned[mask] = centered @ inverse_sqrt
    return aligned


def prepare_features(X_train, train_ids, X_eval, eval_ids, condition, shrinkage=0.5):
    if condition not in {"baseline_zscore", "subject_residual", "coral_identity"}:
        raise ValueError(f"Unknown condition: {condition}")
    if set(train_ids) & set(eval_ids):
        raise ValueError("Training and evaluation subjects must be disjoint")
    method = "center" if condition == "subject_residual" else "zscore"
    train = normalize_by_subject(X_train, train_ids, method)
    evaluation = normalize_by_subject(X_eval, eval_ids, method)
    if condition == "coral_identity":
        train = coral_training_alignment(train, train_ids, shrinkage)
    return train, evaluation


def evaluate(condition, X, X_public, data, folds, config, model_config):
    y = data["y_train"]
    ids = data["subject_ids"]
    probabilities = np.full(len(y), np.nan)
    fold_metrics = []
    for fold in range(5):
        train_mask, valid_mask = folds != fold, folds == fold
        train, valid = prepare_features(
            X[train_mask], ids[train_mask], X[valid_mask], ids[valid_mask],
            condition, config["coral_shrinkage"],
        )
        model = protocol.make_model(model_config, config["random_seed"])
        model.fit(train, y[train_mask])
        probabilities[valid_mask] = model.predict_proba(valid)[:, 1]
        fold_metrics.append({
            "fold": fold, "n_subjects": int(len(np.unique(ids[valid_mask]))),
            "n_samples": int(valid_mask.sum()),
            **protocol.classification_metrics(y[valid_mask], probabilities[valid_mask]),
        })
        print(f"{condition}: fold {fold + 1}/5", flush=True)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Incomplete OOF predictions")
    output = EXPERIMENT_DIR / "predictions"
    output.mkdir(exist_ok=True)
    pd.DataFrame({
        "subject_id": ids, "health_group": data["health_groups"],
        "original_trial_id": data["original_trial_ids"], "fold": folds,
        "y_true": y, "prob_positive": probabilities,
    }).to_csv(output / f"{condition}_oof.csv", index=False)
    train, public = prepare_features(
        X, ids, X_public, data["public_user_ids"], condition, config["coral_shrinkage"],
    )
    model = protocol.make_model(model_config, config["random_seed"])
    model.fit(train, y)
    submission = pd.DataFrame({
        "user_id": data["public_user_ids"], "trial_id": data["public_trial_ids"],
        "prob. of positive emotion": model.predict_proba(public)[:, 1],
    })
    validate_submission(submission, [f"P_test{i}" for i in range(1, 11)])
    submission.to_csv(output / f"{condition}_public.csv", index=False)
    subjects = []
    for subject in np.unique(ids):
        mask = ids == subject
        subjects.append({"subject_id": str(subject), **protocol.classification_metrics(y[mask], probabilities[mask])})
    return {
        "n_subjects": int(len(np.unique(ids))), "n_samples": len(y), "n_features": X.shape[1],
        "overall": protocol.classification_metrics(y, probabilities), "folds": fold_metrics,
        "fold_auc_mean": float(np.mean([row["roc_auc"] for row in fold_metrics])),
        "fold_auc_std": float(np.std([row["roc_auc"] for row in fold_metrics], ddof=1)),
        "subject_auc_mean": float(np.mean([row["roc_auc"] for row in subjects])),
        "subjects": subjects,
    }


def main():
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    original_config = json.loads((BASE_DIR / "config.json").read_text(encoding="utf-8"))
    data = protocol.load_feature_cache((EXPERIMENT_DIR / config["feature_cache"]).resolve())
    folds = protocol.make_subject_folds(data["subject_ids"], data["health_groups"], config["random_seed"])
    X, names = protocol.select_log_bandpower(data["X_train"], data["feature_names"])
    X_public, _ = protocol.select_log_bandpower(data["X_public"], data["feature_names"])
    keep = data["subject_ids"] != "HC43"
    X, folds = X[keep], folds[keep]
    for key in protocol.TRAIN_DATA_KEYS:
        data[key] = data[key][keep]
    if len(np.unique(data["subject_ids"])) != 69 or X.shape[1] != 150 or X_public.shape[1] != 150:
        raise ValueError("Expected 69 subjects and 150 log_bandpower features")
    results = {"experiment_config": config, "model": original_config["model"], "feature_names": names.tolist(), "runs": {}}
    rows = []
    with threadpool_limits(limits=config["threads"]):
        for condition in config["conditions"]:
            result = evaluate(condition, X, X_public, data, folds, config, original_config["model"])
            results["runs"][condition] = result
            rows.append({"run_name": condition, "n_subjects": result["n_subjects"],
                         "n_samples": result["n_samples"], "n_features": result["n_features"],
                         "oof_auc": result["overall"]["roc_auc"],
                         "oof_binary_f1": result["overall"]["binary_f1"],
                         "oof_balanced_accuracy": result["overall"]["balanced_accuracy"],
                         "fold_auc_mean": result["fold_auc_mean"], "fold_auc_std": result["fold_auc_std"]})
            (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            pd.DataFrame(rows).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
