import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emo_rec.features import CHANNEL_NAMES, FREQUENCY_BANDS, compute_relative_log_bandpower
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
TRAIN_DATA_KEYS = ("y_train", "subject_ids", "health_groups", "original_trial_ids")
RUN_NAMES = ("absolute_log_bandpower", "relative_log_bandpower", "absolute_plus_relative")


def load_feature_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Feature cache not found: {path}")
    with np.load(path) as cache:
        return {key: cache[key] for key in cache.files}


def make_subject_folds(subject_ids: np.ndarray, health_groups: np.ndarray, seed: int) -> np.ndarray:
    subjects = np.unique(subject_ids)
    subject_health = np.asarray([health_groups[np.flatnonzero(subject_ids == subject)[0]] for subject in subjects])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_by_subject = {}
    for fold, (_, validation_indices) in enumerate(splitter.split(subjects, subject_health)):
        for subject in subjects[validation_indices]:
            fold_by_subject[subject] = fold
    return np.asarray([fold_by_subject[subject] for subject in subject_ids], dtype=np.int64)


def select_log_bandpower(X: np.ndarray, names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray([name.startswith("log_bandpower_") for name in names])
    selected_names = names[mask]
    expected_names = np.asarray(
        [
            f"log_bandpower_{channel}_{band}"
            for channel in CHANNEL_NAMES
            for band in FREQUENCY_BANDS
        ]
    )
    if not np.array_equal(selected_names, expected_names):
        raise ValueError("Cached log_bandpower features do not follow the expected channel-major band order")
    return X[:, mask], selected_names


def make_feature_set(
    X_absolute: np.ndarray, absolute_names: np.ndarray, feature_set: str
) -> tuple[np.ndarray, np.ndarray]:
    if feature_set == "absolute":
        return X_absolute, absolute_names

    shaped = X_absolute.reshape(len(X_absolute), len(CHANNEL_NAMES), len(FREQUENCY_BANDS))
    relative = compute_relative_log_bandpower(shaped).reshape(len(X_absolute), -1)
    relative_names = np.asarray(
        [name.replace("log_bandpower_", "relative_log_bandpower_", 1) for name in absolute_names]
    )
    if feature_set == "relative":
        return relative, relative_names
    if feature_set == "absolute_plus_relative":
        return (
            np.concatenate([X_absolute, relative], axis=1),
            np.concatenate([absolute_names, relative_names]),
        )
    raise ValueError(f"Unknown feature set: {feature_set}")


def make_model(model_config: dict, seed: int):
    if model_config["name"] != "logistic_regression":
        raise ValueError(f"Unknown model: {model_config['name']}")
    classifier = LogisticRegression(
        C=model_config["C"],
        penalty=model_config["penalty"],
        max_iter=model_config["max_iter"],
        solver=model_config["solver"],
        random_state=seed,
    )
    return make_pipeline(StandardScaler(), classifier)


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


def evaluate_run(
    run_name: str,
    X: np.ndarray,
    X_public: np.ndarray,
    data: dict[str, np.ndarray],
    folds: np.ndarray,
    config: dict,
    feature_names: np.ndarray,
) -> dict:
    y = data["y_train"]
    seed = config["random_seed"]
    n_splits = config["validation"]["n_splits"]
    threshold = config["validation"]["probability_threshold"]
    oof_probabilities = np.full(len(y), np.nan)
    for fold in range(n_splits):
        train_mask = folds != fold
        validation_mask = folds == fold
        model = make_model(config["model"], seed)
        model.fit(X[train_mask], y[train_mask])
        oof_probabilities[validation_mask] = model.predict_proba(X[validation_mask])[:, 1]
        print(f"{run_name}: fold {fold + 1}/{n_splits}", flush=True)
    if not np.isfinite(oof_probabilities).all():
        raise RuntimeError(f"{run_name}: incomplete OOF predictions")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame(
        {
            "subject_id": data["subject_ids"],
            "health_group": data["health_groups"],
            "original_trial_id": data["original_trial_ids"],
            "fold": folds,
            "y_true": y,
            "prob_positive": oof_probabilities,
        }
    )
    oof.to_csv(PREDICTIONS_DIR / f"{run_name}_oof.csv", index=False)

    fold_metrics = []
    for fold in range(n_splits):
        mask = folds == fold
        fold_metrics.append(
            {
                "fold": fold,
                "n_subjects": int(len(np.unique(data["subject_ids"][mask]))),
                "n_samples": int(mask.sum()),
                **classification_metrics(y[mask], oof_probabilities[mask], threshold),
            }
        )

    subject_rows = []
    for subject in np.unique(data["subject_ids"]):
        mask = data["subject_ids"] == subject
        subject_rows.append(
            {
                "run_name": run_name,
                "subject_id": subject,
                "health_group": data["health_groups"][mask][0],
                **classification_metrics(y[mask], oof_probabilities[mask], threshold),
            }
        )
    subject_metrics = pd.DataFrame(subject_rows)

    health_metrics = {}
    for health_group in ("HC", "DEP"):
        mask = data["health_groups"] == health_group
        group_subjects = subject_metrics[subject_metrics["health_group"] == health_group]
        health_metrics[health_group] = {
            "n_subjects": int(len(group_subjects)),
            "pooled": classification_metrics(y[mask], oof_probabilities[mask], threshold),
            "subject_auc_mean": float(group_subjects["roc_auc"].mean()),
            "subject_auc_std": float(group_subjects["roc_auc"].std(ddof=1)),
        }

    final_model = make_model(config["model"], seed)
    final_model.fit(X, y)
    public_probabilities = final_model.predict_proba(X_public)[:, 1]
    public = pd.DataFrame(
        {
            "user_id": data["public_user_ids"],
            "trial_id": data["public_trial_ids"],
            "prob. of positive emotion": public_probabilities,
        }
    )
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
    public.to_csv(PREDICTIONS_DIR / f"{run_name}_public.csv", index=False)

    return {
        "n_subjects": int(len(np.unique(data["subject_ids"]))),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names.tolist(),
        "overall": classification_metrics(y, oof_probabilities, threshold),
        "folds": fold_metrics,
        "fold_auc_mean": float(np.mean([fold["roc_auc"] for fold in fold_metrics])),
        "fold_auc_std": float(np.std([fold["roc_auc"] for fold in fold_metrics], ddof=1)),
        "subject_auc_mean": float(subject_metrics["roc_auc"].mean()),
        "subject_auc_std": float(subject_metrics["roc_auc"].std(ddof=1)),
        "health_groups": health_metrics,
        "subject_metrics": subject_rows,
    }


def verify_absolute_baseline(result: dict, config: dict) -> dict:
    reference_config = config["baseline_reference"]
    reference_dir = (EXPERIMENT_DIR / reference_config["experiment_dir"]).resolve()
    reference_run_name = reference_config["run_name"]
    reference_metrics = json.loads((reference_dir / "metrics.json").read_text(encoding="utf-8"))
    reference_result = reference_metrics["runs"][reference_run_name]

    actual_oof = pd.read_csv(PREDICTIONS_DIR / f"{RUN_NAMES[0]}_oof.csv")
    reference_oof = pd.read_csv(reference_dir / "predictions" / f"{reference_run_name}_oof.csv")
    identity_columns = ["subject_id", "health_group", "original_trial_id", "fold", "y_true"]
    pd.testing.assert_frame_equal(
        actual_oof[identity_columns], reference_oof[identity_columns], check_dtype=False, check_exact=True
    )
    actual_probabilities = actual_oof["prob_positive"].to_numpy()
    reference_probabilities = reference_oof["prob_positive"].to_numpy()
    max_difference = float(np.max(np.abs(actual_probabilities - reference_probabilities)))
    if reference_config["require_exact_oof_probabilities"]:
        np.testing.assert_array_equal(actual_probabilities, reference_probabilities)

    if result["overall"]["roc_auc"] != reference_config["expected_oof_auc"]:
        raise AssertionError("Absolute baseline OOF AUC differs from the stated experiment 002 best value")

    metric_pairs = {
        "overall": (result["overall"], reference_result["overall"]),
        "subject_auc_mean": (result["subject_auc_mean"], reference_result["subject_auc_mean"]),
        "hc_pooled_auc": (
            result["health_groups"]["HC"]["pooled"]["roc_auc"],
            reference_result["health_groups"]["HC"]["pooled"]["roc_auc"],
        ),
        "dep_pooled_auc": (
            result["health_groups"]["DEP"]["pooled"]["roc_auc"],
            reference_result["health_groups"]["DEP"]["pooled"]["roc_auc"],
        ),
    }
    if reference_config["require_exact_metrics"]:
        for metric_name, (actual, expected) in metric_pairs.items():
            if actual != expected:
                raise AssertionError(f"Absolute baseline {metric_name} differs from experiment 002")

    return {
        "reference_run": reference_run_name,
        "oof_probabilities_exact": bool(np.array_equal(actual_probabilities, reference_probabilities)),
        "max_abs_oof_probability_difference": max_difference,
        "metrics_exact": all(actual == expected for actual, expected in metric_pairs.values()),
        "expected_oof_auc_exact": result["overall"]["roc_auc"] == reference_config["expected_oof_auc"],
    }


def summary_row(run_name: str, result: dict) -> dict:
    return {
        "run_name": run_name,
        "n_subjects": result["n_subjects"],
        "n_samples": result["n_samples"],
        "n_features": result["n_features"],
        "oof_auc": result["overall"]["roc_auc"],
        "oof_binary_f1": result["overall"]["binary_f1"],
        "oof_balanced_accuracy": result["overall"]["balanced_accuracy"],
        "fold_auc_mean": result["fold_auc_mean"],
        "fold_auc_std": result["fold_auc_std"],
        "subject_auc_mean": result["subject_auc_mean"],
        "hc_pooled_auc": result["health_groups"]["HC"]["pooled"]["roc_auc"],
        "dep_pooled_auc": result["health_groups"]["DEP"]["pooled"]["roc_auc"],
        "hc_subject_auc_mean": result["health_groups"]["HC"]["subject_auc_mean"],
        "dep_subject_auc_mean": result["health_groups"]["DEP"]["subject_auc_mean"],
    }


def main() -> None:
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    if tuple(condition["run_name"] for condition in config["conditions"]) != RUN_NAMES:
        raise ValueError(f"Conditions must exactly match: {RUN_NAMES}")

    cache_path = (EXPERIMENT_DIR / config["feature_cache"]).resolve()
    data = load_feature_cache(cache_path)
    folds = make_subject_folds(data["subject_ids"], data["health_groups"], config["random_seed"])
    X_absolute_all, absolute_names = select_log_bandpower(data["X_train"], data["feature_names"])
    X_public_absolute_all, _ = select_log_bandpower(data["X_public"], data["feature_names"])

    train_mask = data["subject_ids"] != config["normalization"]["excluded_subject"]
    condition_data = {key: data[key][train_mask] for key in TRAIN_DATA_KEYS}
    condition_data["public_user_ids"] = data["public_user_ids"]
    condition_data["public_trial_ids"] = data["public_trial_ids"]
    condition_folds = folds[train_mask]
    if np.any(condition_data["subject_ids"] == "HC43"):
        raise RuntimeError("HC43 was not completely excluded")

    all_metrics = {"experiment_config": config, "runs": {}}
    summary_rows = []
    for condition in config["conditions"]:
        if not condition["exclude_hc43"]:
            raise ValueError("All experiment 003 conditions must exclude HC43")
        run_name = condition["run_name"]
        X_train, feature_names = make_feature_set(X_absolute_all, absolute_names, condition["feature_set"])
        X_public, public_feature_names = make_feature_set(
            X_public_absolute_all, absolute_names, condition["feature_set"]
        )
        if not np.array_equal(feature_names, public_feature_names):
            raise RuntimeError(f"{run_name}: train/public feature names differ")
        if X_train.shape[1] != condition["n_features"] or X_public.shape[1] != condition["n_features"]:
            raise ValueError(f"{run_name}: expected {condition['n_features']} features")

        normalization = config["normalization"]
        X = normalize_by_subject(
            X_train[train_mask],
            condition_data["subject_ids"],
            normalization["method"],
            near_zero_std=normalization["near_zero_std"],
        )
        X_public = normalize_by_subject(
            X_public,
            data["public_user_ids"],
            normalization["method"],
            near_zero_std=normalization["near_zero_std"],
        )
        result = evaluate_run(
            run_name, X, X_public, condition_data, condition_folds, config, feature_names
        )
        all_metrics["runs"][run_name] = result
        summary_rows.append(summary_row(run_name, result))
        if run_name == RUN_NAMES[0]:
            all_metrics["baseline_verification"] = verify_absolute_baseline(result, config)
        (EXPERIMENT_DIR / "metrics.json").write_text(
            json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    baseline = summary_rows[0]
    comparison_metrics = ("oof_auc", "oof_binary_f1", "subject_auc_mean", "hc_pooled_auc", "dep_pooled_auc")
    comparisons = {}
    for row in summary_rows:
        comparisons[row["run_name"]] = {}
        for metric_name in comparison_metrics:
            delta = row[metric_name] - baseline[metric_name]
            row[f"delta_{metric_name}_vs_absolute"] = delta
            comparisons[row["run_name"]][f"delta_{metric_name}"] = delta
    all_metrics["comparisons_to_absolute_baseline"] = comparisons
    (EXPERIMENT_DIR / "metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
