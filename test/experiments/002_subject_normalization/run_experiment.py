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
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
TRAIN_DATA_KEYS = ("y_train", "subject_ids", "health_groups", "original_trial_ids")
RUN_NAMES = (
    "raw_exclude_hc43",
    "subject_center_all",
    "subject_zscore_all",
    "subject_center_exclude_hc43",
    "subject_zscore_exclude_hc43",
)


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
    return X[:, mask], names[mask]


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


def main() -> None:
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    if tuple(condition["run_name"] for condition in config["conditions"]) != RUN_NAMES:
        raise ValueError(f"Conditions must exactly match: {RUN_NAMES}")

    cache_path = (EXPERIMENT_DIR / config["feature_cache"]).resolve()
    data = load_feature_cache(cache_path)
    folds = make_subject_folds(data["subject_ids"], data["health_groups"], config["random_seed"])
    X_all, selected_names = select_log_bandpower(data["X_train"], data["feature_names"])
    X_public_all, _ = select_log_bandpower(data["X_public"], data["feature_names"])
    if X_all.shape[1] != config["n_features"] or X_public_all.shape[1] != config["n_features"]:
        raise ValueError(f"Expected {config['n_features']} log_bandpower features")

    all_metrics = {"experiment_config": config, "runs": {}}
    summary_rows = []
    for condition in config["conditions"]:
        run_name = condition["run_name"]
        if condition["exclude_hc43"]:
            train_mask = data["subject_ids"] != "HC43"
        else:
            train_mask = np.ones(len(data["subject_ids"]), dtype=bool)
        condition_data = {key: data[key][train_mask] for key in TRAIN_DATA_KEYS}
        condition_data["public_user_ids"] = data["public_user_ids"]
        condition_data["public_trial_ids"] = data["public_trial_ids"]
        condition_folds = folds[train_mask]
        if condition["exclude_hc43"] and np.any(condition_data["subject_ids"] == "HC43"):
            raise RuntimeError(f"{run_name}: HC43 was not completely excluded")

        method = condition["normalization"]
        near_zero_std = config["normalization"]["near_zero_std"]
        X = normalize_by_subject(
            X_all[train_mask], condition_data["subject_ids"], method, near_zero_std=near_zero_std
        )
        X_public = normalize_by_subject(
            X_public_all, data["public_user_ids"], method, near_zero_std=near_zero_std
        )
        result = evaluate_run(
            run_name, X, X_public, condition_data, condition_folds, config, selected_names
        )
        all_metrics["runs"][run_name] = result
        summary_rows.append(
            {
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
        )
        (EXPERIMENT_DIR / "metrics.json").write_text(
            json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame(summary_rows).sort_values("oof_auc", ascending=False).to_csv(
            EXPERIMENT_DIR / "summary.csv", index=False
        )
    print(pd.DataFrame(summary_rows).sort_values("oof_auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
