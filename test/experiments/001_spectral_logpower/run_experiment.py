import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from emo_rec.data import load_public_test_segments, load_train_segments
from emo_rec.features import extract_spectral_features
from emo_rec.submission import validate_submission

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CACHE_PATH = EXPERIMENT_DIR / "cache" / "features.npz"
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"


def numeric_test_sort_key(path: Path) -> int:
    return int(path.stem.replace("P_test", ""))


def build_feature_cache() -> None:
    train_features = []
    labels = []
    subjects = []
    health_groups = []
    original_trial_ids = []
    feature_names = None

    train_files = sorted((ROOT / "数据" / "训练集").glob("*/*.mat"))
    for index, path in enumerate(train_files, start=1):
        segments, subject_labels, trial_ids = load_train_segments(path)
        features, feature_names = extract_spectral_features(segments)
        train_features.append(features)
        labels.append(subject_labels)
        subjects.extend([path.stem] * len(segments))
        health_groups.extend(["DEP" if path.parent.name == "抑郁症患者" else "HC"] * len(segments))
        original_trial_ids.append(trial_ids)
        print(f"Features: {index}/{len(train_files)} {path.stem}", flush=True)

    public_features = []
    public_user_ids = []
    public_trial_ids = []
    test_files = sorted((ROOT / "数据" / "公开测试集").glob("*.mat"), key=numeric_test_sort_key)
    for path in test_files:
        segments = load_public_test_segments(path)
        features, _ = extract_spectral_features(segments)
        public_features.append(features)
        public_user_ids.extend([path.stem] * len(segments))
        public_trial_ids.extend(np.arange(1, 9))

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        X_train=np.concatenate(train_features),
        y_train=np.concatenate(labels),
        subject_ids=np.asarray(subjects),
        health_groups=np.asarray(health_groups),
        original_trial_ids=np.concatenate(original_trial_ids),
        feature_names=np.asarray(feature_names),
        X_public=np.concatenate(public_features),
        public_user_ids=np.asarray(public_user_ids),
        public_trial_ids=np.asarray(public_trial_ids),
    )


def load_feature_cache() -> dict[str, np.ndarray]:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Feature cache not found: {CACHE_PATH}")
    with np.load(CACHE_PATH) as cache:
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


def make_model(name: str, config: dict, seed: int):
    if name == "logistic_regression":
        classifier = LogisticRegression(
            C=config["C"],
            penalty=config["penalty"],
            max_iter=config["max_iter"],
            solver="lbfgs",
            random_state=seed,
        )
    elif name == "rbf_svm":
        classifier = SVC(
            C=config["C"],
            kernel=config["kernel"],
            gamma=config["gamma"],
            probability=config["probability"],
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return make_pipeline(StandardScaler(), classifier)


def select_features(X: np.ndarray, names: np.ndarray, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    if feature_set == "log_bandpower":
        mask = np.asarray([name.startswith("log_bandpower_") for name in names])
    elif feature_set == "asymmetry":
        mask = np.asarray([name.startswith("asymmetry_") for name in names])
    elif feature_set == "combined":
        mask = np.ones(len(names), dtype=bool)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return X[:, mask], names[mask]


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
    model_name: str,
    model_config: dict,
    feature_names: np.ndarray,
    seed: int,
) -> dict:
    y = data["y_train"]
    oof_probabilities = np.full(len(y), np.nan)
    for fold in range(5):
        train_mask = folds != fold
        validation_mask = folds == fold
        model = make_model(model_name, model_config, seed)
        model.fit(X[train_mask], y[train_mask])
        oof_probabilities[validation_mask] = model.predict_proba(X[validation_mask])[:, 1]
        print(f"{run_name}: fold {fold + 1}/5", flush=True)
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
    for fold in range(5):
        mask = folds == fold
        fold_metrics.append(
            {
                "fold": fold,
                "n_subjects": int(len(np.unique(data["subject_ids"][mask]))),
                "n_samples": int(mask.sum()),
                **classification_metrics(y[mask], oof_probabilities[mask]),
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
                **classification_metrics(y[mask], oof_probabilities[mask]),
            }
        )
    subject_metrics = pd.DataFrame(subject_rows)

    health_metrics = {}
    for health_group in ("HC", "DEP"):
        mask = data["health_groups"] == health_group
        group_subjects = subject_metrics[subject_metrics["health_group"] == health_group]
        health_metrics[health_group] = {
            "n_subjects": int(len(group_subjects)),
            "pooled": classification_metrics(y[mask], oof_probabilities[mask]),
            "subject_auc_mean": float(group_subjects["roc_auc"].mean()),
            "subject_auc_std": float(group_subjects["roc_auc"].std(ddof=1)),
        }

    final_model = make_model(model_name, model_config, seed)
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
        "n_features": int(X.shape[1]),
        "feature_names": feature_names.tolist(),
        "overall": classification_metrics(y, oof_probabilities),
        "folds": fold_metrics,
        "fold_auc_mean": float(np.mean([fold["roc_auc"] for fold in fold_metrics])),
        "fold_auc_std": float(np.std([fold["roc_auc"] for fold in fold_metrics], ddof=1)),
        "subject_auc_mean": float(subject_metrics["roc_auc"].mean()),
        "subject_auc_std": float(subject_metrics["roc_auc"].std(ddof=1)),
        "health_groups": health_metrics,
        "subject_metrics": subject_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    if args.rebuild_features or not CACHE_PATH.exists():
        build_feature_cache()
    data = load_feature_cache()
    folds = make_subject_folds(data["subject_ids"], data["health_groups"], config["random_seed"])

    all_metrics = {"experiment_config": config, "runs": {}}
    summary_rows = []
    for feature_set in config["feature_sets"]:
        X, selected_names = select_features(data["X_train"], data["feature_names"], feature_set)
        X_public, _ = select_features(data["X_public"], data["feature_names"], feature_set)
        for model_name, model_config in config["models"].items():
            run_name = f"{feature_set}__{model_name}"
            result = evaluate_run(
                run_name, X, X_public, data, folds, model_name, model_config, selected_names, config["random_seed"]
            )
            all_metrics["runs"][run_name] = result
            summary_rows.append(
                {
                    "run_name": run_name,
                    "n_features": result["n_features"],
                    "oof_auc": result["overall"]["roc_auc"],
                    "oof_binary_f1": result["overall"]["binary_f1"],
                    "oof_balanced_accuracy": result["overall"]["balanced_accuracy"],
                    "fold_auc_mean": result["fold_auc_mean"],
                    "fold_auc_std": result["fold_auc_std"],
                    "subject_auc_mean": result["subject_auc_mean"],
                    "hc_pooled_auc": result["health_groups"]["HC"]["pooled"]["roc_auc"],
                    "dep_pooled_auc": result["health_groups"]["DEP"]["pooled"]["roc_auc"],
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
