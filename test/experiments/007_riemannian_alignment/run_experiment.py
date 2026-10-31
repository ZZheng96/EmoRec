import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.covariance import EuclideanTangentSpace, regularized_covariances
from emo_rec.data import load_public_test_segments, load_train_segments
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[2]


def load_data(config):
    source_path = (EXPERIMENT_DIR / config["source_config"]).resolve()
    source_config = json.loads(source_path.read_text(encoding="utf-8"))
    with np.load(source_path.parent / source_config["feature_cache"]) as cache:
        keys = ("y_train", "subject_ids", "health_groups", "original_trial_ids", "public_user_ids", "public_trial_ids")
        data = {key: cache[key] for key in keys}
    covariances = []
    subjects, labels, trials, health = [], [], [], []
    for path in sorted((ROOT / "数据" / "训练集").glob("*/*.mat")):
        segments, y, trial_ids = load_train_segments(path)
        covariances.append(regularized_covariances(segments, **{"shrinkage": config["covariance"]["shrinkage"]}))
        subjects.extend([path.stem] * len(y))
        health.extend(["DEP" if path.parent.name == "抑郁症患者" else "HC"] * len(y))
        labels.extend(y)
        trials.extend(trial_ids)
        print(f"Covariance: {path.stem}", flush=True)
    for key, actual in (("subject_ids", subjects), ("health_groups", health), ("y_train", labels), ("original_trial_ids", trials)):
        if not np.array_equal(data[key], actual):
            raise ValueError(f"Raw data order differs from experiment 002 cache: {key}")
    public_covariances, public_ids, public_trials = [], [], []
    for path in sorted((ROOT / "数据" / "公开测试集").glob("*.mat"), key=lambda path: int(path.stem.replace("P_test", ""))):
        segments = load_public_test_segments(path)
        public_covariances.append(regularized_covariances(segments, config["covariance"]["shrinkage"]))
        public_ids.extend([path.stem] * len(segments))
        public_trials.extend(range(1, len(segments) + 1))
    if not np.array_equal(public_ids, data["public_user_ids"]) or not np.array_equal(public_trials, data["public_trial_ids"]):
        raise ValueError("Public sample order differs from experiment 002 cache")
    subjects = np.unique(data["subject_ids"])
    subject_health = [data["health_groups"][np.flatnonzero(data["subject_ids"] == subject)[0]] for subject in subjects]
    folds = np.empty(len(data["subject_ids"]), dtype=np.int64)
    splitter = StratifiedKFold(n_splits=config["validation"]["n_splits"], shuffle=True, random_state=config["random_seed"])
    for fold, (_, validation) in enumerate(splitter.split(subjects, subject_health)):
        folds[np.isin(data["subject_ids"], subjects[validation])] = fold
    keep = data["subject_ids"] != config["excluded_subject"]
    for key in ("y_train", "subject_ids", "health_groups", "original_trial_ids"):
        data[key] = data[key][keep]
    if len(np.unique(data["subject_ids"])) != config["expected_subjects"]:
        raise ValueError("Unexpected training subject count")
    return data, np.concatenate(covariances)[keep], np.concatenate(public_covariances), folds[keep]


def metrics(y, probabilities, threshold):
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, probabilities)),
            "binary_f1": float(f1_score(y, predicted, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "threshold": threshold, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    with threadpool_limits(limits=config["cpu_threads"]):
        run(config)


def run(config):
    data, covariances, public_covariances, folds = load_data(config)
    y, subjects = data["y_train"], data["subject_ids"]
    predictions_dir = EXPERIMENT_DIR / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    all_metrics = {"experiment_config": config, "sample_order_verified_against_002_cache": True, "runs": {}}
    summary = []
    for method in config["conditions"]:
        name = f"euclidean_tangent_{method}"
        probabilities = np.full(len(y), np.nan)
        def normalize(features, ids):
            return normalize_by_subject(features, ids, method, config["normalization"]["near_zero_std"])

        def model():
            return make_pipeline(StandardScaler(), LogisticRegression(**config["model"], random_state=config["random_seed"]))

        for fold in range(config["validation"]["n_splits"]):
            train, valid = folds != fold, folds == fold
            space = EuclideanTangentSpace().fit(covariances[train])
            X_train = normalize(space.transform(covariances[train]), subjects[train])
            X_valid = normalize(space.transform(covariances[valid]), subjects[valid])
            classifier = model().fit(X_train, y[train])
            probabilities[valid] = classifier.predict_proba(X_valid)[:, 1]
            print(f"{name}: fold {fold + 1}/5", flush=True)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Non-finite or missing OOF predictions")
        threshold = config["validation"]["probability_threshold"]
        fold_results = [{"fold": int(fold), **metrics(y[folds == fold], probabilities[folds == fold], threshold)} for fold in np.unique(folds)]
        subject_results = [{"subject_id": subject, "health_group": data["health_groups"][subjects == subject][0], **metrics(y[subjects == subject], probabilities[subjects == subject], threshold)} for subject in np.unique(subjects)]
        health_results = {group: metrics(y[data["health_groups"] == group], probabilities[data["health_groups"] == group], threshold) for group in ("HC", "DEP")}
        result = {"n_subjects": len(np.unique(subjects)), "n_samples": len(y), "n_features": X_train.shape[1], "overall": metrics(y, probabilities, threshold), "folds": fold_results, "subject_metrics": subject_results, "health_groups": health_results, "fold_auc_mean": float(np.mean([row["roc_auc"] for row in fold_results])), "fold_auc_std": float(np.std([row["roc_auc"] for row in fold_results], ddof=1)), "subject_auc_mean": float(np.mean([row["roc_auc"] for row in subject_results]))}
        pd.DataFrame({"subject_id": subjects, "health_group": data["health_groups"], "original_trial_id": data["original_trial_ids"], "fold": folds, "y_true": y, "prob_positive": probabilities}).to_csv(predictions_dir / f"{name}_oof.csv", index=False)
        space = EuclideanTangentSpace().fit(covariances)
        classifier = model().fit(normalize(space.transform(covariances), subjects), y)
        public_probabilities = classifier.predict_proba(normalize(space.transform(public_covariances), data["public_user_ids"]))[:, 1]
        public = pd.DataFrame({"user_id": data["public_user_ids"], "trial_id": data["public_trial_ids"], "prob. of positive emotion": public_probabilities})
        validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
        public.to_csv(predictions_dir / f"{name}_public.csv", index=False)
        all_metrics["runs"][name] = result
        summary.append({"run_name": name, "n_subjects": result["n_subjects"], "n_samples": len(y), "n_features": result["n_features"], "oof_auc": result["overall"]["roc_auc"], "oof_binary_f1": result["overall"]["binary_f1"], "oof_balanced_accuracy": result["overall"]["balanced_accuracy"], "fold_auc_mean": result["fold_auc_mean"], "fold_auc_std": result["fold_auc_std"], "subject_auc_mean": result["subject_auc_mean"], "hc_pooled_auc": health_results["HC"]["roc_auc"], "dep_pooled_auc": health_results["DEP"]["roc_auc"]})
        (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame(summary).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
