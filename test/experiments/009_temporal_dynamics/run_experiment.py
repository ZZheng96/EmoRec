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

from emo_rec.data import load_public_test_segments, load_train_segments
from emo_rec.features import extract_spectral_features
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CACHE_PATH = EXPERIMENT_DIR / "cache" / "temporal_features.npz"
BASELINE_CACHE = ROOT / "test" / "experiments" / "001_spectral_logpower" / "cache" / "features.npz"
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
RUN_NAMES = ("baseline_zscore_exclude_hc43", "temporal_stats_raw_exclude_hc43", "temporal_stats_zscore_exclude_hc43")


def numeric_test_sort_key(path: Path) -> int:
    return int(path.stem.replace("P_test", ""))


def temporal_log_features(segments: np.ndarray) -> tuple[np.ndarray, list[str]]:
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (30, 2500):
        raise ValueError("Expected 10-second segments with shape (n, 30, 2500)")
    subwindows = segments.reshape(len(segments), 30, 5, 500).transpose(0, 2, 1, 3).reshape(-1, 30, 500)
    sub_features, names = extract_spectral_features(subwindows)
    mask = np.asarray([name.startswith("log_bandpower_") for name in names])
    sub_features = sub_features[:, mask].reshape(len(segments), 5, 150)
    t = np.arange(5, dtype=np.float64)
    centered_t = t - t.mean()
    slope = np.sum((sub_features - sub_features.mean(axis=1, keepdims=True)) * centered_t[None, :, None], axis=1)
    slope /= np.sum(centered_t**2)
    mean = sub_features.mean(axis=1)
    std = sub_features.std(axis=1)
    output = np.concatenate([mean, std, slope], axis=1)
    output_names = [f"temporal_mean_{name.removeprefix('log_bandpower_')}" for name in names if name.startswith("log_bandpower_")]
    output_names += [f"temporal_std_{name.removeprefix('log_bandpower_')}" for name in names if name.startswith("log_bandpower_")]
    output_names += [f"temporal_slope_{name.removeprefix('log_bandpower_')}" for name in names if name.startswith("log_bandpower_")]
    return output, output_names


def build_feature_cache() -> None:
    train_features, labels, subjects, groups, trial_ids = [], [], [], [], []
    feature_names = None
    train_files = sorted((ROOT / "数据" / "训练集").rglob("*.mat"))
    for index, path in enumerate(train_files, start=1):
        segments, subject_labels, subject_trial_ids = load_train_segments(path)
        features, feature_names = temporal_log_features(segments)
        train_features.append(features)
        labels.append(subject_labels)
        subjects.extend([path.stem] * len(segments))
        groups.extend(["DEP" if path.parent.name == "抑郁症患者" else "HC"] * len(segments))
        trial_ids.append(subject_trial_ids)
        print(f"Features: {index}/{len(train_files)} {path.stem}", flush=True)

    public_features, public_user_ids, public_trial_ids = [], [], []
    test_files = sorted((ROOT / "数据" / "公开集" / "公开测试集").glob("*.mat"), key=numeric_test_sort_key)
    for path in test_files:
        features, _ = temporal_log_features(load_public_test_segments(path))
        public_features.append(features)
        public_user_ids.extend([path.stem] * len(features))
        public_trial_ids.extend(np.arange(1, 9))

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, X_train=np.concatenate(train_features), y_train=np.concatenate(labels),
                        subject_ids=np.asarray(subjects), health_groups=np.asarray(groups),
                        original_trial_ids=np.concatenate(trial_ids), feature_names=np.asarray(feature_names),
                        X_public=np.concatenate(public_features), public_user_ids=np.asarray(public_user_ids),
                        public_trial_ids=np.asarray(public_trial_ids))


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as cache:
        return {key: cache[key] for key in cache.files}


def make_subject_folds(subject_ids: np.ndarray, health_groups: np.ndarray, seed: int) -> np.ndarray:
    subjects = np.unique(subject_ids)
    subject_health = np.asarray([health_groups[np.flatnonzero(subject_ids == subject)[0]] for subject in subjects])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_by_subject = {}
    for fold, (_, valid) in enumerate(splitter.split(subjects, subject_health)):
        for subject in subjects[valid]:
            fold_by_subject[subject] = fold
    return np.asarray([fold_by_subject[subject] for subject in subject_ids], dtype=np.int64)


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y_true, probabilities)), "binary_f1": float(f1_score(y_true, predictions, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def evaluate(run_name: str, X: np.ndarray, X_public: np.ndarray, data: dict, folds: np.ndarray, config: dict, names: np.ndarray) -> dict:
    y = data["y_train"]
    model_config = config["model"]
    oof = np.full(len(y), np.nan)
    for fold in range(5):
        train_mask, valid_mask = folds != fold, folds == fold
        model = make_pipeline(StandardScaler(), LogisticRegression(C=model_config["C"], penalty=model_config["penalty"], solver=model_config["solver"], max_iter=model_config["max_iter"], random_state=config["random_seed"]))
        model.fit(X[train_mask], y[train_mask])
        oof[valid_mask] = model.predict_proba(X[valid_mask])[:, 1]
        print(f"{run_name}: fold {fold + 1}/5", flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError("incomplete OOF predictions")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject_id": data["subject_ids"], "health_group": data["health_groups"], "original_trial_id": data["original_trial_ids"], "fold": folds, "y_true": y, "prob_positive": oof}).to_csv(PREDICTIONS_DIR / f"{run_name}_oof.csv", index=False)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=model_config["C"], penalty=model_config["penalty"], solver=model_config["solver"], max_iter=model_config["max_iter"], random_state=config["random_seed"]))
    model.fit(X, y)
    public = pd.DataFrame({"user_id": data["public_user_ids"], "trial_id": data["public_trial_ids"], "prob. of positive emotion": model.predict_proba(X_public)[:, 1]})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
    public.to_csv(PREDICTIONS_DIR / f"{run_name}_public.csv", index=False)
    fold_metrics = [metrics(y[folds == fold], oof[folds == fold]) for fold in range(5)]
    return {"n_subjects": int(len(np.unique(data["subject_ids"]))), "n_samples": int(len(y)), "n_features": int(X.shape[1]), "feature_names": names.tolist(), "overall": metrics(y, oof), "folds": fold_metrics, "fold_auc_mean": float(np.mean([m["roc_auc"] for m in fold_metrics])), "fold_auc_std": float(np.std([m["roc_auc"] for m in fold_metrics], ddof=1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    if args.rebuild_features or not CACHE_PATH.exists():
        build_feature_cache()
    temporal = load_cache(CACHE_PATH)
    baseline = load_cache(BASELINE_CACHE)
    folds_all = make_subject_folds(baseline["subject_ids"], baseline["health_groups"], config["random_seed"])
    keep = baseline["subject_ids"] != "HC43"
    data = {key: baseline[key][keep] for key in ("y_train", "subject_ids", "health_groups", "original_trial_ids")}
    data["public_user_ids"], data["public_trial_ids"] = baseline["public_user_ids"], baseline["public_trial_ids"]
    folds = folds_all[keep]
    baseline_names = np.asarray([n for n in baseline["feature_names"] if n.startswith("log_bandpower_")])
    baseline_X, baseline_public_X = baseline["X_train"][:, :150], baseline["X_public"][:, :150]
    if len(baseline_names) != 150 or temporal["X_train"].shape[1] != 450:
        raise ValueError("Unexpected baseline or temporal feature dimensions")
    runs = {}
    summary = []
    condition_data = [(RUN_NAMES[0], normalize_by_subject(baseline_X[keep], data["subject_ids"], "zscore"), normalize_by_subject(baseline_public_X, data["public_user_ids"], "zscore"), baseline_names),
                      (RUN_NAMES[1], temporal["X_train"][keep], temporal["X_public"], temporal["feature_names"]),
                      (RUN_NAMES[2], normalize_by_subject(temporal["X_train"][keep], data["subject_ids"], "zscore"), normalize_by_subject(temporal["X_public"], data["public_user_ids"], "zscore"), temporal["feature_names"])]
    for run_name, X, X_public, names in condition_data:
        result = evaluate(run_name, X, X_public, data, folds, config, np.asarray(names))
        runs[run_name] = result
        summary.append({"run_name": run_name, "n_features": result["n_features"], "oof_auc": result["overall"]["roc_auc"], "oof_binary_f1": result["overall"]["binary_f1"], "oof_balanced_accuracy": result["overall"]["balanced_accuracy"], "fold_auc_mean": result["fold_auc_mean"], "fold_auc_std": result["fold_auc_std"]})
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps({"experiment_config": config, "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary).sort_values("oof_auc", ascending=False).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    print(pd.DataFrame(summary).sort_values("oof_auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
