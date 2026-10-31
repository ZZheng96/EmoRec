import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
OOF_COLUMNS = ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"]


def load_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def subject_zscore(X, subject_ids, near_zero_std=1e-12):
    X = np.asarray(X, dtype=np.float64)
    subject_ids = np.asarray(subject_ids)
    if X.ndim != 2 or len(subject_ids) != len(X):
        raise ValueError("X and subject_ids shape mismatch")
    out = np.empty_like(X)
    for subject in np.unique(subject_ids):
        mask = subject_ids == subject
        values = X[mask]
        std = values.std(axis=0)
        out[mask] = (values - values.mean(axis=0)) / np.where(std > near_zero_std, std, 1.0)
    return out


def coral_transform(X, subject_ids, shrinkage=0.5, near_zero_std=1e-12):
    Z = subject_zscore(X, subject_ids, near_zero_std)
    out = np.empty_like(Z)
    dimension = Z.shape[1]
    identity = np.eye(dimension)
    for subject in np.unique(subject_ids):
        mask = subject_ids == subject
        values = Z[mask]
        covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
        covariance = (1.0 - shrinkage) * covariance + shrinkage * identity
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, near_zero_std)))) @ eigenvectors.T
        out[mask] = values @ inverse_sqrt
    return out


def model(config, seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=seed))


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, p)), "binary_f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": 0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def grouped_metrics(reference, probabilities):
    y = reference["y_true"].to_numpy()
    result = {"overall": metrics(y, probabilities), "groups": {}, "subject_macro": {}}
    for group in ("HC", "DEP"):
        mask = reference["health_group"].to_numpy() == group
        result["groups"][group] = metrics(y[mask], probabilities[mask])
    rows = []
    for subject in reference["subject_id"].unique():
        mask = reference["subject_id"].to_numpy() == subject
        rows.append(metrics(y[mask], probabilities[mask]))
    result["subject_macro"] = {key: float(np.mean([row[key] for row in rows])) for key in ("roc_auc", "binary_f1", "balanced_accuracy")}
    return result


def assert_reference(reference, outer, config):
    if len(reference) != config["expected_oof_rows"] or reference["subject_id"].nunique() != config["expected_subjects"]:
        raise RuntimeError("unexpected OOF size")
    if set(outer) != set(range(config["outer_folds"])):
        raise RuntimeError("unexpected outer folds")
    for fold in range(config["outer_folds"]):
        train_subjects = set(reference.loc[outer != fold, "subject_id"])
        valid_subjects = set(reference.loc[outer == fold, "subject_id"])
        if train_subjects & valid_subjects:
            raise RuntimeError("subject leakage")


def evaluate(name, X, X_public, reference, outer, cache, config):
    y = reference["y_true"].to_numpy()
    oof = np.full(len(y), np.nan)
    fold_rows = []
    for fold in range(config["outer_folds"]):
        train, valid = outer != fold, outer == fold
        fitted = model(config, config["random_seed"] + fold)
        fitted.fit(X[train], y[train])
        oof[valid] = fitted.predict_proba(X[valid])[:, 1]
        fold_result = grouped_metrics(reference.loc[valid].reset_index(drop=True), oof[valid])
        fold_rows.append({"fold": fold, "n_train_subjects": int(reference.loc[train, "subject_id"].nunique()),
                          "n_validation_subjects": int(reference.loc[valid, "subject_id"].nunique()), **fold_result})
        print(f"{name}: fold {fold + 1}/5", flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError("incomplete OOF predictions")
    output = reference.copy()
    output["prob_positive"] = oof
    output.to_csv(PREDICTIONS_DIR / f"{name}_oof.csv", index=False)
    final = model(config, config["random_seed"])
    final.fit(X, y)
    public = pd.DataFrame({"user_id": cache["public_user_ids"], "trial_id": cache["public_trial_ids"],
                           "prob. of positive emotion": final.predict_proba(X_public)[:, 1]})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
    public.to_csv(PREDICTIONS_DIR / f"{name}_public.csv", index=False)
    return {"n_features": int(X.shape[1]), **grouped_metrics(reference, oof), "folds": fold_rows}


def main():
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache((EXPERIMENT_DIR / config["feature_cache"]).resolve())
    names = np.asarray(cache["feature_names"]).astype(str)
    feature_mask = np.char.startswith(names, "log_bandpower_")
    if feature_mask.sum() != config["n_features"]:
        raise RuntimeError("expected 150 log_bandpower features")
    keep = cache["subject_ids"] != config["excluded_subject"]
    subjects = cache["subject_ids"][keep]
    reference = pd.DataFrame({"subject_id": subjects, "health_group": cache["health_groups"][keep],
                              "original_trial_id": cache["original_trial_ids"][keep], "fold": -1, "y_true": cache["y_train"][keep]})
    outer_reference = pd.read_csv((EXPERIMENT_DIR / config["outer_reference"]).resolve())
    if list(outer_reference.columns) != OOF_COLUMNS or not np.array_equal(outer_reference[["subject_id", "health_group", "original_trial_id"]].to_numpy(), reference[["subject_id", "health_group", "original_trial_id"]].to_numpy()):
        raise RuntimeError("002 outer reference metadata mismatch")
    if not np.array_equal(outer_reference["y_true"].to_numpy(), reference["y_true"].to_numpy()):
        raise RuntimeError("002 outer reference label mismatch")
    outer = outer_reference["fold"].to_numpy(dtype=int)
    reference["fold"] = outer
    assert_reference(reference, outer, config)
    X_raw = cache["X_train"][keep][:, feature_mask]
    X_public_raw = cache["X_public"][:, feature_mask]
    X_z = subject_zscore(X_raw, subjects, config["coral"]["near_zero_std"])
    X_public_z = subject_zscore(X_public_raw, cache["public_user_ids"], config["coral"]["near_zero_std"])
    X_coral = coral_transform(X_raw, subjects, config["coral"]["shrinkage"], config["coral"]["near_zero_std"])
    X_public_coral = coral_transform(X_public_raw, cache["public_user_ids"], config["coral"]["shrinkage"], config["coral"]["near_zero_std"])
    runs = {"baseline_zscore": evaluate("baseline_zscore", X_z, X_public_z, reference, outer, cache, config),
            "symmetric_coral": evaluate("symmetric_coral", X_coral, X_public_coral, reference, outer, cache, config)}
    summary = []
    for name, result in runs.items():
        summary.append({"run_name": name, "n_features": result["n_features"], "oof_auc": result["overall"]["roc_auc"],
                        "oof_f1": result["overall"]["binary_f1"], "oof_balanced_accuracy": result["overall"]["balanced_accuracy"],
                        "hc_auc": result["groups"]["HC"]["roc_auc"], "dep_auc": result["groups"]["DEP"]["roc_auc"],
                        "subject_macro_auc": result["subject_macro"]["roc_auc"], "subject_macro_f1": result["subject_macro"]["binary_f1"]})
    result = {"experiment_config": config, "protocol": {"input_rows": 2760, "public_rows": 80, "outer_folds_reused_from_002": True,
        "excluded_subject": "HC43", "feature_names": names[feature_mask].tolist(), "no_labels_in_transforms": True,
        "symmetric_train_validation_public_transform": True, "no_tuning_or_fusion": True}, "runs": runs}
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    report = "# 实验022：对称CORAL与HGB\n\n严格复用002固定被试5折，输入为001的150维log_bandpower，排除HC43。baseline_zscore与symmetric_coral均先按被试无标签zscore；后者再以固定shrinkage=0.5将协方差收缩至identity并使用逆平方根。训练、验证和公开用户完全同构变换，不使用标签或original_trial_id；模型固定HGB与StandardScaler pipeline，不调参、不融合。\n\n## 真实结果\n\n| 条件 | OOF AUC | OOF F1 | HC AUC | DEP AUC | 被试macro AUC | 被试macro F1 |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for row in summary:
        report += f"| {row['run_name']} | {row['oof_auc']:.6f} | {row['oof_f1']:.6f} | {row['hc_auc']:.6f} | {row['dep_auc']:.6f} | {row['subject_macro_auc']:.6f} | {row['subject_macro_f1']:.6f} |\n"
    report += "\n每个条件生成2760行OOF和80行公开预测；逐折结果详见metrics.json。\n"
    (EXPERIMENT_DIR / "实验报告.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
