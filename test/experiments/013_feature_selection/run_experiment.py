import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
BASE_DIR = EXPERIMENT_DIR.parent / "002_subject_normalization"
spec = importlib.util.spec_from_file_location("feature_selection_baseline", BASE_DIR / "run_experiment.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
K_VALUES = (20, 50, 100, 150)


def select_features(X_train, y_train, k):
    if not 1 <= k <= X_train.shape[1]:
        raise ValueError("k must be between 1 and the number of features")
    scores = np.asarray([abs(roc_auc_score(y_train, column) - 0.5) for column in X_train.T])
    # Stable ties favor the original feature index; retain original column order.
    selected = np.sort(np.argsort(-scores, kind="stable")[:k])
    return selected, scores


def fit_predict(X_train, y_train, train_subjects, X_target, target_subjects, k, config):
    if set(train_subjects) & set(target_subjects):
        raise ValueError("Training and target subjects must be disjoint")
    epsilon = config["normalization"]["near_zero_std"]
    train = normalize_by_subject(X_train, train_subjects, "zscore", near_zero_std=epsilon)
    target = normalize_by_subject(X_target, target_subjects, "zscore", near_zero_std=epsilon)
    selected, scores = select_features(train, y_train, k)
    model = baseline.make_model(config["model"], config["random_seed"])
    model.fit(train[:, selected], y_train)
    return model.predict_proba(target[:, selected])[:, 1], selected, scores


def main():
    config = json.loads((BASE_DIR / "config.json").read_text(encoding="utf-8"))
    data = baseline.load_feature_cache((BASE_DIR / config["feature_cache"]).resolve())
    folds = baseline.make_subject_folds(data["subject_ids"], data["health_groups"], config["random_seed"])
    X, names = baseline.select_log_bandpower(data["X_train"], data["feature_names"])
    X_public, _ = baseline.select_log_bandpower(data["X_public"], data["feature_names"])
    keep = data["subject_ids"] != "HC43"
    X, folds = X[keep], folds[keep]
    filtered = {key: data[key][keep] for key in baseline.TRAIN_DATA_KEYS}
    y, subjects = filtered["y_train"], filtered["subject_ids"]
    assert X.shape[1] == X_public.shape[1] == 150
    assert len(np.unique(subjects)) == 69
    reference = json.loads((BASE_DIR / "metrics.json").read_text(encoding="utf-8"))["runs"]["subject_zscore_exclude_hc43"]["overall"]
    results = {
        "experiment_id": "013_feature_selection",
        "protocol": {
            "seed": config["random_seed"], "n_splits": 5,
            "fold_assignment": "002: stratify all 70 subjects by health group, then exclude HC43",
            "feature_cache": str((BASE_DIR / config["feature_cache"]).resolve()),
            "normalization": "independent label-free subject zscore in training and target partitions",
            "selection": "training-partition-only abs(univariate ROC AUC - 0.5); stable index ties",
            "k_values": list(K_VALUES), "k_tuning": False,
            "model": config["model"], "threshold": 0.5,
            "public_fit": "select and fit on all 69 labeled training subjects only; no public labels",
            "evaluation_unit": "cached training segments (same as 002), not trial-aggregated",
        }, "baseline_002": reference, "runs": {},
    }
    predictions_dir = EXPERIMENT_DIR / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    rows = []
    for k in K_VALUES:
        name = f"auc_top{k}"
        oof = np.full(len(y), np.nan)
        fold_results = []
        for fold in range(5):
            train, valid = folds != fold, folds == fold
            probabilities, selected, scores = fit_predict(
                X[train], y[train], subjects[train], X[valid], subjects[valid], k, config
            )
            oof[valid] = probabilities
            fold_results.append({
                "fold": fold, "train_subjects": np.unique(subjects[train]).tolist(),
                "validation_subjects": np.unique(subjects[valid]).tolist(),
                "selected_indices": selected.tolist(), "selected_features": names[selected].tolist(),
                "training_feature_scores": scores.tolist(),
                **baseline.classification_metrics(y[valid], probabilities),
            })
            print(f"{name}: fold {fold + 1}/5", flush=True)
        assert np.isfinite(oof).all()
        pd.DataFrame({
            "subject_id": subjects, "health_group": filtered["health_groups"],
            "original_trial_id": filtered["original_trial_ids"], "fold": folds,
            "y_true": y, "prob_positive": oof,
        }).to_csv(predictions_dir / f"{name}_oof.csv", index=False)
        public_probabilities, selected, scores = fit_predict(
            X, y, subjects, X_public, data["public_user_ids"], k, config
        )
        public = pd.DataFrame({
            "user_id": data["public_user_ids"], "trial_id": data["public_trial_ids"],
            "prob. of positive emotion": public_probabilities,
        })
        validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
        assert len(public) == 80
        public.to_csv(predictions_dir / f"{name}_public.csv", index=False)
        overall = baseline.classification_metrics(y, oof)
        result = {
            "n_subjects": 69, "n_samples": len(y), "n_features": k,
            "overall": overall, "folds": fold_results,
            "public_selected_indices": selected.tolist(), "public_selected_features": names[selected].tolist(),
            "public_training_feature_scores": scores.tolist(), "public_rows": len(public),
            "auc_delta_vs_002": overall["roc_auc"] - reference["roc_auc"],
            "f1_delta_vs_002": overall["binary_f1"] - reference["binary_f1"],
        }
        results["runs"][name] = result
        rows.append({"run_name": name, "k": k, "oof_auc": overall["roc_auc"],
                     "oof_binary_f1": overall["binary_f1"],
                     "oof_balanced_accuracy": overall["balanced_accuracy"],
                     "auc_delta_vs_002": result["auc_delta_vs_002"],
                     "f1_delta_vs_002": result["f1_delta_vs_002"]})
        (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
