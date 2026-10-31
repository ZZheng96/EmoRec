import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[2]
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
OOF_COLUMNS = ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"]


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, p)), "binary_f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": 0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def subject_folds(frame, n_splits, seed):
    subjects = frame["subject_id"].drop_duplicates().to_numpy()
    labels = frame.groupby("subject_id", sort=False)["y_true"].first().reindex(subjects).to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    by_subject = {}
    for fold, (_, val) in enumerate(splitter.split(subjects, labels)):
        for subject in subjects[val]:
            by_subject[subject] = fold
    return np.array([by_subject[s] for s in frame["subject_id"]], dtype=int)


def model(config, seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(
        **config["hgb"], random_state=seed))


def baseline_model(config, seed):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=config["baseline_lr"]["C"], penalty=config["baseline_lr"]["penalty"],
        max_iter=config["baseline_lr"]["max_iter"], solver=config["baseline_lr"]["solver"],
        random_state=seed))


def load_oof(path, reference=None):
    frame = pd.read_csv(path)
    if list(frame.columns) != OOF_COLUMNS:
        raise RuntimeError(f"Unexpected OOF columns in {path}")
    if reference is not None:
        for key in OOF_COLUMNS[:-1]:
            if not np.array_equal(frame[key].to_numpy(), reference[key].to_numpy()):
                raise RuntimeError(f"OOF alignment mismatch for {path.name}: {key}")
    if not np.isfinite(frame["prob_positive"]).all():
        raise RuntimeError(f"Non-finite probabilities in {path}")
    return frame


def choose_weight(y, baseline, hgb, mask, candidates):
    scores = []
    for weight in candidates:
        p = weight * baseline + (1 - weight) * hgb
        score = metrics(y[mask], p[mask])
        scores.append((score["roc_auc"], score["binary_f1"], -weight, weight))
    best = max(scores)
    return best[3], scores


def write_prediction(name, reference, probabilities):
    output = reference.copy()
    output["prob_positive"] = probabilities
    output.to_csv(PREDICTIONS_DIR / f"{name}_oof.csv", index=False)


def main():
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with np.load((EXPERIMENT_DIR / config["feature_cache"]).resolve()) as cache:
        all_names = cache["feature_names"]
        feature_mask = np.asarray([str(name).startswith("log_bandpower_") for name in all_names])
        X_all = cache["X_train"][:, feature_mask]
        X_public_all = cache["X_public"][:, feature_mask]
        names = all_names[feature_mask]
        subjects_all, health_all = cache["subject_ids"], cache["health_groups"]
        trials_all, y_all = cache["original_trial_ids"], cache["y_train"]
        public_users, public_trials = cache["public_user_ids"], cache["public_trial_ids"]
    keep = subjects_all != config["excluded_subject"]
    if X_all.shape[1] != config["n_features"] or len(np.unique(subjects_all[keep])) != 69:
        raise RuntimeError("Expected 150 features and 69 subjects after HC43 exclusion")
    X = normalize_by_subject(X_all[keep], subjects_all[keep], "zscore", 1e-12)
    X_public = normalize_by_subject(X_public_all, public_users, "zscore", 1e-12)
    reference = pd.DataFrame({"subject_id": subjects_all[keep], "health_group": health_all[keep],
                              "original_trial_id": trials_all[keep], "fold": -1, "y_true": y_all[keep]})
    outer_reference = load_oof(EXPERIMENT_DIR.parent / "002_subject_normalization" / "predictions" / "subject_zscore_exclude_hc43_oof.csv")
    if not np.array_equal(outer_reference[["subject_id", "health_group", "original_trial_id", "y_true"]].to_numpy(), reference[["subject_id", "health_group", "original_trial_id", "y_true"]].to_numpy()):
        raise RuntimeError("002 outer-fold reference alignment mismatch")
    outer = outer_reference["fold"].to_numpy()
    reference["fold"] = outer
    y = reference["y_true"].to_numpy()
    baseline_oof = np.full(len(y), np.nan)
    hgb_oof = np.full(len(y), np.nan)
    nested = np.full(len(y), np.nan)
    fixed = np.full(len(y), np.nan)
    candidates = np.arange(0.0, 1.01, 0.1)
    fold_rows, candidate_rows = [], []
    for fold in range(config["outer_folds"]):
        train, val = outer != fold, outer == fold
        inner_frame = reference.loc[train].reset_index(drop=True)
        inner_folds = subject_folds(inner_frame, config["inner_folds"], config["random_seed"] + fold)
        baseline_inner_oof = np.full(train.sum(), np.nan)
        hgb_inner_oof = np.full(train.sum(), np.nan)
        for inner in range(config["inner_folds"]):
            inner_train, inner_val = inner_folds != inner, inner_folds == inner
            lr_fitted = baseline_model(config, config["random_seed"] + fold * 10 + inner)
            hgb_fitted = model(config, config["random_seed"] + fold * 10 + inner)
            lr_fitted.fit(X[train][inner_train], y[train][inner_train])
            hgb_fitted.fit(X[train][inner_train], y[train][inner_train])
            baseline_inner_oof[inner_val] = lr_fitted.predict_proba(X[train][inner_val])[:, 1]
            hgb_inner_oof[inner_val] = hgb_fitted.predict_proba(X[train][inner_val])[:, 1]
        if not np.isfinite(baseline_inner_oof).all() or not np.isfinite(hgb_inner_oof).all():
            raise RuntimeError("Incomplete inner baseline/HGB OOF")
        weight, scores = choose_weight(y[train], baseline_inner_oof, hgb_inner_oof, np.ones(train.sum(), dtype=bool), candidates)
        lr_fitted = baseline_model(config, config["random_seed"] + fold)
        hgb_fitted = model(config, config["random_seed"] + fold)
        lr_fitted.fit(X[train], y[train])
        hgb_fitted.fit(X[train], y[train])
        baseline_oof[val] = lr_fitted.predict_proba(X[val])[:, 1]
        hgb_oof[val] = hgb_fitted.predict_proba(X[val])[:, 1]
        nested[val] = weight * baseline_oof[val] + (1 - weight) * hgb_oof[val]
        fixed[val] = config["fixed_baseline_weight"] * baseline_oof[val] + (1 - config["fixed_baseline_weight"]) * hgb_oof[val]
        selected = next(row for row in scores if row[3] == weight)
        fold_rows.append({"fold": fold, "n_train_subjects": int(reference.loc[train, "subject_id"].nunique()),
                          "n_validation_subjects": int(reference.loc[val, "subject_id"].nunique()),
                          "selected_baseline_weight": float(weight), "selected_hgb_weight": float(1 - weight),
                          "inner_auc": selected[0], "inner_binary_f1": selected[1],
                          "outer_validation_baseline": metrics(y[val], baseline_oof[val]),
                          "outer_validation_hgb": metrics(y[val], hgb_oof[val]),
                          "outer_validation_fusion": metrics(y[val], nested[val]),
                          "outer_validation_fixed_fusion": metrics(y[val], fixed[val])})
        for auc, f1, _, candidate in scores:
            candidate_rows.append({"outer_fold": fold, "baseline_weight": candidate, "hgb_weight": 1 - candidate,
                                   "inner_auc": auc, "inner_binary_f1": f1})
    if not np.isfinite(baseline_oof).all() or not np.isfinite(hgb_oof).all() or not np.isfinite(nested).all() or not np.isfinite(fixed).all():
        raise RuntimeError("Incomplete outer OOF predictions")
    write_prediction("baseline_lr", reference, baseline_oof)
    write_prediction("hgb", reference, hgb_oof)
    write_prediction("nested_hgb_fusion", reference, nested)
    write_prediction("fixed_hgb_fusion", reference, fixed)
    final_lr = baseline_model(config, config["random_seed"])
    final_lr.fit(X, y)
    final = model(config, config["random_seed"])
    final.fit(X, y)
    public_base = final_lr.predict_proba(X_public)[:, 1]
    public_hgb = final.predict_proba(X_public)[:, 1]
    public_weight = float(np.mean([row["selected_baseline_weight"] for row in fold_rows]))
    public_fusion = public_weight * public_base + (1 - public_weight) * public_hgb
    public_fixed = config["fixed_baseline_weight"] * public_base + (1 - config["fixed_baseline_weight"]) * public_hgb
    public = pd.DataFrame({"user_id": public_users, "trial_id": public_trials, "prob. of positive emotion": public_hgb})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
    public.to_csv(PREDICTIONS_DIR / "hgb_public.csv", index=False)
    diagnostic = public.copy()
    diagnostic["prob. of positive emotion"] = public_fusion
    diagnostic.to_csv(PREDICTIONS_DIR / "diagnostic_fusion_public.csv", index=False)
    diagnostic["prob. of positive emotion"] = public_fixed
    diagnostic.to_csv(PREDICTIONS_DIR / "fixed_fusion_public.csv", index=False)
    runs = {"baseline_lr": metrics(y, baseline_oof), "hgb": metrics(y, hgb_oof),
            "nested_hgb_fusion": metrics(y, nested), "fixed_hgb_fusion": metrics(y, fixed)}
    results = {"experiment_config": config, "protocol": {
        "outer_folds_reused_from_002": True, "excluded_subject": "HC43",
        "input": "002 150-dimensional log-bandpower, per-subject unlabeled zscore",
        "scaler": "StandardScaler fitted separately on each outer training fold",
        "inner_folds": "subject-wise stratified 3-fold within each outer training fold",
        "baseline_inner_oof": True, "hgb_inner_oof": True, "outer_validation_labels_used_for_weight": False,
        "selection": "AUC highest, ties by F1; weights 0.0..1.0 step 0.1",
        "baseline_protocol": "same 150-dimensional subject-zscore features and LR protocol as experiment 002",
        "public_fusion": "diagnostic only; mean selected baseline weight; not used as primary result"},
        "hgb_parameters": config["hgb"], "folds": fold_rows, "candidate_scores": candidate_rows,
        "runs": runs, "comparison": {"current_auc": config["comparison"]["current_auc"],
        "current_f1": config["comparison"]["current_f1"],
        "old_015_reference_status": "exploratory_only_non_strict_baseline_probabilities",
        "hgb_delta_auc": runs["hgb"]["roc_auc"] - config["comparison"]["current_auc"],
        "hgb_delta_f1": runs["hgb"]["binary_f1"] - config["comparison"]["current_f1"],
        "fusion_delta_auc": runs["nested_hgb_fusion"]["roc_auc"] - config["comparison"]["current_auc"],
        "fusion_delta_f1": runs["nested_hgb_fusion"]["binary_f1"] - config["comparison"]["current_f1"]}}
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"run_name": k, "oof_auc": v["roc_auc"], "oof_binary_f1": v["binary_f1"], "oof_balanced_accuracy": v["balanced_accuracy"]} for k, v in runs.items()]).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    report = f"""# 实验016：HGB Fusion

## 结论

严格嵌套协议：HGB独立OOF AUC={runs['hgb']['roc_auc']:.6f}、F1={runs['hgb']['binary_f1']:.6f}；嵌套融合 AUC={runs['nested_hgb_fusion']['roc_auc']:.6f}、F1={runs['nested_hgb_fusion']['binary_f1']:.6f}；固定0.9/0.1融合 AUC={runs['fixed_hgb_fusion']['roc_auc']:.6f}、F1={runs['fixed_hgb_fusion']['binary_f1']:.6f}。旧015概率未用于新实验，其结果仅作探索性参考，不能作为最佳结果。

## 协议

输入为002的150维log-bandpower，按被试无标签zscore；外层折沿用002且逐列对齐。每个外层训练折内部按被试3折分别生成LR baseline与HGB内层OOF，按AUC最高、并列F1最高选择0.0至1.0权重，再用完整outer-train重训两模型预测outer-val。公开集只输出独立HGB及诊断融合，不以公开集替代OOF结论。详见 `metrics.json`、`summary.csv` 和 `predictions/`。
"""
    (EXPERIMENT_DIR / "实验报告.md").write_text(report, encoding="utf-8")
    print(pd.read_csv(EXPERIMENT_DIR / "summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
