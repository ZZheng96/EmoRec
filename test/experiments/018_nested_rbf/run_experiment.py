import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
OOF_COLUMNS = ["subject_id", "health_group", "original_trial_id", "fold", "y_true"]
CANDIDATES = tuple((c, g) for c in (0.1, 1.0, 10.0) for g in (0.001, 0.006666666666666667, 0.03))


def load_cache(path):
    with np.load(path) as cache:
        return {key: cache[key] for key in cache.files}


def load_outer_reference(path, reference):
    frame = pd.read_csv(path)
    expected = OOF_COLUMNS + ["prob_positive"]
    if list(frame.columns) != expected:
        raise RuntimeError("Unexpected outer reference columns")
    for key in OOF_COLUMNS[:3] + ["y_true"]:
        if not np.array_equal(frame[key].to_numpy(), reference[key].to_numpy()):
            raise RuntimeError(f"outer metadata alignment mismatch: {key}")
    return frame["fold"].to_numpy(dtype=int)


def assert_subject_isolation(subjects, folds, n_folds=5):
    for fold in range(n_folds):
        train_subjects = set(subjects[folds != fold])
        valid_subjects = set(subjects[folds == fold])
        if train_subjects & valid_subjects:
            raise RuntimeError(f"subject leakage in fold {fold}")


def make_inner_folds(subjects, health_groups, seed):
    unique = np.unique(subjects)
    health = np.asarray([health_groups[np.flatnonzero(subjects == s)[0]] for s in unique])
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    by_subject = {}
    for fold, (_, val) in enumerate(splitter.split(unique, health)):
        for subject in unique[val]:
            by_subject[subject] = fold
    folds = np.asarray([by_subject[s] for s in subjects], dtype=int)
    assert_subject_isolation(subjects, folds, 3)
    return folds


def model(c, gamma):
    return make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma, probability=False, kernel="rbf"))


def inner_select(X, y, subjects, health, seed):
    folds = make_inner_folds(subjects, health, seed)
    rows = []
    oof_by_candidate = {}
    for c, gamma in CANDIDATES:
        margins = np.full(len(y), np.nan)
        fold_auc = []
        for fold in range(3):
            train, valid = folds != fold, folds == fold
            fitted = model(c, gamma)
            fitted.fit(X[train], y[train])
            margins[valid] = fitted.decision_function(X[valid])
            fold_auc.append(float(roc_auc_score(y[valid], margins[valid])))
        if not np.isfinite(margins).all():
            raise RuntimeError("incomplete inner OOF margins")
        mean_auc = float(np.mean(fold_auc))
        rows.append({"C": c, "gamma": gamma, "fold_auc": fold_auc, "mean_auc": mean_auc})
        oof_by_candidate[(c, gamma)] = margins
    selected_row = max(rows, key=lambda row: (row["mean_auc"], -CANDIDATES.index((row["C"], row["gamma"]))))
    selected = (selected_row["C"], selected_row["gamma"])
    return {"folds": folds, "candidates": rows, "selected": {"C": selected[0], "gamma": selected[1], "mean_auc": selected_row["mean_auc"]}, "selected_margins": oof_by_candidate[selected]}


def calibrate(margins, y):
    return LogisticRegression(C=1, solver="lbfgs", max_iter=1000).fit(np.asarray(margins).reshape(-1, 1), y)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, p)), "binary_f1": float(f1_score(y, pred, zero_division=0)), "threshold": 0.5, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def subject_macro(y, p, subjects):
    rows = [metrics(y[subjects == s], p[subjects == s]) for s in np.unique(subjects)]
    return {"auc_mean": float(np.mean([r["roc_auc"] for r in rows])), "auc_std": float(np.std([r["roc_auc"] for r in rows], ddof=1)), "f1_mean": float(np.mean([r["binary_f1"] for r in rows])), "f1_std": float(np.std([r["binary_f1"] for r in rows], ddof=1))}


def evaluate(X, X_public, reference, folds, config):
    y = reference["y_true"].to_numpy(dtype=int)
    subjects = reference["subject_id"].to_numpy()
    health = reference["health_group"].to_numpy()
    oof = np.full(len(y), np.nan)
    fold_rows, selection_rows = [], []
    for outer in range(config["outer_folds"]):
        train, valid = folds != outer, folds == outer
        selected = inner_select(X[train], y[train], subjects[train], health[train], config["random_seed"] + outer)
        calibrator = calibrate(selected["selected_margins"], y[train])
        final_svc = model(selected["selected"]["C"], selected["selected"]["gamma"])
        final_svc.fit(X[train], y[train])
        valid_margin = final_svc.decision_function(X[valid])
        oof[valid] = calibrator.predict_proba(valid_margin.reshape(-1, 1))[:, 1]
        selection_rows.append({"outer_fold": outer, **selected["selected"], "candidates": selected["candidates"]})
        row = {"fold": outer, "n_train_subjects": int(np.unique(subjects[train]).size), "n_validation_subjects": int(np.unique(subjects[valid]).size), "overall": metrics(y[valid], oof[valid])}
        for group in ("HC", "DEP"):
            mask = valid & (health == group)
            row[group] = metrics(y[mask], oof[mask])
        fold_rows.append(row)
    if not np.isfinite(oof).all():
        raise RuntimeError("incomplete outer OOF predictions")
    output = reference.copy()
    output["prob_positive"] = oof
    output.to_csv(PREDICTIONS_DIR / "nested_rbf_oof.csv", index=False)

    public_selection = inner_select(X, y, subjects, health, config["random_seed"] + config["public_seed_offset"])
    public_calibrator = calibrate(public_selection["selected_margins"], y)
    public_svc = model(public_selection["selected"]["C"], public_selection["selected"]["gamma"])
    public_svc.fit(X, y)
    public_margin = public_svc.decision_function(X_public)
    public_prob = public_calibrator.predict_proba(public_margin.reshape(-1, 1))[:, 1]
    public = pd.DataFrame({"user_id": reference.attrs["public_user_ids"], "trial_id": reference.attrs["public_trial_ids"], "prob. of positive emotion": public_prob})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
    public.to_csv(PREDICTIONS_DIR / "nested_rbf_public.csv", index=False)
    group_metrics = {g: metrics(y[health == g], oof[health == g]) for g in ("HC", "DEP")}
    return {"overall": metrics(y, oof), "folds": fold_rows, "groups": group_metrics, "subject_macro": subject_macro(y, oof, subjects), "fold_auc_mean": float(np.mean([r["overall"]["roc_auc"] for r in fold_rows])), "fold_auc_std": float(np.std([r["overall"]["roc_auc"] for r in fold_rows], ddof=1)), "fold_f1_mean": float(np.mean([r["overall"]["binary_f1"] for r in fold_rows])), "fold_f1_std": float(np.std([r["overall"]["binary_f1"] for r in fold_rows], ddof=1)), "outer_selection": selection_rows, "public_selection": {"selected": public_selection["selected"], "candidates": public_selection["candidates"]}}


def main():
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache((EXPERIMENT_DIR / config["feature_cache"]).resolve())
    keep = cache["subject_ids"] != config["excluded_subject"]
    subject_ids, health = cache["subject_ids"][keep], cache["health_groups"][keep]
    reference = pd.DataFrame({"subject_id": subject_ids, "health_group": health, "original_trial_id": cache["original_trial_ids"][keep], "fold": -1, "y_true": cache["y_train"][keep]})
    folds = load_outer_reference((EXPERIMENT_DIR / config["outer_reference"]).resolve(), reference)
    reference["fold"] = folds
    assert_subject_isolation(subject_ids, folds)
    reference.attrs["public_user_ids"] = cache["public_user_ids"]
    reference.attrs["public_trial_ids"] = cache["public_trial_ids"]
    X = normalize_by_subject(cache["X_train"][keep, :150], subject_ids, "zscore", 1e-12)
    X_public = normalize_by_subject(cache["X_public"][:, :150], cache["public_user_ids"], "zscore", 1e-12)
    result = evaluate(X, X_public, reference, folds, config)
    result["n_features"] = int(X.shape[1])
    result["n_samples"] = int(len(y := reference["y_true"]))
    result["n_subjects"] = int(np.unique(subject_ids).size)
    result["protocol"] = {"outer_folds_reused_from_002": True, "subject_zscore": True, "excluded_subject": "HC43", "svc_probability": False, "inner_selection": "3-fold subject-level health stratified mean AUC", "calibration": "LogisticRegression(C=1) on selected inner OOF margins", "threshold": 0.5, "no_fusion": True, "selection_bias_note": "内层选择后的校准训练分数有选择偏差，外层评估严格独立"}
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps({"experiment_config": config, "run": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"oof_auc": result["overall"]["roc_auc"], "oof_f1": result["overall"]["binary_f1"], "hc_auc": result["groups"]["HC"]["roc_auc"], "dep_auc": result["groups"]["DEP"]["roc_auc"], "subject_macro_auc": result["subject_macro"]["auc_mean"], "subject_macro_f1": result["subject_macro"]["f1_mean"], "fold_auc_mean": result["fold_auc_mean"], "fold_auc_std": result["fold_auc_std"], "fold_f1_mean": result["fold_f1_mean"], "fold_f1_std": result["fold_f1_std"]}
    pd.DataFrame([summary]).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    report = f"""# 实验018：Nested RBF SVC

## 协议
沿用017读取 `001_spectral_logpower/cache/features.npz` 的前150维、按被试无标签 zscore、剔除HC43，并逐行复用002的5折OOF元数据；严格断言样本、公开集ID和被试隔离。每个外层训练集内部按被试health分层、seed=2026+外层折的StratifiedKFold(3)产生OOF decision margin。固定候选SVC(probability=False)：C={{0.1,1,10}}、gamma={{0.001,0.006666666666666667,0.03}}；每候选按3折AUC均值选择，严格按候选顺序打tie。选定候选的内层OOF margins训练LogisticRegression(C=1) sigmoid校准，然后外层训练重fit SVC并校准外层margin。公开模型在全部训练被试上重新进行内层选择和校准，使用全部训练标签，但不依据外层评估成绩选择参数；不使用公开测试标签。

固定阈值0.5，无融合。内层选择后的校准训练分数有选择偏差，但外层评估严格独立。所有候选内层分数、选择参数和预测配置均保存在 `metrics.json` 与 `predictions/`。

## 实际结果
总体ROC AUC={summary['oof_auc']:.6f}，F1={summary['oof_f1']:.6f}；HC AUC={summary['hc_auc']:.6f}，DEP AUC={summary['dep_auc']:.6f}；subject macro AUC={summary['subject_macro_auc']:.6f}，F1={summary['subject_macro_f1']:.6f}；外层fold AUC均值±std={summary['fold_auc_mean']:.6f}±{summary['fold_auc_std']:.6f}，F1均值±std={summary['fold_f1_mean']:.6f}±{summary['fold_f1_std']:.6f}。完整折别指标、混淆矩阵见metrics.json。目标未达时不夸大结果。

## 局限
重复使用相同外层折进行实验比较会产生筛选偏差。被试zscore使用目标被试无标签数据，属于传导式处理；训练每人40窗口、公开每人8窗口，统计量精度存在分布失配。评价单位为10秒，不能与50秒聚合分数混用。相对016频谱HGB（0.782416/0.718671），本轮AUC略升而F1下降；未同时改进，更未达到双0.95。
"""
    (EXPERIMENT_DIR / "实验报告.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
