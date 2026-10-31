import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

EXPERIMENT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, p)), "binary_f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": 0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def subject_folds(frame, n_splits, seed):
    subjects = frame["subject_id"].drop_duplicates().to_numpy()
    label_column = "health_group" if "health_group" in frame else "y_true"
    labels = frame.groupby("subject_id", sort=False)[label_column].first().reindex(subjects).to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = np.full(len(frame), -1, dtype=int)
    by_subject = {}
    for fold, (_, val) in enumerate(splitter.split(subjects, labels)):
        for subject in subjects[val]:
            by_subject[subject] = fold
    return np.array([by_subject[s] for s in frame["subject_id"]], dtype=int)


def load_oof(path, reference=None):
    frame = pd.read_csv(path)
    required = ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"]
    if list(frame.columns) != required:
        raise RuntimeError(f"Unexpected OOF columns in {path}")
    if reference is not None:
        for key in required[:-1]:
            if not np.array_equal(frame[key].to_numpy(), reference[key].to_numpy()):
                raise RuntimeError(f"OOF alignment mismatch for {path.name}: {key}")
    if not np.isfinite(frame["prob_positive"]).all():
        raise RuntimeError(f"Non-finite probabilities in {path}")
    return frame


def choose_weight(y, baseline, auxiliary, train_mask, candidates):
    scores = []
    aux = np.mean(auxiliary, axis=0)
    for weight in candidates:
        p = weight * baseline + (1 - weight) * aux
        score = metrics(y[train_mask], p[train_mask])
        scores.append((score["roc_auc"], score["binary_f1"], -weight, weight))
    return max(scores)[3], scores


def public_weight(config, frame, baseline, auxiliary):
    folds = subject_folds(frame, config["inner_folds"], config["random_seed"] + 1000)
    return choose_weight(frame["y_true"].to_numpy(), baseline, auxiliary, folds != -1, config["candidate_weights"])[0]


def write_prediction(name, reference, probabilities):
    output = reference.copy()
    output["prob_positive"] = probabilities
    output.to_csv(PREDICTIONS_DIR / f"{name}_oof.csv", index=False)


def main():
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    base_path = (EXPERIMENT_DIR / config["sources"]["baseline"]).resolve()
    baseline_frame = load_oof(base_path)
    if len(baseline_frame["subject_id"].unique()) != config["expected_subjects"]:
        raise RuntimeError("Expected 69 subjects after HC43 exclusion")
    aux_frames = [load_oof((EXPERIMENT_DIR / path).resolve(), baseline_frame) for path in config["sources"]["auxiliary"]]
    y = baseline_frame["y_true"].to_numpy()
    outer = baseline_frame["fold"].to_numpy()
    if set(outer) != set(range(config["outer_folds"])):
        raise RuntimeError("Baseline folds are not the expected reused 5 folds")
    baseline = baseline_frame["prob_positive"].to_numpy()
    auxiliary = np.vstack([f["prob_positive"].to_numpy() for f in aux_frames])
    aux_mean = auxiliary.mean(axis=0)
    nested = np.full(len(y), np.nan)
    fixed = 0.9 * baseline + 0.1 * aux_mean
    fold_rows = []
    candidate_rows = []
    for fold in range(config["outer_folds"]):
        train = outer != fold
        val = outer == fold
        inner_frame = baseline_frame.loc[train].reset_index(drop=True)
        inner = subject_folds(inner_frame, config["inner_folds"], config["random_seed"] + fold)
        weight, scores = choose_weight(y[train], baseline[train], auxiliary[:, train], inner != -1, config["candidate_weights"])
        nested[val] = weight * baseline[val] + (1 - weight) * aux_mean[val]
        selected = next(row for row in scores if row[3] == weight)
        fold_rows.append({"fold": fold, "n_train_subjects": int(baseline_frame.loc[train, "subject_id"].nunique()), "n_validation_subjects": int(baseline_frame.loc[val, "subject_id"].nunique()), "selected_baseline_weight": weight, "selected_auxiliary_mean_weight": 1 - weight, "inner_auc": selected[0], "inner_binary_f1": selected[1], "outer_validation": metrics(y[val], nested[val])})
        for auc, f1, _, candidate in scores:
            candidate_rows.append({"outer_fold": fold, "baseline_weight": candidate, "auxiliary_mean_weight": 1 - candidate, "inner_auc": auc, "inner_binary_f1": f1})
    if not np.isfinite(nested).all():
        raise RuntimeError("Nested OOF is incomplete")
    write_prediction("nested_fusion", baseline_frame, nested)
    fixed_frame = baseline_frame.copy()
    fixed_frame["prob_positive"] = fixed
    fixed_frame.to_csv(PREDICTIONS_DIR / "fixed_fusion_0.9_0.1_oof.csv", index=False)
    public_base = pd.read_csv((EXPERIMENT_DIR / config["public_sources"]["baseline"]).resolve())
    public_aux = [pd.read_csv((EXPERIMENT_DIR / p).resolve()) for p in config["public_sources"]["auxiliary"]]
    for frame in public_aux:
        if not public_base[["user_id", "trial_id"]].equals(frame[["user_id", "trial_id"]]):
            raise RuntimeError("Public prediction key alignment mismatch")
    public_weight_value = public_weight(config, baseline_frame, baseline, auxiliary)
    public_p = public_weight_value * public_base["prob. of positive emotion"].to_numpy() + (1 - public_weight_value) * np.mean([f["prob. of positive emotion"].to_numpy() for f in public_aux], axis=0)
    public = public_base[["user_id", "trial_id"]].copy()
    public["prob. of positive emotion"] = public_p
    public.to_csv(PREDICTIONS_DIR / "nested_fusion_public.csv", index=False)
    fixed_public = public_base[["user_id", "trial_id"]].copy()
    fixed_public["prob. of positive emotion"] = 0.9 * public_base["prob. of positive emotion"] + 0.1 * np.mean([f["prob. of positive emotion"] for f in public_aux], axis=0)
    fixed_public.to_csv(PREDICTIONS_DIR / "fixed_fusion_0.9_0.1_public.csv", index=False)
    runs = {"002_subject_zscore": metrics(y, baseline), "nested_fusion": metrics(y, nested), "fixed_fusion_0.9_0.1": metrics(y, fixed)}
    results = {"experiment_config": config, "protocol": {"outer_folds_reused_from_002": True, "inner_folds": "subject-wise stratified 3-fold within each outer training fold", "selection": "inner OOF AUC, ties by F1, candidates 0.0..1.0 step 0.1", "outer_validation_labels_used_for_weight": False, "public_weight": public_weight_value}, "folds": fold_rows, "candidate_scores": candidate_rows, "runs": runs, "baseline_comparison": {"baseline_auc": config["baseline_auc"], "baseline_f1": config["baseline_f1"], "nested_delta_auc": runs["nested_fusion"]["roc_auc"] - config["baseline_auc"], "nested_delta_f1": runs["nested_fusion"]["binary_f1"] - config["baseline_f1"]}}
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"run_name": k, "oof_auc": v["roc_auc"], "oof_binary_f1": v["binary_f1"], "oof_balanced_accuracy": v["balanced_accuracy"]} for k, v in runs.items()]).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    report = f"""# 实验015：Nested Fusion\n\n## 结论\n\n复用002的69名被试外层五折；每个外层训练折内部按被试三折，仅用内层OOF选择候选权重。nested OOF AUC={runs['nested_fusion']['roc_auc']:.6f}、F1={runs['nested_fusion']['binary_f1']:.6f}；002基线AUC={config['baseline_auc']:.6f}、F1={config['baseline_f1']:.6f}。AUC差值={results['baseline_comparison']['nested_delta_auc']:.6f}，F1差值={results['baseline_comparison']['nested_delta_f1']:.6f}。请以差值如实判断，不以外层标签选权重。\n\n固定0.9/0.1对照：AUC={runs['fixed_fusion_0.9_0.1']['roc_auc']:.6f}、F1={runs['fixed_fusion_0.9_0.1']['binary_f1']:.6f}。nested与固定对照均来自同一批严格对齐的002、004、006 OOF。\n\n## 协议\n\n预测样本键（subject_id、health_group、original_trial_id、fold、y_true）逐列严格一致；公开预测的user_id/trial_id也逐文件检查。候选baseline权重为0.0至1.0、步长0.1，内层OOF按AUC最高、并列按F1选择。公开权重仅用全训练集的无外层标签流程诊断产生，不能替代真正nested OOF结论。\n\n每个外层折的权重、内层指标和外层验证指标见 `metrics.json`；预测见 `predictions/`；汇总见 `summary.csv`。\n"""
    (EXPERIMENT_DIR / "实验报告.md").write_text(report, encoding="utf-8")
    print(pd.read_csv(EXPERIMENT_DIR / "summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
