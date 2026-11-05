import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def score(y, p):
    return {"auc": auc(y, p), "f1": float(f1_score(y, p >= CONFIG["threshold"], zero_division=0)), "n": int(len(y)), "positive": int(np.sum(y))}


def load_inputs():
    plus = pd.read_csv(HERE / CONFIG["base_plus_pe_oof"])
    graph = pd.read_csv(HERE / CONFIG["base_graph_oof"])
    keys = ["subject_id", "health_group", "original_trial_id", "fold", "y_true"]
    if not plus[keys].equals(graph[keys]):
        raise RuntimeError("057/006 OOF alignment mismatch")
    frame = plus[keys].copy()
    frame["base_probability"] = CONFIG["fusion"]["plus_pe_weight"] * plus["prob_positive"] + CONFIG["fusion"]["graph_weight"] * graph["prob_positive"]
    frame["subject_window_index"] = frame.groupby("subject_id", sort=False).cumcount()
    frame["physical_trial_id"] = frame["subject_window_index"] // 5 + 1
    frame["window_index"] = frame["subject_window_index"] % 5
    if frame.groupby("subject_id").size().nunique() != 1 or frame.groupby("subject_id").size().iloc[0] != 40:
        raise RuntimeError("Expected 40 windows per training subject")
    if frame.groupby(["subject_id", "physical_trial_id"]).size().nunique() != 1 or frame.groupby(["subject_id", "physical_trial_id"]).size().iloc[0] != 5:
        raise RuntimeError("Expected 5 windows per physical trial")
    return frame


def adapt(support, query):
    if len(support) == 0 or support["y_true"].nunique() < 2:
        return query["base_probability"].to_numpy(), "frozen_base_single_class_or_k0"
    x = np.clip(support["base_probability"].to_numpy(), 1e-6, 1 - 1e-6)
    z = np.log(x / (1 - x)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=CONFIG["random_seed"])
    model.fit(z, support["y_true"].to_numpy())
    q = np.clip(query["base_probability"].to_numpy(), 1e-6, 1 - 1e-6)
    return model.predict_proba(np.log(q / (1 - q)).reshape(-1, 1))[:, 1], "logistic_support_calibration"


def trial_metrics(pred):
    trial = pred.groupby(["subject_id", "health_group", "support_trials", "physical_trial_id"], as_index=False).agg(y_true=("y_true", "first"), probability=("prob_positive", "mean"))
    return score(trial.y_true.to_numpy(), trial.probability.to_numpy()), trial


def main():
    start = time.perf_counter()
    sys.stdout = Tee(HERE / "run_output.txt")
    frame = load_inputs()
    all_metrics = {"config": CONFIG, "provenance": {"python": platform.python_version(), "base_rows": len(frame), "fusion_fixed": CONFIG["fusion"], "query_labels_used_for_selection": False, "public_labels_used": False}, "runs": {}}
    for k in CONFIG["support_k"]:
        records, combo_rows = [], []
        for si, (subject, group) in enumerate(frame.groupby("subject_id", sort=True), 1):
            sub = group.sort_values(["original_trial_id", "window_index"]).copy()
            trials = sorted(sub.physical_trial_id.unique())
            combos = [()] if k == 0 else list(itertools.combinations(trials, k))
            for ci, support_ids in enumerate(combos, 1):
                support_text = ",".join(map(str, support_ids)) if support_ids else "none"
                support = sub[sub.physical_trial_id.isin(support_ids)]
                query = sub[~sub.physical_trial_id.isin(support_ids)].copy()
                query["prob_positive"], method = adapt(support, query)
                query["support_trials"] = support_text
                query["adaptation_method"] = method
                query["support_k"] = k
                records.append(query)
                combo_rows.append({"subject_id": subject, "health_group": group, "support_k": k, "support_trials": support_text, "support_n_windows": len(support), "query_n_windows": len(query), "method": method})
            print(f"k={k} subject {si}/{frame.subject_id.nunique()} combinations={len(combos)}", flush=True)
        pred = pd.concat(records, ignore_index=True)
        out_cols = ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "base_probability", "prob_positive", "support_k", "support_trials", "adaptation_method"]
        pred[out_cols].to_csv(HERE / "predictions" / f"k{k}_query_oof.csv", index=False)
        rows = []
        for combo, part in pred.groupby("support_trials", sort=True):
            r = score(part.y_true.to_numpy(), part.prob_positive.to_numpy())
            rows.append({"support_trials": combo, **r})
        trial_score, trial = trial_metrics(pred)
        pooled = score(pred.y_true.to_numpy(), pred.prob_positive.to_numpy())
        subject_scores = []
        for subject, part in pred.groupby("subject_id"):
            subject_scores.append(score(part.y_true.to_numpy(), part.prob_positive.to_numpy())["auc"])
        hc = pred[pred.health_group == "HC"]
        dep = pred[pred.health_group == "DEP"]
        all_metrics["runs"][f"k{k}"] = {"window_pooled": pooled, "trial_pooled": trial_score, "subject_auc_mean": float(np.nanmean(subject_scores)), "hc_window": score(hc.y_true.to_numpy(), hc.prob_positive.to_numpy()), "dep_window": score(dep.y_true.to_numpy(), dep.prob_positive.to_numpy()), "n_support_combinations": int(len(combo_rows)), "combination_metrics": rows, "adaptation_methods": pred.adaptation_method.value_counts().to_dict()}
        pd.DataFrame(combo_rows).to_csv(HERE / "input_metadata" / f"k{k}_support_combinations.csv", index=False)
        print(f"k={k} pooled AUC={pooled['auc']:.6f} F1={pooled['f1']:.6f} trial AUC={trial_score['auc']:.6f} F1={trial_score['f1']:.6f}", flush=True)
    metrics = all_metrics
    metrics["elapsed_seconds"] = time.perf_counter() - start
    (HERE / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = []
    for name, result in metrics["runs"].items():
        summary.append({"condition": name, "window_auc": result["window_pooled"]["auc"], "window_f1": result["window_pooled"]["f1"], "trial_auc": result["trial_pooled"]["auc"], "trial_f1": result["trial_pooled"]["f1"], "subject_auc_mean": result["subject_auc_mean"], "hc_auc": result["hc_window"]["auc"], "dep_auc": result["dep_window"]["auc"], "support_combinations": result["n_support_combinations"]})
    pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    lines = ["# 实验198：严格物理trial支持/查询适配", "", "基线为实验057 plus_pe 与实验006 frequency_graph_zscore 的固定 0.85/0.15 概率融合。每个被试8个物理trial、每trial 5个窗口；support 与 query 完全按物理trial隔离。k=0不使用目标标签；k>0仅为训练集物理trial上的仿真适配，不代表公开集可用协议。", "", "外层查询标签仅用于最终评分，未参与模型、校准规则或融合选择；所有支持集组合均按预先固定规则计算。", "", "|条件|窗口AUC|窗口F1|trial AUC|trial F1|被试AUC均值|HC AUC|DEP AUC|支持组合数|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"|{row['condition']}|{row['window_auc']:.6f}|{row['window_f1']:.6f}|{row['trial_auc']:.6f}|{row['trial_f1']:.6f}|{row['subject_auc_mean']:.6f}|{row['hc_auc']:.6f}|{row['dep_auc']:.6f}|{row['support_combinations']}|")
    lines += ["", f"运行耗时：{metrics['elapsed_seconds']:.2f} 秒。详细逐组合指标见 metrics.json；OOF见 predictions/k*_query_oof.csv；支持集审计见 input_metadata/k*_support_combinations.csv。"]
    (HERE / "实验报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class Tee:
    def __init__(self, path):
        self.file = path.open("w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()


if __name__ == "__main__":
    main()
