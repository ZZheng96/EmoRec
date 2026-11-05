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
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def metric(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return {"auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "f1": float(f1_score(y, p >= CONFIG["threshold"], zero_division=0)), "n": int(len(y)),
            "positive": int(y.sum())}


def load_inputs():
    plus = pd.read_csv(HERE / CONFIG["base_plus_pe_oof"])
    graph = pd.read_csv(HERE / CONFIG["base_graph_oof"])
    keys = ["subject_id", "health_group", "original_trial_id", "fold", "y_true"]
    if not plus[keys].equals(graph[keys]):
        raise RuntimeError("057/006 OOF alignment mismatch")
    frame = plus[keys].copy()
    frame["base_probability"] = CONFIG["fusion"]["plus_pe_weight"] * plus["prob_positive"] + CONFIG["fusion"]["graph_weight"] * graph["prob_positive"]
    if frame.groupby("subject_id").size().nunique() != 1 or frame.groupby("subject_id").size().iloc[0] != 40:
        raise RuntimeError("Expected 40 windows per subject")
    frame["subject_window_index"] = frame.groupby("subject_id", sort=False).cumcount()
    frame["physical_trial_id"] = frame["subject_window_index"] // 5 + 1
    frame["window_index"] = frame["subject_window_index"] % 5
    if frame.groupby(["subject_id", "physical_trial_id"]).size().nunique() != 1 or frame.groupby(["subject_id", "physical_trial_id"]).size().iloc[0] != 5:
        raise RuntimeError("Expected 5 windows per physical trial")
    selection = pd.read_csv(HERE / CONFIG["selection_per_subject"])
    required = {"subject", "group", "linear_svc_raw_auc"}
    if not required.issubset(selection.columns):
        raise RuntimeError(f"Missing selection columns: {required - set(selection.columns)}")
    frozen = selection.loc[selection.linear_svc_raw_auc > 0.90, "subject"].astype(str).sort_values().tolist()
    if not frozen:
        raise RuntimeError("Frozen subgroup is empty")
    stable = CONFIG["stable_hc"]
    if not set(stable).issubset(set(frame.subject_id)):
        raise RuntimeError("Stable HC subject missing from OOF")
    return frame, selection, frozen


def adapt(support, query):
    if len(support) == 0 or support.y_true.nunique() < 2:
        return query.base_probability.to_numpy(), "frozen_base_single_class_or_k0"
    z = np.log(np.clip(support.base_probability.to_numpy(), 1e-6, 1 - 1e-6) / np.clip(1 - support.base_probability.to_numpy(), 1e-6, 1))[:, None]
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=CONFIG["random_seed"])
    model.fit(z, support.y_true.to_numpy())
    q = np.clip(query.base_probability.to_numpy(), 1e-6, 1 - 1e-6)
    return model.predict_proba((np.log(q / (1 - q)))[:, None])[:, 1], "logistic_support_calibration"


def evaluate(part):
    out = {}
    for scope, sub in [("window_pooled", part), ("trial_pooled", part.groupby(["subject_id", "health_group", "physical_trial_id"], as_index=False).agg(y_true=("y_true", "first"), prob_positive=("prob_positive", "mean")))]:
        out[scope] = metric(sub.y_true, sub.prob_positive)
    subjects = [metric(s.y_true, s.prob_positive) for _, s in part.groupby("subject_id")]
    out["subject_macro"] = {k: float(np.nanmean([x[k] for x in subjects])) for k in ("auc", "f1")}
    for name, sub in [("all", part), ("subgroup", part[part.subgroup]), ("stable_hc", part[part.subject_id.isin(CONFIG["stable_hc"])]), ("non_subgroup", part[~part.subgroup]), ("HC", part[part.health_group == "HC"]), ("DEP", part[part.health_group == "DEP"])]:
        if len(sub):
            trial = sub.groupby(["subject_id", "physical_trial_id"], as_index=False).agg(y_true=("y_true", "first"), prob_positive=("prob_positive", "mean"))
            subject_scores = [metric(s.y_true, s.prob_positive) for _, s in sub.groupby("subject_id")]
            out[name] = {"window": metric(sub.y_true, sub.prob_positive), "trial": metric(trial.y_true, trial.prob_positive), "subject_macro": {k: float(np.nanmean([x[k] for x in subject_scores])) for k in ("auc", "f1")}}
        else:
            out[name] = None
    return out


def main():
    start = time.perf_counter(); print("[204] loading and validating inputs", flush=True)
    frame, selection, frozen = load_inputs(); stable = CONFIG["stable_hc"]
    frame["subgroup"] = frame.subject_id.isin(frozen)
    frame["base_probability"] = np.clip(frame.base_probability, 0, 1)
    (HERE / "input_metadata" / "frozen_high_auc_subjects.csv").write_text("subject,health_group,linear_svc_raw_auc\n" + "\n".join(f"{r.subject},{r.group},{r.linear_svc_raw_auc}" for _, r in selection[selection.subject.isin(frozen)].sort_values("subject").iterrows()) + "\n", encoding="utf-8")
    (HERE / "input_metadata" / "stable_hc_subjects.txt").write_text("\n".join(stable) + "\n", encoding="utf-8")
    metadata = {"frozen_subjects": frozen, "stable_hc": stable, "n_subjects": int(frame.subject_id.nunique()), "n_rows": int(len(frame)), "rows_per_subject": frame.groupby("subject_id").size().to_dict(), "selection_rule": CONFIG["selection_rule"], "support_query_physical_trial_intersection": {}}
    all_metrics = {"config": CONFIG, "provenance": {"python": platform.python_version(), "public_labels_used": False, "query_labels_for_selection": False}, "frozen_subjects": frozen, "runs": {}}
    for k in CONFIG["support_k"]:
        records = []; combos = []
        for si, (subject, sub) in enumerate(frame.groupby("subject_id", sort=True), 1):
            sub = sub.sort_values(["physical_trial_id", "window_index"]).copy(); trials = sorted(sub.physical_trial_id.unique()); support_sets = [()] if k == 0 else list(itertools.combinations(trials, k))
            for support_ids in support_sets:
                support = sub[sub.physical_trial_id.isin(support_ids)]; query = sub[~sub.physical_trial_id.isin(support_ids)].copy(); query["prob_positive"], method = adapt(support, query); query["support_k"] = k; query["support_trials"] = ",".join(map(str, support_ids)) if support_ids else "none"; query["adaptation_method"] = method; records.append(query)
                combos.append({"subject_id": subject, "support_k": k, "support_trials": query.support_trials.iloc[0], "support_n_windows": len(support), "query_n_windows": len(query), "physical_trial_intersection": sorted(set(support.physical_trial_id) & set(query.physical_trial_id)), "method": method})
            print(f"[204] k={k} subject {si}/{frame.subject_id.nunique()} combinations={len(support_sets)}", flush=True)
        pred = pd.concat(records, ignore_index=True); pred.to_csv(HERE / "predictions" / f"k{k}_query_oof.csv", index=False); pd.DataFrame(combos).to_csv(HERE / "input_metadata" / f"k{k}_support_combinations.csv", index=False)
        metadata["support_query_physical_trial_intersection"][f"k{k}"] = bool(any(c["physical_trial_intersection"] for c in combos))
        all_metrics["runs"][f"k{k}"] = evaluate(pred); all_metrics["runs"][f"k{k}"]["n_combinations"] = len(combos)
        print(f"[204] completed k={k}", flush=True)
    metadata["support_query_physical_trial_intersection_all_clear"] = not any(metadata["support_query_physical_trial_intersection"].values())
    (HERE / "input_metadata" / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    all_metrics["elapsed_seconds"] = time.perf_counter() - start; (HERE / "metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for k, result in all_metrics["runs"].items():
        for scope in ("all", "subgroup", "stable_hc", "non_subgroup", "HC", "DEP"):
            if result[scope]: rows.append({"condition": k, "scope": scope, "window_auc": result[scope]["window"]["auc"], "window_f1": result[scope]["window"]["f1"], "trial_auc": result[scope]["trial"]["auc"], "trial_f1": result[scope]["trial"]["f1"]})
        rows.append({"condition": k, "scope": "subject_macro", "window_auc": result["subject_macro"]["auc"], "window_f1": result["subject_macro"]["f1"], "trial_auc": None, "trial_f1": None})
    pd.DataFrame(rows).to_csv(HERE / "summary.csv", index=False)
    lines = ["# 实验204：高AUC冻结子群上限评估", "", f"按实验095 per_subject.csv 实际计算 linear_svc_raw_auc > 0.90，冻结被试 {len(frozen)} 人：{', '.join(frozen)}。稳定HC五人组：{', '.join(stable)}。", "", "基线为057 fixed002 plus_pe 与006 graph OOF固定0.85/0.15概率融合；k=0不使用支持标签，k=1/2/4严格按完整物理trial support/query隔离。k>0是有标签上限模拟，不能公开提交。", "", f"support/query物理trial交集审计：{'通过，无交集' if metadata['support_query_physical_trial_intersection_all_clear'] else '失败'}。", "", "详见 summary.csv（全体、冻结子群、稳定五人组、非子群、HC/DEP及subject macro），metrics.json 和 predictions。"]
    (HERE / "实验报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
