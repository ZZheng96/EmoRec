import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "031_nested_classifier_calibration"
spec = importlib.util.spec_from_file_location("experiment031", BASE / "run_experiment.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
NAMES = ["raw_probability", "platt_probability", "raw_logit", "logit_zscore"]


def logit(p):
    eps = np.finfo(float).eps
    return np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))


def fuse(inner, labels, target):
    inner_logits = np.column_stack([logit(p) for p in inner])
    target_logits = np.column_stack([logit(p) for p in target])
    means, scales = inner_logits.mean(axis=0), inner_logits.std(axis=0, ddof=0)
    scales = np.where(scales == 0, 1.0, scales)
    calibrators = [base.fit_platt(p, labels) for p in inner]
    predictions = {
        "raw_probability": np.mean(target, axis=0),
        "platt_probability": np.mean([base.apply_platt(c, p) for c, p in zip(calibrators, target)], axis=0),
        "raw_logit": expit(target_logits.mean(axis=1)),
        "logit_zscore": expit(((target_logits - means) / scales).mean(axis=1)),
    }
    stats = {"logit_mean": means.tolist(), "logit_std_ddof0": scales.tolist(),
             "branch_order": ["spectral", "temporal"], "n_inner_oof": len(labels),
             "platt_coef": [c.coef_.ravel().tolist() for c in calibrators],
             "platt_intercept": [c.intercept_.tolist() for c in calibrators]}
    return predictions, stats


def summarize(ref, p):
    y = ref.y_true.to_numpy()
    subjects = []
    for subject, frame in ref.groupby("subject_id", sort=True):
        subjects.append({"subject_id": subject, "health_group": frame.health_group.iloc[0],
                         **base.metrics(frame.y_true.to_numpy(), p[frame.index])})
    def macro(rows):
        return {"auc": float(np.mean([r["auc"] for r in rows])),
                "f1": float(np.mean([r["f1"] for r in rows]))}
    groups = {}
    for group in ("HC", "DEP"):
        mask = ref.health_group.to_numpy() == group
        rows = [r for r in subjects if r["health_group"] == group]
        groups[group] = {"pooled": base.metrics(y[mask], p[mask]), "subject_macro": macro(rows), "n_subjects": len(rows)}
    return {"overall": base.metrics(y, p), "health_groups": groups,
            "subject_macro": macro(subjects), "subjects": subjects}


def main():
    config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    base.CONFIG = config
    sources = [BASE / "config.json", BASE / "run_experiment.py", BASE / "metrics.json"]
    def read_prediction(name):
        path = BASE / "predictions" / name
        sources.append(path)
        return pd.read_csv(path)
    spectral = read_prediction("spectral_selected_oof.csv")
    temporal = read_prediction("temporal_selected_oof.csv")
    keys = base.OOF_COLUMNS[:-1]
    pd.testing.assert_frame_equal(spectral[keys], temporal[keys])
    ref = spectral[keys].copy()
    if ref.groupby("subject_id").fold.nunique().max() != 1:
        raise RuntimeError("outer subject isolation violated")
    inner_cache = read_prediction("inner_oof.csv")
    selected_leaves = read_prediction("leaf_selection.csv")
    predictions = {name: np.full(len(ref), np.nan) for name in NAMES}
    statistics = []
    for fold in range(config["outer_folds"]):
        train, valid = ref.fold.to_numpy() != fold, ref.fold.to_numpy() == fold
        train_ref = ref.loc[train].reset_index(drop=True)
        inner = []
        assignments = []
        for branch in ("spectral", "temporal"):
            rows = inner_cache.loc[(inner_cache.outer_fold == fold) & (inner_cache.branch == branch) & inner_cache.is_selected].reset_index(drop=True)
            for key in ("subject_id", "health_group", "y_true"):
                if not np.array_equal(rows[key].to_numpy(), train_ref[key].to_numpy()):
                    raise RuntimeError(f"inner cache alignment mismatch: {fold}/{branch}/{key}")
            if rows.groupby("subject_id").inner_fold.nunique().max() != 1:
                raise RuntimeError("inner subject isolation violated")
            leaf = int(selected_leaves.loc[selected_leaves.outer_fold == fold, f"{branch}_selected_leaf"].iloc[0])
            if not (rows.leaf == leaf).all():
                raise RuntimeError("selected leaf mismatch")
            inner.append(rows.probability.to_numpy())
            assignments.append(rows.inner_fold.to_numpy())
        if not np.array_equal(*assignments):
            raise RuntimeError("branch inner split mismatch")
        fused, stats = fuse(inner, train_ref.y_true.to_numpy(), [spectral.prob_positive.to_numpy()[valid], temporal.prob_positive.to_numpy()[valid]])
        statistics.append({"outer_fold": fold, **stats})
        for name, p in fused.items():
            predictions[name][valid] = p
        print(f"Outer fold {fold + 1}/5: reused 031 nested OOF; fitted inner-only transforms", flush=True)
    # Check exact baseline reproduction before producing new deliverables.
    for name, original in (("raw_probability", "raw_fusion"), ("platt_probability", "platt_fusion")):
        cached = read_prediction(f"{original}_oof.csv")
        pd.testing.assert_frame_equal(ref, cached[keys])
        if not np.allclose(predictions[name], cached.prob_positive, atol=1e-10, rtol=0):
            raise RuntimeError(f"031 baseline reproduction failed: {name}")
    public_frames = [read_prediction(f"{branch}_selected_public.csv") for branch in ("spectral", "temporal")]
    pd.testing.assert_frame_equal(public_frames[0].iloc[:, :2], public_frames[1].iloc[:, :2])
    full_inner, public_inner_rows, full_leaves = [], [], {}
    with threadpool_limits(limits=config["threads"]):
        for branch, cache_key in (("spectral", "spectral_cache"), ("temporal", "temporal_cache")):
            path = (BASE / config[cache_key]).resolve()
            sources.append(path)
            cache = base.load_cache(path)
            keep = cache["subject_ids"] != config["excluded_subject"]
            for cache_key2, ref_key in (("subject_ids", "subject_id"), ("health_groups", "health_group"), ("original_trial_ids", "original_trial_id"), ("y_train", "y_true")):
                if not np.array_equal(cache[cache_key2][keep], ref[ref_key].to_numpy()):
                    raise RuntimeError(f"feature cache alignment mismatch: {branch}/{ref_key}")
            if not np.array_equal(cache["public_user_ids"], public_frames[0].user_id) or not np.array_equal(cache["public_trial_ids"], public_frames[0].trial_id):
                raise RuntimeError("public cache alignment mismatch")
            x = cache["X_train"][keep]
            if branch == "spectral":
                x = x[:, :150]
            x = base.normalize_by_subject(x, cache["subject_ids"][keep], "zscore")
            candidates = []
            for leaf in config["hgb_candidates"]:
                print(f"Public full-training inner OOF: {branch}, leaf={leaf}, 3 subject folds starting", flush=True)
                p, assignment = base.inner_oof(x, ref.y_true.to_numpy(), ref.subject_id.to_numpy(), leaf, 2026, config["inner_folds"])
                candidates.append({"leaf": leaf, "prob": p, "assignment": assignment, **base.metrics(ref.y_true, p)})
                print(f"Public full-training inner OOF: {branch}, leaf={leaf} complete", flush=True)
            chosen = max(candidates, key=lambda r: (r["auc"], r["f1"], -abs(r["leaf"] - config["control_leaf"])))
            full_inner.append(chosen["prob"])
            full_leaves[branch] = chosen["leaf"]
            frame = ref.drop(columns="fold").copy()
            frame["inner_fold"] = chosen["assignment"]
            frame["branch"] = branch
            frame["leaf"] = chosen["leaf"]
            frame["probability"] = chosen["prob"]
            public_inner_rows.append(frame)
    public, public_stats = fuse(full_inner, ref.y_true.to_numpy(), [f.iloc[:, 2].to_numpy() for f in public_frames])
    cached_public = read_prediction("platt_fusion_public.csv")
    if not np.allclose(public["platt_probability"], cached_public.iloc[:, 2], atol=1e-10, rtol=0):
        raise RuntimeError("public full-training inner OOF does not reproduce 031 Platt predictions")
    runs, summary = {}, []
    for name, p in predictions.items():
        if not np.isfinite(p).all():
            raise RuntimeError("incomplete OOF predictions")
        run = summarize(ref, p)
        run["folds"] = []
        for fold in range(config["outer_folds"]):
            mask = ref.fold.to_numpy() == fold
            run["folds"].append({"fold": fold, **summarize(ref.loc[mask].reset_index(drop=True), p[mask])})
        runs[name] = run
        summary.append({"run_name": name, "auc": run["overall"]["auc"], "f1": run["overall"]["f1"],
                        "subject_macro_auc": run["subject_macro"]["auc"], "subject_macro_f1": run["subject_macro"]["f1"],
                        **{f"{group.lower()}_{metric}": run["health_groups"][group]["pooled"][metric] for group in ("HC", "DEP") for metric in ("auc", "f1")}})
    baseline = json.loads((BASE / "metrics.json").read_text(encoding="utf-8"))["runs"]["platt_fusion"]["overall"]
    comparisons = {name: {"auc_delta_vs_031_platt": run["overall"]["auc"] - baseline["auc"], "f1_delta_vs_031_platt": run["overall"]["f1"] - baseline["f1"], "both_improved": run["overall"]["auc"] > baseline["auc"] and run["overall"]["f1"] > baseline["f1"]} for name, run in runs.items()}
    result = {"experiment_id": "203_logit_fusion", "base_config": config,
              "protocol": {"outer_predictions_reused_from_031": True, "inner_oof_reused_from_031": True, "public_inner_oof_recomputed_exact_031": True, "fusion_weight": 0.5, "threshold": 0.5, "logit_clip_epsilon": float(np.finfo(float).eps), "zscore_source": "selected branch inner OOF logits only, pooled over outer-training samples; public uses full-training inner OOF", "outer_labels_scoring_only": True, "public_labels_used": False, "method_selection": "none; four prespecified methods reported"},
              "provenance": {"python": platform.python_version(), "source_sha256": {str(path): base.sha256(path) for path in sources}},
              "outer_transform_statistics": statistics, "public_transform_statistics": public_stats, "public_selected_leaves": full_leaves,
              "runs": runs, "baseline_031_platt": baseline, "comparison": comparisons}
    table = pd.DataFrame(summary)
    report = "# 实验203：概率与logit融合比较\n\n严格复用031的spectral 150维、temporal 450维特征缓存、69名被试（排除HC43）、2760样本、固定5折外层被试划分及3折inner OOF。leaf候选7/15/31按inner AUC、F1、接近15排序。外层直接复用031已选分支预测与inner OOF；公开阶段复算全训练inner OOF并复用031公开分支预测，Platt结果一致性检查通过。旧实验不修改。\n\n四种融合均固定等权、F1阈值0.5，不从外层标签选择方法、权重、阈值或统计量。raw probability为概率均值；Platt probability先以分支inner OOF拟合031的Platt再均值；raw logit为sigmoid(两个logit均值)；logit-zscore为sigmoid(两个标准化logit均值)。zscore每个分支的均值和总体标准差(ddof=0)严格仅来自当前外层训练集的inner OOF，公开阶段仅来自全训练inner OOF；绝不使用外层或公开预测分布拟合。逐折参数可在metrics.json审计。\n\n## OOF结果\n\n|方法|AUC|F1|HC AUC|HC F1|DEP AUC|DEP F1|subject宏AUC|subject宏F1|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in summary:
        report += f"|{row['run_name']}|" + "|".join(f"{row[key]:.6f}" for key in ("auc", "f1", "hc_auc", "hc_f1", "dep_auc", "dep_f1", "subject_macro_auc", "subject_macro_f1")) + "|\n"
    report += f"\n031 Platt基线：AUC={baseline['auc']:.6f}，F1={baseline['f1']:.6f}。\n\n"
    for name, comparison in comparisons.items():
        report += f"- {name}：ΔAUC={comparison['auc_delta_vs_031_platt']:+.6f}，ΔF1={comparison['f1_delta_vs_031_platt']:+.6f}；两者均超过基线：{'是' if comparison['both_improved'] else '否'}。\n"
    report += "\n## 每折结果\n\n|方法|折|AUC|F1|HC AUC|HC F1|DEP AUC|DEP F1|\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for name, run in runs.items():
        for fold in run["folds"]:
            values = [fold["overall"][m] for m in ("auc", "f1")] + [fold["health_groups"][g]["pooled"][m] for g in ("HC", "DEP") for m in ("auc", "f1")]
            report += f"|{name}|{fold['fold']}|" + "|".join(f"{v:.6f}" for v in values) + "|\n"
    report += "\nsubject macro是逐被试指标等权平均；HC/DEP为组内pooled指标，组内宏指标及每折/每被试明细见metrics.json。raw probability与raw logit对两个等权分支在0.5阈值的判别理论一致，AUC可因排序变化而不同。公开标签不可用，因此仅提供4份公开概率预测，不报告公开AUC/F1，也不以外层最佳方法作为正式选择。固定外层折反复比较仍有探索性选择偏差，不能声称无偏泛化提升。沿用031的CPU sklearn HGB实现以保持协议一致，限制单线程，不新增GPU模型。\n"
    out = HERE / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        frame = ref.copy()
        frame["prob_positive"] = predictions[name]
        frame.to_csv(out / f"{name}_oof.csv", index=False)
        base.public_frame(public_frames[0].user_id.to_numpy(), public_frames[0].trial_id.to_numpy(), public[name]).to_csv(out / f"{name}_public.csv", index=False)
    pd.concat(public_inner_rows, ignore_index=True).to_csv(out / "public_training_inner_oof.csv", index=False)
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    table.to_csv(HERE / "summary.csv", index=False)
    (HERE / "实验报告.md").write_text(report, encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(json.dumps(comparisons, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
