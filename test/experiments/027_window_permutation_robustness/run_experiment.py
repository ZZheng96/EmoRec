import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("experiment026", HERE.parent / "026_window_matched_hgb/run_experiment.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
normalize = protocol.normalize
align_reference = protocol.align_reference
metrics = protocol.metrics


def random_batches(subjects, seed):
    batches = np.empty(len(subjects), dtype=np.int64)
    for subject in np.unique(subjects):
        indices = np.flatnonzero(subjects == subject)
        if len(indices) != 40:
            raise ValueError(f"Expected 40 rows for {subject}")
        digest = hashlib.sha256(str(subject).encode("utf-8")).digest()
        entropy = [seed] + [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 32, 4)]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        batches[indices[rng.permutation(40)]] = np.repeat(np.arange(5), 8)
    return batches


def describe(y, p, subjects, health, folds):
    rows = [{"subject_id": s, "health_group": health[subjects == s][0],
             **metrics(y[subjects == s], p[subjects == s])} for s in np.unique(subjects)]
    result = {"overall": metrics(y, p), "subjects": rows,
              "subject_auc_mean": float(np.mean([r["auc"] for r in rows])),
              "subject_f1_mean": float(np.mean([r["f1"] for r in rows])),
              "folds": [{"fold": f, **metrics(y[folds == f], p[folds == f])} for f in range(5)],
              "health_groups": {}}
    for group in ("HC", "DEP"):
        subset = [r for r in rows if r["health_group"] == group]
        mask = health == group
        result["health_groups"][group] = {"n_subjects": len(subset), "pooled": metrics(y[mask], p[mask]),
            "subject_auc_mean": float(np.mean([r["auc"] for r in subset])),
            "subject_f1_mean": float(np.mean([r["f1"] for r in subset]))}
    return result


def main():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    config026 = json.loads((HERE.parent / "026_window_matched_hgb/config.json").read_text(encoding="utf-8"))
    for key in ("feature_cache", "reference_oof", "excluded_subject", "hgb", "outer_folds", "near_zero_std", "feature_prefix", "random_seed"):
        assert config[key] == config026[key], key
    cache_path, reference_path = [(HERE / config[k]).resolve() for k in ("feature_cache", "reference_oof")]
    with np.load(cache_path, allow_pickle=False) as cache:
        data = {k: cache[k] for k in cache.files}
    keep = data["subject_ids"] != config["excluded_subject"]
    cache_rows = np.flatnonzero(keep)
    for key in ("X_train", "y_train", "subject_ids", "health_groups", "original_trial_ids"):
        data[key] = data[key][keep]
    reference = pd.read_csv(reference_path)
    folds = align_reference(data, reference)
    subjects, y = data["subject_ids"], data["y_train"]
    assert len(y) == config["n_samples"] and len(np.unique(subjects)) == config["expected_subjects"]
    features = np.array([str(n).startswith(config["feature_prefix"]) for n in data["feature_names"]])
    X = data["X_train"][:, features]
    assert X.shape[1] == config["n_features"] and np.isfinite(X).all()
    modulo = protocol.segment_batches(subjects)
    X40 = normalize(X, subjects, modulo, 40, config["near_zero_std"])
    public_subjects = data["public_user_ids"]
    assert len(public_subjects) == 80 and all(np.sum(public_subjects == s) == 8 for s in np.unique(public_subjects))
    public_X = normalize(data["X_public"][:, features], public_subjects, None, 40, config["near_zero_std"])
    output = HERE / "predictions"
    output.mkdir(exist_ok=True)
    results = {"config": config, "provenance": {"cache_sha256": protocol.sha256(cache_path),
        "reference_sha256": protocol.sha256(reference_path), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "threads": 1}, "runs": {}}
    summaries = []
    control_models = []

    def model(fold):
        return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=config["random_seed"] + fold))

    def run(name, condition, seed, batches, train_X, eval_X):
        p = np.full(len(y), np.nan)
        for fold in range(5):
            train, valid = folds != fold, folds == fold
            if condition == "eval_only_40_8":
                estimator = control_models[fold]
            else:
                estimator = model(fold)
                estimator.fit(train_X[train], y[train])
                if condition == "control_40_40":
                    control_models.append(estimator)
            p[valid] = estimator.predict_proba(eval_X[valid])[:, 1]
        assert np.isfinite(p).all()
        oof = reference.copy()
        oof["prob_positive"] = p
        oof["cache_row"] = cache_rows
        oof["normalization_batch"] = batches
        oof.to_csv(output / f"{name}_oof.csv", index=False)
        detail = describe(y, p, subjects, data["health_groups"], folds)
        row = {"run_name": name, "condition": condition, "seed": seed, **detail["overall"],
               "subject_auc_mean": detail["subject_auc_mean"], "subject_f1_mean": detail["subject_f1_mean"]}
        for group in ("HC", "DEP"):
            h = detail["health_groups"][group]
            row.update({f"{group.lower()}_pooled_auc": h["pooled"]["auc"], f"{group.lower()}_pooled_f1": h["pooled"]["f1"],
                        f"{group.lower()}_subject_auc_mean": h["subject_auc_mean"], f"{group.lower()}_subject_f1_mean": h["subject_f1_mean"]})
        if condition in ("control_40_40", "baseline_8_8"):
            old_name = "matched_8_8" if condition == "baseline_8_8" else condition
            old = pd.read_csv(HERE.parent / f"026_window_matched_hgb/predictions/{old_name}_oof.csv")
            diff = float(np.max(np.abs(old.prob_positive.to_numpy() - p)))
            assert diff < 1e-8, diff
            detail["reference_026_max_abs_probability_difference"] = diff
        if condition != "control_40_40":
            baseline = detail if condition == "baseline_8_8" else results["runs"]["baseline_8_8"]
            base_rows = {r["subject_id"]: r for r in baseline["subjects"]}
            for r in detail["subjects"]:
                for metric in ("auc", "f1"):
                    r[f"delta_{metric}_vs_baseline"] = r[metric] - base_rows[r["subject_id"]][metric]
            for metric in ("auc", "f1"):
                row[f"delta_{metric}_vs_baseline"] = detail["overall"][metric] - baseline["overall"][metric]
                row[f"delta_subject_{metric}_mean_vs_baseline"] = detail[f"subject_{metric}_mean"] - baseline[f"subject_{metric}_mean"]
                deltas = np.array([r[f"delta_{metric}_vs_baseline"] for r in detail["subjects"]])
                row[f"subjects_{metric}_improved"] = int(np.sum(deltas > 1e-12))
                row[f"subjects_{metric}_declined"] = int(np.sum(deltas < -1e-12))
                row[f"subjects_{metric}_tied"] = int(np.sum(np.abs(deltas) <= 1e-12))
            detail["comparison_vs_baseline"] = {k: v for k, v in row.items() if k.startswith(("delta_", "subjects_"))}
        if condition == "eval_only_40_8":
            final = control_models[5]
        else:
            final = model(0)
            final.fit(train_X, y)
            if condition == "control_40_40":
                control_models.append(final)
        public = pd.DataFrame({"user_id": public_subjects, "trial_id": data["public_trial_ids"],
                               "prob. of positive emotion": final.predict_proba(public_X)[:, 1]})
        validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
        assert len(public) == 80
        public.to_csv(output / f"{name}_public.csv", index=False)
        results["runs"][name] = detail
        summaries.append(row)
        print(f"{name}: AUC={row['auc']:.6f}, macro={row['subject_auc_mean']:.6f}, F1={row['f1']:.6f}", flush=True)

    with threadpool_limits(limits=1):
        run("control_40_40", "control_40_40", None, np.full(len(y), -1), X40, X40)
        X8 = normalize(X, subjects, modulo, 8, config["near_zero_std"])
        run("baseline_8_8", "baseline_8_8", None, modulo, X8, X8)
        for seed in config["permutation_seeds"]:
            batches = random_batches(subjects, seed)
            X8 = normalize(X, subjects, batches, 8, config["near_zero_std"])
            run(f"matched_random_8_8_seed_{seed}", "matched_random_8_8", seed, batches, X8, X8)
            run(f"eval_only_40_8_seed_{seed}", "eval_only_40_8", seed, batches, X40, X8)
    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE / "summary.csv", index=False)
    results["seed_summary"] = {}
    columns = [c for c in summary.select_dtypes(include="number").columns if c not in ("seed", "threshold")]
    for condition in ("matched_random_8_8", "eval_only_40_8"):
        subset = summary[summary.condition == condition]
        results["seed_summary"][condition] = {c: {"mean": float(subset[c].mean()), "std": float(subset[c].std(ddof=1))} for c in columns}
    (HERE / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report = "# 实验027：8窗口归一化的缓存行序依赖检验\n\n## 预固定协议\n"
    report += "严格复用026缓存、002参考OOF及五折、排除HC43（69被试2760行）、150维log_bandpower、HGB参数及fold seed=2026+fold，单线程。40/40控制和原缓存序号%5的8/8 baseline逐行复现026，最大概率差均小于1e-8。\n\n"
    report += f"预固定十种子：{config['permutation_seeds']}。每个被试以seed和subject_id的SHA256派生独立随机流，置乱40行后连续分成5组各8行；不接收标签或original_trial_id，不按类别/试次配平，不根据结果筛选种子。分组只在被试内；训练/验证被试严格隔离。matched训练验证都用本被试随机8行统计，eval-only训练40行、验证同一随机8行统计。归一化为ddof=0，阈值0.5。eval-only复用完全相同的40行训练模型，等价于逐seed重训。\n\n"
    report += "## 逐条件/种子结果\n|条件|seed|总体AUC|宏AUC|F1|宏F1|HC AUC|DEP AUC|ΔAUC|Δ宏AUC|AUC改善/下降/持平|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    for r in summaries:
        counts = '/'.join(str(r.get(f'subjects_auc_{k}', '-')) for k in ('improved', 'declined', 'tied'))
        report += f"|{r['condition']}|{r['seed']}|{r['auc']:.6f}|{r['subject_auc_mean']:.6f}|{r['f1']:.6f}|{r['subject_f1_mean']:.6f}|{r['hc_pooled_auc']:.6f}|{r['dep_pooled_auc']:.6f}|{r.get('delta_auc_vs_baseline', 0):+.6f}|{r.get('delta_subject_auc_mean_vs_baseline', 0):+.6f}|{counts}|\n"
    report += "\nΔ以原取模8/8为参照；40/40行的Δ占位0不表示相同。HC/DEP完整pooled与宏AUC/F1、每被试AUC/F1和变化、F1改善/下降数见summary.csv及metrics.json。\n\n## 十种子汇总（均值 ± 样本标准差，ddof=1）\n|指标|matched random 8/8|eval-only 40/8|\n|---|---:|---:|\n"
    for col in columns:
        values = [results['seed_summary'][c][col] for c in ('matched_random_8_8', 'eval_only_40_8')]
        report += f"|{col}|{values[0]['mean']:.6f} ± {values[0]['std']:.6f}|{values[1]['mean']:.6f} ± {values[1]['std']:.6f}|\n"
    report += "\n## 实际结论\n"
    for condition in ("matched_random_8_8", "eval_only_40_8"):
        subset = summary[summary.condition == condition]
        report += f"- {condition}：十种子中{int((subset.delta_auc_vs_baseline < 0).sum())}/10总体AUC低于原取模baseline；平均AUC变化{subset.delta_auc_vs_baseline.mean():+.6f}，平均被试宏AUC变化{subset.delta_subject_auc_mean_vs_baseline.mean():+.6f}。\n"
    report += "原取模8/8的收益在随机窗口中没有保持，结果支持026对缓存行序定义的分组存在明显敏感性；不能将原取模优势泛化为任意8窗口分组优势。\n"
    report += "\n## 解释边界与产物\n随机重分组改变同组样本构成，因此衡量缓存行序定义的窗口敏感性，不是证明时间因果或真实独立试次性能。10种子共享被试/固定折，标准差仅是分组敏感性，不是独立数据置信区间；仍为被试内无标签传导式统计。不得挑选最好seed作为结论。每个条件/seed均保存2760行OOF及80行公开预测；公开每用户原本仅8行，因此eval-only的十个公开文件相同是预期行为。未读取公开标签，未输出替代逐seed结果的集成。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
