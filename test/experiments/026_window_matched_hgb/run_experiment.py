import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("experiment019", ROOT / "test/experiments/019_window_matched_normalization/run_experiment.py")
_protocol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_protocol)
segment_batches = _protocol.segment_batches
normalize = _protocol.normalize
align_reference = _protocol.align_reference


def metrics(y, p):
    pred = p >= 0.5
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "threshold": 0.5}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    cache_path = (HERE / config["feature_cache"]).resolve()
    reference_path = (HERE / config["reference_oof"]).resolve()
    with np.load(cache_path, allow_pickle=False) as cache:
        data = {key: cache[key] for key in cache.files}
    keep = data["subject_ids"] != config["excluded_subject"]
    cache_rows = np.flatnonzero(keep)
    for key in ("X_train", "y_train", "subject_ids", "health_groups", "original_trial_ids"):
        data[key] = data[key][keep]
    reference = pd.read_csv(reference_path)
    folds = align_reference(data, reference)
    subjects, y = data["subject_ids"], data["y_train"]
    if len(y) != config["n_samples"] or len(np.unique(subjects)) != config["expected_subjects"]:
        raise RuntimeError("Unexpected filtered cache dimensions")
    batches = segment_batches(subjects)
    feature_mask = np.asarray([str(name).startswith(config["feature_prefix"]) for name in data["feature_names"]])
    X = data["X_train"][:, feature_mask]
    if X.shape[1] != config["n_features"] or not np.isfinite(X).all():
        raise RuntimeError("Unexpected feature matrix")
    normalized = {size: normalize(X, subjects, batches, size, config["near_zero_std"]) for size in (8, 40)}
    public_subjects = data["public_user_ids"]
    if not all(np.sum(public_subjects == s) == 8 for s in np.unique(public_subjects)):
        raise RuntimeError("Public rows are not 8 per user")
    public_X = normalize(data["X_public"][:, feature_mask], public_subjects, None, 40, config["near_zero_std"])
    output = HERE / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    results = {"config": config, "provenance": {"cache_sha256": sha256(cache_path), "reference_sha256": sha256(reference_path),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "threads": 1}, "runs": {}}
    summaries = []

    def model(fold):
        return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=2026 + fold))

    with threadpool_limits(limits=1):
        for name, (train_size, eval_size) in config["conditions"].items():
            p = np.full(len(y), np.nan)
            fold_rows = []
            for fold in range(config["outer_folds"]):
                train, valid = folds != fold, folds == fold
                estimator = model(fold)
                estimator.fit(normalized[train_size][train], y[train])
                p[valid] = estimator.predict_proba(normalized[eval_size][valid])[:, 1]
                fold_rows.append({"fold": fold, "n_samples": int(valid.sum()), **metrics(y[valid], p[valid])})
                print(f"{name}: fold {fold + 1}/5", flush=True)
            if not np.isfinite(p).all():
                raise RuntimeError("Incomplete OOF predictions")
            oof = reference.copy()
            oof["prob_positive"] = p
            oof["cache_row"] = cache_rows
            oof["segment_position"] = batches
            oof["normalization_batch"] = batches if eval_size == 8 else -1
            oof.to_csv(output / f"{name}_oof.csv", index=False)
            subject_rows = [{"subject_id": s, "health_group": data["health_groups"][subjects == s][0], **metrics(y[subjects == s], p[subjects == s])} for s in np.unique(subjects)]
            health = {}
            for group in ("HC", "DEP"):
                mask = data["health_groups"] == group
                group_rows = [r for r in subject_rows if r["health_group"] == group]
                health[group] = {"n_subjects": len(group_rows), "pooled": metrics(y[mask], p[mask]),
                                 "subject_auc_mean": float(np.mean([r["auc"] for r in group_rows])),
                                 "subject_auc_std": float(np.std([r["auc"] for r in group_rows], ddof=1))}
            run = {"overall": metrics(y, p), "folds": fold_rows, "subjects": subject_rows,
                   "health_groups": health, "subject_auc_mean": float(np.mean([r["auc"] for r in subject_rows])),
                   "fold_auc_mean": float(np.mean([r["auc"] for r in fold_rows])),
                   "segments": [{"segment_position": b, **metrics(y[batches == b], p[batches == b])} for b in range(5)]}
            if name == "control_40_40":
                ref016 = pd.read_csv(ROOT / "test/experiments/016_hgb_fusion/predictions/hgb_oof.csv")
                if not np.array_equal(ref016[["subject_id", "health_group", "original_trial_id", "fold", "y_true"]].to_numpy(), reference[["subject_id", "health_group", "original_trial_id", "fold", "y_true"]].to_numpy()):
                    raise RuntimeError("016 reference alignment mismatch")
                diff = float(np.max(np.abs(p - ref016["prob_positive"].to_numpy())))
                run["reference_016_max_abs_probability_difference"] = diff
                if diff >= 1e-8:
                    raise RuntimeError(f"Control failed 016 reproduction: {diff}")
            results["runs"][name] = run
            summaries.append({"run_name": name, "n_samples": len(y), "n_subjects": len(np.unique(subjects)), **run["overall"],
                              "fold_auc_mean": run["fold_auc_mean"], "subject_auc_mean": run["subject_auc_mean"],
                              "hc_pooled_auc": health["HC"]["pooled"]["auc"], "dep_pooled_auc": health["DEP"]["pooled"]["auc"],
                              "hc_subject_auc_mean": health["HC"]["subject_auc_mean"], "dep_subject_auc_mean": health["DEP"]["subject_auc_mean"]})
            final = model(0)
            final.fit(normalized[train_size], y)
            public_p = final.predict_proba(public_X)[:, 1]
            public = pd.DataFrame({"user_id": public_subjects, "trial_id": data["public_trial_ids"], "prob. of positive emotion": public_p})
            validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
            public.to_csv(output / f"{name}_public.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE / "summary.csv", index=False)
    (HERE / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    control = results["runs"]["control_40_40"]
    report = "# 实验026：窗口匹配归一化与HGB\n\n"
    report += "## 协议与泄漏审计\n- 严格复用019的stable cache row ordinal % 5分为5批、8/40行逐被试总体zscore及002固定五折；输入为001的150维log_bandpower，排除HC43，2760行，未读取公开标签。\n"
    report += "- 模型为StandardScaler + HistGradientBoostingClassifier：max_iter=200、learning_rate=.03、max_leaf_nodes=15、min_samples_leaf=20、l2_regularization=1、early_stopping=False；fold seed=2026+fold；数值线程固定1。无调参、无阈值选择，F1固定0.5。\n"
    report += "- control_40_40逐行复现016 hgb_oof；matched_8_8训练/评估均8行；eval_only_40_8训练40行、评估8行。每行只预测一次，公开集每用户8行独立无标签归一化并校验提交格式。\n\n## 结果\n"
    report += "|协议|OOF AUC|F1|平衡准确率|被试宏AUC|HC pooled AUC|DEP pooled AUC|\n|---|---:|---:|---:|---:|---:|---:|\n"
    for row in summaries:
        report += f"|{row['run_name']}|{row['auc']:.6f}|{row['f1']:.6f}|{row['balanced_accuracy']:.6f}|{row['subject_auc_mean']:.6f}|{row['hc_pooled_auc']:.6f}|{row['dep_pooled_auc']:.6f}|\n"
    report += f"\ncontrol与016逐行概率最大绝对差：{control['reference_016_max_abs_probability_difference']:.3g}。完整分折、分被试、HC/DEP、5个segment指标见metrics.json。\n"
    report += "\n## 解释边界\n窗口按缓存行序取模，可能与实验条件/原trial排列相关；这不是随机窗口采样。被试内统计是无标签、传导式推断，行间相关，不能把2760行当独立样本。反复查看同一OOF进行协议选择会产生选择偏差，本实验不将比较结果解释为无偏的选模后性能或公开域保证。\n"
    report += "\n运行：`.venv\\Scripts\\python.exe test/experiments/026_window_matched_hgb/run_experiment.py`\n测试：`.venv\\Scripts\\python.exe -m pytest test/experiments/026_window_matched_hgb/test_protocol.py -q`\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
