import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent


def segment_batches(subjects):
    batches = np.empty(len(subjects), dtype=np.int64)
    for subject in np.unique(subjects):
        indices = np.flatnonzero(subjects == subject)
        if len(indices) != 40:
            raise ValueError(f"Expected 40 rows for {subject}, got {len(indices)}")
        batches[indices] = np.arange(40) % 5
    return batches


def normalize(X, subjects, batches, size, epsilon=1e-12):
    X = np.asarray(X, dtype=np.float64)
    result = np.empty_like(X)
    for subject in np.unique(subjects):
        subject_mask = subjects == subject
        groups = range(5) if size == 8 else [None]
        for batch in groups:
            mask = subject_mask if batch is None else subject_mask & (batches == batch)
            values = X[mask]
            std = values.std(axis=0, ddof=0)
            result[mask] = (values - values.mean(axis=0)) / np.where(std > epsilon, std, 1.0)
    return result


def align_reference(data, reference):
    columns = {
        "subject_id": "subject_ids", "health_group": "health_groups",
        "original_trial_id": "original_trial_ids", "y_true": "y_train",
    }
    for column, key in columns.items():
        if not np.array_equal(reference[column].to_numpy(), data[key]):
            raise ValueError(f"Reference OOF row mismatch: {column}")
    folds = reference.fold.to_numpy(dtype=np.int64)
    if set(folds) != set(range(5)):
        raise ValueError("Expected fixed outer folds 0..4")
    for subject in np.unique(data["subject_ids"]):
        if len(np.unique(folds[data["subject_ids"] == subject])) != 1:
            raise ValueError("Subject crosses outer folds")
    return folds


def metrics(y, p):
    return {
        "auc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, p >= 0.5, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
    }


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
    assert len(y) == config["n_samples"] and len(np.unique(subjects)) == 69
    batches = segment_batches(subjects)
    features = np.asarray([name.startswith(config["feature_prefix"]) for name in data["feature_names"]])
    X = data["X_train"][:, features]
    assert X.shape[1] == config["n_features"] and np.isfinite(X).all()
    normalized = {size: normalize(X, subjects, batches, size, config["near_zero_std"]) for size in (8, 40)}
    public_subjects = data["public_user_ids"]
    assert all(np.sum(public_subjects == s) == 8 for s in np.unique(public_subjects))
    public_X = normalize(data["X_public"][:, features], public_subjects, None, 40, config["near_zero_std"])
    output = HERE / "predictions"
    output.mkdir(exist_ok=True)
    results = {
        "config": config,
        "provenance": {"cache_sha256": sha256(cache_path), "reference_sha256": sha256(reference_path),
                       "python": platform.python_version(), "numpy": np.__version__,
                       "pandas": pd.__version__, "sklearn": sklearn.__version__, "threads": 1},
        "runs": {},
    }
    summaries = []
    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(**config["model"], random_state=config["random_seed"]))

    with threadpool_limits(limits=1):
        for name, (train_size, eval_size) in config["conditions"].items():
            p = np.full(len(y), np.nan)
            fold_rows = []
            for fold in range(5):
                train, valid = folds != fold, folds == fold
                estimator = model()
                estimator.fit(normalized[train_size][train], y[train])
                p[valid] = estimator.predict_proba(normalized[eval_size][valid])[:, 1]
                fold_rows.append({"fold": fold, "n_samples": int(valid.sum()), **metrics(y[valid], p[valid])})
                print(f"{name}: fold {fold + 1}/5", flush=True)
            assert np.isfinite(p).all()
            oof = reference.drop(columns=["prob_positive"]).copy()
            oof["prob_positive"] = p
            oof["cache_row"] = cache_rows
            oof["segment_position"] = batches
            oof["normalization_batch"] = batches if eval_size == 8 else -1
            oof.to_csv(output / f"{name}_oof.csv", index=False)
            subject_rows = [{"subject_id": s, **metrics(y[subjects == s], p[subjects == s])} for s in np.unique(subjects)]
            run = {"overall": metrics(y, p), "folds": fold_rows, "subjects": subject_rows,
                   "segments": [{"segment_position": b, **metrics(y[batches == b], p[batches == b])} for b in range(5)],
                   "subject_auc_mean": float(np.mean([r["auc"] for r in subject_rows]))}
            if name == "control_40_40":
                run["reference_max_abs_probability_difference"] = float(np.max(np.abs(p - reference.prob_positive.to_numpy())))
                if run["reference_max_abs_probability_difference"] > 1e-6:
                    raise RuntimeError("Control failed to reproduce reference LR predictions")
            results["runs"][name] = run
            summaries.append({"run_name": name, "n_samples": len(y), "n_subjects": 69,
                              **run["overall"], "subject_auc_mean": run["subject_auc_mean"]})
            final_model = model()
            final_model.fit(normalized[train_size], y)
            public_p = final_model.predict_proba(public_X)[:, 1]
            pd.DataFrame({"user_id": public_subjects, "trial_id": data["public_trial_ids"],
                          "prob. of positive emotion": public_p}).to_csv(output / f"{name}_public.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE / "summary.csv", index=False)
    (HERE / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report = """# 019 窗口数量匹配归一化实验

## 协议与泄漏审计
- 排除 HC43，69 被试、2760 行、150 维 log_bandpower。逐行核对参考002 OOF的被试、健康组、原trial编号、标签，直接复用其5折，不重新划分。
- 使用002的 StandardScaler + LR（C=1，lbfgs，L2，max_iter=2000，seed=2026）；全局Scaler及分类器仅在外层训练被试拟合。没有选特征、调参、按验证标签选择协议或阈值；F1阈值固定0.5。
- 每被试按缓存稳定行序编号0..39，以 ordinal % 5 划为5批；每批8行，分别为8个连续5行块的相同segment位置。不用标签或 original_trial_id 选窗口；原编号仅用于逐行核验，不能用来分组，因为编号会重复。
- control_40_40：训练与验证分别按各被试全部40窗口计算均值和总体标准差，复现002。
- matched_8_8：训练与验证每行仅使用自己所属8窗口批次的均值和总体标准差；五批独立标准化再回填原行序，一个模型使用全部训练行，每个验证行恰预测一次。没有丢弃32行，也没有把同一行的五次预测平均。
- eval_only_40_8：训练每被试40窗口统计，验证每行所属8窗口批次统计；其与control的差异隔离了评估统计量改变的影响。
- 归一化包含预测窗口自身，属于公开集允许的无标签、被试内传导式推断，不是严格逐窗口在线推断。验证统计不读取标签，不跨被试，不进入训练模型；预计算归一化与逐折计算等价。8窗口协议不会读取该验证被试其他32窗口的特征。
- 公开集每用户8窗口独立标准化，用全部训练被试拟合后生成80行预测；没有公开标签，所以没有公开AUC/F1。control和eval_only训练相同，公开预测也相同，是预期行为。

## 实测结果

|协议|完整OOF AUC|F1 @ 0.5|平衡准确率|被试平均AUC|
|---|---:|---:|---:|---:|
"""
    for row in summaries:
        report += f"|{row['run_name']}|{row['auc']:.6f}|{row['f1']:.6f}|{row['balanced_accuracy']:.6f}|{row['subject_auc_mean']:.6f}|\n"
    control = results["runs"]["control_40_40"]
    report += f"\n控制组与002参考预测最大绝对差：{control['reference_max_abs_probability_difference']:.3g}。\n"
    for name in ("matched_8_8", "eval_only_40_8"):
        delta = results["runs"][name]["overall"]["auc"] - control["overall"]["auc"]
        report += f"\n{name} 相对40窗口控制的 AUC 变化：{delta:+.6f}。\n"
    report += """
## 结论边界与复现
这里只检验统计窗口数量及训练/评估匹配，不声称复现公开域：同segment方案保留了每个原trial一个窗口，但缓存排列可能关联实验条件顺序；跨trial无标签选择不会消除这种结构。五个segment全覆盖避免挑选最优位置，分segment和分折指标见metrics.json。窗口同被试且相关，不应把2760行当独立样本进行显著性检验。本实验未做显著性或公开泛化保证，也不把OOF比较用作无偏的选模后性能估计。

运行：`.venv\\Scripts\\python.exe test/experiments/019_window_matched_normalization/run_experiment.py`

测试：`.venv\\Scripts\\python.exe -m pytest test/experiments/019_window_matched_normalization/test_protocol.py -q`

依赖版本、输入SHA256、完整逐折/逐被试/逐segment指标保存在metrics.json；OOF额外保存原缓存行号与segment位置便于审计。使用单线程数值计算，不修改源缓存或既有实验。
"""
    (HERE / "实验报告.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
