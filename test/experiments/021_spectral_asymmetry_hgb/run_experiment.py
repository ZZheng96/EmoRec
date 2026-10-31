import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emo_rec.features import ASYMMETRY_PAIRS, CHANNEL_NAMES, FREQUENCY_BANDS
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
META_COLUMNS = {"subject_ids": "subject_id", "health_groups": "health_group",
                "original_trial_ids": "original_trial_id", "y_train": "y_true"}


def load_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def validate_columns(cache):
    spectral = [f"log_bandpower_{channel}_{band}" for channel in CHANNEL_NAMES for band in FREQUENCY_BANDS]
    asymmetry = [f"asymmetry_{left}_{right}_{band}" for left, right in ASYMMETRY_PAIRS for band in FREQUENCY_BANDS]
    names = cache["feature_names"].tolist()
    if len(spectral) != 150 or len(asymmetry) != 60 or names != spectral + asymmetry:
        raise RuntimeError("feature column alignment mismatch: expected ordered spectral150 + asymmetry60")
    indices = {name: i for i, name in enumerate(names)}
    for key, rows in (("X_train", 2800), ("X_public", 80)):
        X = cache[key]
        if X.shape != (rows, 210) or not np.isfinite(X).all():
            raise RuntimeError(f"invalid feature matrix: {key}")
        for left, right in ASYMMETRY_PAIRS:
            for band in FREQUENCY_BANDS:
                expected = X[:, indices[f"log_bandpower_{left}_{band}"]] - X[:, indices[f"log_bandpower_{right}_{band}"]]
                if not np.allclose(X[:, indices[f"asymmetry_{left}_{right}_{band}"]], expected, rtol=1e-12, atol=1e-12):
                    raise RuntimeError(f"asymmetry column alignment mismatch: {key} {left} {right} {band}")
    return np.arange(150), np.arange(210)


def validate_metadata(cache, train, public):
    for key, column in META_COLUMNS.items():
        if not np.array_equal(cache[key], train[column].to_numpy()):
            raise RuntimeError(f"metadata alignment mismatch: {key}")
    for key, column in (("public_user_ids", "user_id"), ("public_trial_ids", "trial_id")):
        if not np.array_equal(cache[key], public[column].to_numpy()):
            raise RuntimeError(f"metadata alignment mismatch: {key}")
    subjects, counts = np.unique(cache["subject_ids"], return_counts=True)
    if len(subjects) != 70 or not np.all(counts == 40):
        raise RuntimeError("expected 70 subjects with 40 windows each")
    if not np.array_equal(cache["public_user_ids"], np.repeat([f"P_test{i}" for i in range(1, 11)], 8)):
        raise RuntimeError("unexpected public users or ordering")
    if not np.array_equal(cache["public_trial_ids"], np.tile(np.arange(1, 9), 10)):
        raise RuntimeError("unexpected public trials or ordering")


def assert_subject_isolation(subjects, folds):
    if set(folds) != set(range(5)):
        raise RuntimeError("expected exactly five outer folds")
    for fold in range(5):
        if set(subjects[folds == fold]) & set(subjects[folds != fold]):
            raise RuntimeError(f"subject leakage in fold {fold}")


def load_outer_reference(path, cache, keep):
    frame = pd.read_csv(path)
    for key, column in META_COLUMNS.items():
        if not np.array_equal(frame[column].to_numpy(), cache[key][keep]):
            raise RuntimeError(f"outer metadata alignment mismatch: {key}")
    folds = frame["fold"].to_numpy()
    assert_subject_isolation(cache["subject_ids"][keep], folds)
    return frame, folds


def metrics(y, p):
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, p)), "binary_f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": .5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def model(config, seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=seed))


def grouped_metrics(reference, probabilities):
    result = {"overall": metrics(reference.y_true.to_numpy(), probabilities)}
    result["groups"] = {}
    for group in ("HC", "DEP"):
        mask = reference.health_group.to_numpy() == group
        result["groups"][group] = metrics(reference.y_true.to_numpy()[mask], probabilities[mask])
    return result


def evaluate(name, X, X_public, reference, folds, cache, config):
    y = reference.y_true.to_numpy()
    oof = np.full(len(y), np.nan)
    fold_rows = []
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        fitted = model(config, config["random_seed"] + fold)
        fitted.fit(X[train], y[train])
        oof[valid] = fitted.predict_proba(X[valid])[:, 1]
        fold_rows.append({"fold": fold, "n_train_subjects": int(reference.loc[train, "subject_id"].nunique()),
                          "n_validation_subjects": int(reference.loc[valid, "subject_id"].nunique()),
                          **grouped_metrics(reference.loc[valid], oof[valid])})
        print(f"{name}: fold {fold + 1}/5", flush=True)
    if len(oof) != 2760 or not np.isfinite(oof).all():
        raise RuntimeError("incomplete OOF predictions")
    output = reference.copy()
    output["prob_positive"] = oof
    output.to_csv(PREDICTIONS_DIR / f"{name}_oof.csv", index=False)
    final = model(config, config["random_seed"])
    final.fit(X, y)
    public = pd.DataFrame({"user_id": cache["public_user_ids"], "trial_id": cache["public_trial_ids"],
                           "prob. of positive emotion": final.predict_proba(X_public)[:, 1]})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
    public.to_csv(PREDICTIONS_DIR / f"{name}_public.csv", index=False)
    return {"n_features": X.shape[1], **grouped_metrics(reference, oof), "folds": fold_rows}


def main():
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    cache = load_cache(EXPERIMENT_DIR / config["spectral_cache"])
    spectral, combined = validate_columns(cache)
    validate_metadata(cache, pd.read_csv(EXPERIMENT_DIR / config["metadata_reference"]),
                      pd.read_csv(EXPERIMENT_DIR / config["public_reference"]))
    keep = cache["subject_ids"] != config["excluded_subject"]
    subjects = cache["subject_ids"][keep]
    if keep.sum() != 2760 or len(np.unique(subjects)) != 69:
        raise RuntimeError("unexpected HC43 exclusion result")
    reference, folds = load_outer_reference(EXPERIMENT_DIR / config["outer_reference"], cache, keep)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    runs = {}
    for name, columns in (("spectral150", spectral), ("spectral_asymmetry210", combined)):
        X = normalize_by_subject(cache["X_train"][keep][:, columns], subjects, "zscore", 1e-12)
        X_public = normalize_by_subject(cache["X_public"][:, columns], cache["public_user_ids"], "zscore", 1e-12)
        runs[name] = evaluate(name, X, X_public, reference, folds, cache, config)
    summary = []
    for name, run in runs.items():
        summary.append({"run_name": name, "n_features": run["n_features"], "oof_auc": run["overall"]["roc_auc"],
                        "oof_f1": run["overall"]["binary_f1"], "oof_balanced_accuracy": run["overall"]["balanced_accuracy"],
                        "hc_auc": run["groups"]["HC"]["roc_auc"], "hc_f1": run["groups"]["HC"]["binary_f1"],
                        "dep_auc": run["groups"]["DEP"]["roc_auc"], "dep_f1": run["groups"]["DEP"]["binary_f1"]})
    result = {"experiment_config": config, "protocol": {"feature_names": cache["feature_names"].tolist(),
              "train_and_public_asymmetry_identity_checked": True, "row_metadata_checked_against_001": True,
              "outer_folds_reused_from_002": True, "subject_isolation_checked": True,
              "oof_rows_per_condition": 2760, "public_rows_per_condition": 80,
              "no_tuning_or_model_selection": True}, "runs": runs}
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    report = """# 实验021：固定HGB的频谱与不对称特征对照

## 固定协议与校验

读取001缓存feature_names，严格核验30通道×5频段的150个log_bandpower列与12对通道×5频段的60个asymmetry列，顺序与提取器一致。训练和公开矩阵共用同一列索引，并逐列验证asymmetry=左侧log_bandpower−右侧log_bandpower，避免公开列错位。训练subject_ids、health_groups、original_trial_ids、y_train及公开用户/试次ID逐行对齐001已有预测元数据；排除HC43后再次逐行对齐002固定外层5折，核验被试隔离。

两组均使用被试内无标签zscore（各训练/验证被试自身40窗口；各公开用户自身8窗口），归一化不接收标签。因每个被试仅属于一个折，全体被试分别归一化与训练/验证分别归一化等价。StandardScaler在每折pipeline中仅拟合训练数据。固定HGB：max_iter=200、learning_rate=.03、max_leaf_nodes=15、min_samples_leaf=20、l2_regularization=1、early_stopping=False、random_state=2026+fold。最终公开预测用全体69名训练被试拟合，seed=2026。无调参、无融合、无结果驱动的模型选择；阈值固定0.5。

## 真实运行结果

两组各保存2760行OOF（69名被试）和80行公开预测。metrics.json包含总体、HC、DEP及每折AUC/F1/平衡准确率和混淆矩阵。

| 输入 | OOF AUC | OOF F1 | 平衡准确率 | HC AUC | DEP AUC |
|---|---:|---:|---:|---:|---:|
"""
    for row in summary:
        report += f"| {row['run_name']} | {row['oof_auc']:.6f} | {row['oof_f1']:.6f} | {row['oof_balanced_accuracy']:.6f} | {row['hc_auc']:.6f} | {row['dep_auc']:.6f} |\n"
    delta = summary[1]["oof_auc"] - summary[0]["oof_auc"]
    report += f"\n加入不对称特征的OOF AUC差值为{delta:+.6f}；仅作为预设对照的描述性结果，不据此选模型或调整参数。\n"
    report += "\n## 局限与专项测试\n\n复用外层折的探索性比较不能代替独立被试验证，不声明统计显著性。被试内归一化使用全部无标签窗口，属于传导式设定，不能直接代表在线单窗口推理。公开集无标签，不能报告公开集准确率。test_protocol.py覆盖列名/顺序、训练和公开不对称恒等式、元数据逐行对齐、固定折被试隔离、无标签归一化独立性、固定参数及输出完整性。\n"
    (EXPERIMENT_DIR / "实验报告.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
