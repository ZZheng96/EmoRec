import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
OOF_COLUMNS = ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"]
META_KEYS = ("subject_ids", "health_groups", "original_trial_ids", "y_train")


def load_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def assert_cache_alignment(spectral, graph):
    for key in META_KEYS + ("public_user_ids", "public_trial_ids"):
        if not np.array_equal(spectral[key], graph[key]):
            raise RuntimeError(f"cache alignment mismatch: {key}")
    if spectral["X_train"].shape[0] != 2800 or graph["X_train"].shape[0] != 2800:
        raise RuntimeError("unexpected cache row count")


def load_outer_reference(path, reference):
    frame = pd.read_csv(path)
    if list(frame.columns) != OOF_COLUMNS:
        raise RuntimeError("Unexpected outer reference columns")
    for key in ("subject_id", "health_group", "original_trial_id", "y_true"):
        if not np.array_equal(frame[key].to_numpy(), reference[key].to_numpy()):
            raise RuntimeError(f"outer metadata alignment mismatch: {key}")
    return frame["fold"].to_numpy(dtype=int)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, p)), "binary_f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": 0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def model(config, seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=seed))


def assert_subject_isolation(subjects, folds):
    for fold in range(5):
        if set(subjects[folds != fold]) & set(subjects[folds == fold]):
            raise RuntimeError(f"subject leakage in fold {fold}")


def evaluate(name, X, X_public, reference, folds, config):
    y = reference["y_true"].to_numpy()
    oof = np.full(len(y), np.nan)
    fold_rows = []
    for fold in range(config["outer_folds"]):
        train, valid = folds != fold, folds == fold
        fitted = model(config, config["random_seed"] + fold)
        fitted.fit(X[train], y[train])
        oof[valid] = fitted.predict_proba(X[valid])[:, 1]
        row = {"fold": fold, "n_train_subjects": int(reference.loc[train, "subject_id"].nunique()),
               "n_validation_subjects": int(reference.loc[valid, "subject_id"].nunique()),
               "overall": metrics(y[valid], oof[valid])}
        for group in ("HC", "DEP"):
            mask = valid & (reference["health_group"].to_numpy() == group)
            row[group] = metrics(y[mask], oof[mask])
        fold_rows.append(row)
    if not np.isfinite(oof).all():
        raise RuntimeError(f"incomplete OOF predictions for {name}")
    output = reference.copy()
    output["prob_positive"] = oof
    output.to_csv(PREDICTIONS_DIR / f"{name}_oof.csv", index=False)
    final = model(config, config["random_seed"])
    final.fit(X, y)
    public = pd.DataFrame({"user_id": reference.attrs["public_user_ids"], "trial_id": reference.attrs["public_trial_ids"],
                           "prob. of positive emotion": final.predict_proba(X_public)[:, 1]})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
    public.to_csv(PREDICTIONS_DIR / f"{name}_public.csv", index=False)
    groups = {group: metrics(y[reference["health_group"].to_numpy() == group], oof[reference["health_group"].to_numpy() == group]) for group in ("HC", "DEP")}
    return {"n_features": int(X.shape[1]), "overall": metrics(y, oof), "groups": groups, "folds": fold_rows}


def main():
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    spectral = load_cache((EXPERIMENT_DIR / config["spectral_cache"]).resolve())
    graph = load_cache((EXPERIMENT_DIR / config["graph_cache"]).resolve())
    assert_cache_alignment(spectral, graph)
    spectral_mask = np.asarray([str(n).startswith("log_bandpower_") for n in spectral["feature_names"]])
    if spectral_mask.sum() != 150 or graph["X_train"].shape[1] != 150:
        raise RuntimeError("expected spectral150 and graph150")
    keep = spectral["subject_ids"] != config["excluded_subject"]
    if keep.sum() != config["expected_oof_rows"] or len(np.unique(spectral["subject_ids"][keep])) != config["expected_subjects"]:
        raise RuntimeError("unexpected HC43 exclusion result")
    subjects = spectral["subject_ids"][keep]
    reference = pd.DataFrame({"subject_id": subjects, "health_group": spectral["health_groups"][keep],
                              "original_trial_id": spectral["original_trial_ids"][keep], "fold": -1, "y_true": spectral["y_train"][keep]})
    folds = load_outer_reference((EXPERIMENT_DIR / config["outer_reference"]).resolve(), reference)
    reference["fold"] = folds
    assert_subject_isolation(subjects, folds)
    reference.attrs["public_user_ids"] = spectral["public_user_ids"]
    reference.attrs["public_trial_ids"] = spectral["public_trial_ids"]
    spectral_X = normalize_by_subject(spectral["X_train"][keep][:, spectral_mask], subjects, "zscore", 1e-12)
    graph_X = normalize_by_subject(graph["X_train"][keep], subjects, "zscore", 1e-12)
    spectral_public = normalize_by_subject(spectral["X_public"][:, spectral_mask], spectral["public_user_ids"], "zscore", 1e-12)
    graph_public = normalize_by_subject(graph["X_public"], graph["public_user_ids"], "zscore", 1e-12)
    conditions = {"spectral150": (spectral_X, spectral_public), "graph150": (graph_X, graph_public),
                  "concat300": (np.concatenate([spectral_X, graph_X], axis=1), np.concatenate([spectral_public, graph_public], axis=1))}
    runs = {name: evaluate(name, X, X_public, reference, folds, config) for name, (X, X_public) in conditions.items()}
    summary = []
    for name, result in runs.items():
        summary.append({"run_name": name, "n_features": result["n_features"], "oof_auc": result["overall"]["roc_auc"],
                        "oof_f1": result["overall"]["binary_f1"], "oof_balanced_accuracy": result["overall"]["balanced_accuracy"],
                        "hc_auc": result["groups"]["HC"]["roc_auc"], "hc_f1": result["groups"]["HC"]["binary_f1"],
                        "dep_auc": result["groups"]["DEP"]["roc_auc"], "dep_f1": result["groups"]["DEP"]["binary_f1"]})
    result = {"experiment_config": config, "protocol": {"cache_alignment_checked": True, "outer_folds_reused_from_002": True,
        "excluded_subject": "HC43", "target_propagation": "按被试无标签zscore；公开集按用户自身8行zscore",
        "subject_isolation_checked": True, "no_tuning_fusion_or_model_selection": True, "threshold": 0.5}, "runs": runs}
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    report = """# 实验020：频率图特征在非线性HGB下的互补信息\n\n## 研究问题与协议\n\n读取并逐行核验001 spectral缓存与006 graph缓存的训练/公开集元数据；排除HC43后保留2760条OOF、69名被试。严格沿用002的固定5个外层被试折，模型为固定HGB参数，StandardScaler置于pipeline内，外层种子为2026+fold；不调参、不融合、不按结果选模型。三组输入均按被试无标签zscore，公开集按每个用户自身8行zscore。\n\n## 真实结果\n\n详见 `summary.csv`、`metrics.json` 和 `predictions/`；每组均生成2760行OOF和80行公开集预测，并记录总体、HC、DEP及每折AUC/F1/平衡准确率。\n\n"""
    report += "| 输入 | OOF AUC | OOF F1 | HC AUC | DEP AUC |\n|---|---:|---:|---:|---:|\n"
    for row in summary:
        report += f"| {row['run_name']} | {row['oof_auc']:.6f} | {row['oof_f1']:.6f} | {row['hc_auc']:.6f} | {row['dep_auc']:.6f} |\n"
    report += "\n## 局限\n\n外层折复用002，属于预先固定协议下的比较，仍不能替代全新独立被试验证；zscore使用每个被试的全部无标签行，符合本实验的传导式部署假设但不适用于严格离线单试次场景；公开集没有标签，不能据此判断泛化或选择模型。\n"
    (EXPERIMENT_DIR / "实验报告.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
