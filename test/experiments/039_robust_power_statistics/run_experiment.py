import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
OUT = HERE / "predictions"
SPECTRAL = (HERE / "../001_spectral_logpower/cache/features.npz").resolve()
REFERENCE = (HERE / "../002_subject_normalization/predictions/subject_zscore_exclude_hc43_oof.csv").resolve()
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_cache_alignment(data):
    assert data["X_train"].shape[1] >= 150
    assert data["X_public"].shape[1] >= 150
    assert len(data["subject_ids"]) == len(data["y_train"])
    assert len(data["public_user_ids"]) == len(data["public_trial_ids"])


def robust_power_features(X, subject_ids):
    X = normalize_by_subject(np.asarray(X, dtype=float), np.asarray(subject_ids), "zscore")
    if X.shape[1] != 150:
        raise ValueError("robust power input must have 150 dimensions")
    blocks = X.reshape(len(X), 30, 5)
    median = np.median(blocks, axis=1)
    mad = np.median(np.abs(blocks - median[:, None, :]), axis=1)
    iqr = np.percentile(blocks, 75, axis=1) - np.percentile(blocks, 25, axis=1)
    return np.concatenate([X, median, mad, iqr], axis=1)


def make_subject_folds(subject_ids, groups, seed):
    subjects = np.unique(subject_ids)
    subject_groups = np.asarray([groups[np.flatnonzero(subject_ids == s)[0]] for s in subjects])
    splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
    assignment = {}
    for fold, valid in enumerate(splitter.split(subjects, subject_groups)):
        for subject in subjects[valid[1]]:
            assignment[subject] = fold
    return np.asarray([assignment[s] for s in subject_ids], dtype=int)


def assert_subject_isolation(subject_ids, folds):
    for fold in range(5):
        if set(subject_ids[folds == fold]) & set(subject_ids[folds != fold]):
            raise ValueError("subject leakage")


def model(seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**CONFIG["hgb"], random_state=seed))


def metrics(y, p):
    pred = p >= CONFIG["threshold"]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": CONFIG["threshold"],
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def summarize(y, p, frame, folds):
    subjects = []
    for subject in frame.subject_id.unique():
        mask = frame.subject_id.to_numpy() == subject
        subjects.append({"subject_id": subject, "health_group": frame.loc[mask, "health_group"].iloc[0], **metrics(y[mask], p[mask])})
    groups = {}
    for group in ("HC", "DEP"):
        mask = frame.health_group.to_numpy() == group
        rows = [r for r in subjects if r["health_group"] == group]
        groups[group] = {"n_subjects": len(rows), "pooled": metrics(y[mask], p[mask]),
                         "subject_auc_mean": float(np.mean([r["auc"] for r in rows]))}
    fold_rows = [{"fold": f, "n_subjects": int(frame.loc[folds == f, "subject_id"].nunique()),
                  "n_samples": int((folds == f).sum()), **metrics(y[folds == f], p[folds == f])} for f in range(5)]
    return {"overall": metrics(y, p), "folds": fold_rows,
            "subject_auc_mean": float(np.mean([r["auc"] for r in subjects])), "subjects": subjects,
            "health_groups": groups}


def save_public(name, p, data):
    frame = pd.DataFrame({"user_id": data["public_user_ids"], "trial_id": data["public_trial_ids"],
                          "prob. of positive emotion": p})
    validate_submission(frame, [f"P_test{i}" for i in range(1, 11)])
    frame.to_csv(OUT / f"{name}_public.csv", index=False)


def save_public_fold_probabilities(name, probabilities, data):
    frame = pd.DataFrame({"user_id": data["public_user_ids"], "trial_id": data["public_trial_ids"]})
    for fold, values in enumerate(probabilities):
        frame[f"fold_{fold}_prob_positive"] = values
    frame.to_csv(OUT / f"{name}_public_folds.csv", index=False)


def run_single(name, X, XP, y, reference, folds, data, suffix=""):
    p = np.full(len(y), np.nan)
    fold_rows = []
    public_fold_probabilities = []
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        fitted = model(2026 + fold)
        fitted.fit(X[train], y[train]); p[valid] = fitted.predict_proba(X[valid])[:, 1]
        public_fold_probabilities.append(fitted.predict_proba(XP)[:, 1])
        fold_rows.append({"fold": fold, **metrics(y[valid], p[valid])})
    save_public_fold_probabilities(f"{name}{suffix}", public_fold_probabilities, data)
    if not np.isfinite(p).all() or not np.all((p >= 0) & (p <= 1)):
        raise RuntimeError("invalid or incomplete OOF")
    frame = reference.copy(); frame["prob_positive"] = p
    frame.to_csv(OUT / f"{name}{suffix}_oof.csv", index=False)
    fitted = model(2026); fitted.fit(X, y); save_public(f"{name}{suffix}", fitted.predict_proba(XP)[:, 1], data)
    result = summarize(y, p, reference, folds); result["folds"] = fold_rows
    return result


def run_fixed_fusion(sx, sp, rx, rp, y, reference, folds, data):
    ps = np.full(len(y), np.nan); pr = np.full(len(y), np.nan)
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        ms, mr = model(2026 + fold), model(2026 + fold)
        ms.fit(sx[train], y[train]); mr.fit(rx[train], y[train])
        ps[valid] = ms.predict_proba(sx[valid])[:, 1]; pr[valid] = mr.predict_proba(rx[valid])[:, 1]
    p = .5 * ps + .5 * pr
    frame = reference.copy(); frame["prob_positive"] = p; frame.to_csv(OUT / "fixed_05_05_oof.csv", index=False)
    ms, mr = model(2026), model(2026); ms.fit(sx, y); mr.fit(rx, y)
    save_public("fixed_05_05", .5 * ms.predict_proba(sp)[:, 1] + .5 * mr.predict_proba(rp)[:, 1], data)
    return summarize(y, p, reference, folds)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_cache(SPECTRAL); assert_cache_alignment(data)
    keep = data["subject_ids"] != CONFIG["excluded_subject"]
    subjects, groups, y = data["subject_ids"][keep], data["health_groups"][keep], data["y_train"][keep]
    if len(y) != 2760 or len(np.unique(subjects)) != 69: raise RuntimeError("unexpected filtered data")
    reference = pd.read_csv(REFERENCE)
    if not np.array_equal(reference.subject_id.to_numpy(), subjects): raise RuntimeError("reference alignment mismatch")
    folds = reference.fold.to_numpy(dtype=int); assert_subject_isolation(subjects, folds)
    reference["y_true"] = y
    sx = normalize_by_subject(data["X_train"][keep, :150], subjects, "zscore")
    sp = normalize_by_subject(data["X_public"][:, :150], data["public_user_ids"], "zscore")
    rx = robust_power_features(data["X_train"][keep, :150], subjects)
    rp = robust_power_features(data["X_public"][:, :150], data["public_user_ids"])
    runs = {}; summary = []
    with threadpool_limits(limits=1):
        for name, X, XP in (("spectral_control", sx, sp), ("robust_power", rx, rp)):
            result = run_single(name, X, XP, y, reference, folds, data); result["n_features"] = X.shape[1]; runs[name] = result
            summary.append({"run_name": name, "n_features": X.shape[1], **result["overall"], "subject_auc_mean": result["subject_auc_mean"], "hc_auc": result["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": result["health_groups"]["DEP"]["pooled"]["auc"]})
        result = run_fixed_fusion(sx, sp, rx, rp, y, reference, folds, data); result["n_features"] = 165; runs["fixed_05_05"] = result
        summary.append({"run_name": "fixed_05_05", "n_features": 165, **result["overall"], "subject_auc_mean": result["subject_auc_mean"], "hc_auc": result["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": result["health_groups"]["DEP"]["pooled"]["auc"]})
        for seed in CONFIG["repeated_fold_seeds"]:
            rf = make_subject_folds(subjects, groups, seed); assert_subject_isolation(subjects, rf)
            pd.DataFrame({"subject_id": subjects, "health_group": groups, "original_trial_id": data["original_trial_ids"][keep], "fold": rf, "y_true": y}).to_csv(OUT / f"folds_seed{seed}.csv", index=False)
            for name, X, XP in (("spectral_control", sx, sp), ("robust_power", rx, rp)):
                result = run_single(name, X, XP, y, reference.assign(fold=rf), rf, data, f"_seed{seed}")
                result["n_features"] = X.shape[1]; result["seed"] = seed; runs[f"{name}_seed{seed}"] = result
    output = {"experiment_config": CONFIG, "provenance": {"spectral_cache_sha256": sha256(SPECTRAL), "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}, "protocol": {"cache_alignment_checked": True, "excluded_subject": "HC43", "subjectwise_unlabeled_zscore": True, "no_labels_in_features": True, "row_order_invariant_definition": True, "fixed_outer_reference": True, "repeated_folds_not_pooled": True}, "runs": runs}
    (HERE / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    report = "# 实验039：无标签逐窗口跨通道稳健功率统计\n\n复用001的150维log-bandpower缓存，排除HC43后为69名被试2760个训练窗口，公开80窗口。每被试先对150维做独立无标签z-score；对每个窗口、每个频带的30个通道z-score计算median、MAD和IQR，追加15维形成165维。该统计逐窗口计算，不使用标签、trial id、相邻行或公开标签，且与行序无关。固定复用002外层5折，HGB沿用036/038，阈值0.5；固定融合每折分别重训两个模型并按0.5/0.5融合。spectral_control与robust_power另用seed=2027..2031进行独立被试级5折稳定性复核，重复OOF不拼接。\n\n## 固定外层结果\n\n|方法|维度|AUC|F1|平衡准确率|被试宏AUC|HC AUC|DEP AUC|\n|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"|{r['run_name']}|{r['n_features']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['balanced_accuracy']:.6f}|{r['subject_auc_mean']:.6f}|{r['hc_auc']:.6f}|{r['dep_auc']:.6f}|" for r in summary) + "\n\n## 解释边界\n\n重复折仅报告稳定性，不合并为独立显著性检验；公开集仅保存预测，不宣称公开成绩。若稳健特征未提升固定外层AUC，则停止该方向。完整逐折、逐被试、逐seed指标及预测见metrics.json和predictions。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
