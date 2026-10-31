import hashlib
import json
import platform
import time
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

def load_cache(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def assert_cache_alignment(spectral, temporal):
    for key in ("subject_ids", "health_groups", "original_trial_ids", "y_train", "public_user_ids", "public_trial_ids"):
        if not np.array_equal(spectral[key], temporal[key]):
            raise RuntimeError(f"cache alignment mismatch: {key}")

def make_subject_folds(subject_ids, health_groups, seed):
    subjects = np.unique(subject_ids)
    groups = np.asarray([health_groups[np.flatnonzero(subject_ids == s)[0]] for s in subjects])
    assignment = {}
    for fold, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(subjects, groups)):
        for subject in subjects[valid]: assignment[subject] = fold
    return np.asarray([assignment[s] for s in subject_ids], dtype=np.int64)

def assert_subject_isolation(subject_ids, folds):
    for fold in range(5):
        if set(subject_ids[folds != fold]) & set(subject_ids[folds == fold]):
            raise RuntimeError(f"subject leakage in fold {fold}")

def block_statistics(X, subjects, block_size):
    """Unlabeled contiguous within-subject blocks; block identity is row-order only."""
    X = np.asarray(X, dtype=np.float64)
    subjects = np.asarray(subjects)
    out = np.empty((len(X), X.shape[1] * 3), dtype=np.float64)
    for subject in np.unique(subjects):
        indices = np.flatnonzero(subjects == subject)
        for start in range(0, len(indices), block_size):
            rows = indices[start:start + block_size]
            mean = X[rows].mean(axis=0)
            std = X[rows].std(axis=0)
            out[rows] = np.concatenate((np.broadcast_to(mean, (len(rows), X.shape[1])),
                                        np.broadcast_to(std, (len(rows), X.shape[1])), X[rows] - mean), axis=1)
    if not np.isfinite(out).all(): raise RuntimeError("non-finite block statistics")
    return out

def consistency_features(X, subjects, block_size):
    X = np.asarray(X, dtype=np.float64)
    return np.concatenate((X, block_statistics(X, subjects, block_size)), axis=1)

def make_model(config, seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=seed))

def metrics(y, p):
    pred = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": 0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def summarize(y, p, ref, folds):
    subjects = []
    for s in np.unique(ref.subject_id):
        m = ref.subject_id.to_numpy() == s
        subjects.append({"subject_id": s, "health_group": ref.loc[m, "health_group"].iloc[0], **metrics(y[m], p[m])})
    groups = {}
    for g in ("HC", "DEP"):
        m = ref.health_group.to_numpy() == g; rows = [r for r in subjects if r["health_group"] == g]
        groups[g] = {"n_subjects": len(rows), "pooled": metrics(y[m], p[m]), "subject_auc_mean": float(np.mean([r["auc"] for r in rows]))}
    return {"overall": metrics(y, p), "folds": [{"fold": f, "n_subjects": int(ref.loc[folds == f, "subject_id"].nunique()), "n_samples": int((folds == f).sum()), **metrics(y[folds == f], p[folds == f])} for f in range(5)], "subject_auc_mean": float(np.mean([r["auc"] for r in subjects])), "health_groups": groups, "subjects": subjects}

def evaluate(name, X, XP, y, ref, folds, config):
    oof = np.full(len(y), np.nan)
    fold_public = []
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        model = make_model(config, 2026 + fold); model.fit(X[train], y[train])
        oof[valid] = model.predict_proba(X[valid])[:, 1]
        fold_public.append(model.predict_proba(XP)[:, 1])
    if not np.isfinite(oof).all(): raise RuntimeError(f"incomplete OOF: {name}")
    model = make_model(config, 2026); model.fit(X, y); public_p = model.predict_proba(XP)[:, 1]
    oof_frame = ref.copy(); oof_frame["prob_positive"] = oof; oof_frame.to_csv(OUT / f"{name}_oof.csv", index=False)
    public = pd.DataFrame({"user_id": ref.attrs["public_user_ids"], "trial_id": ref.attrs["public_trial_ids"], "prob. of positive emotion": public_p})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)]); public.to_csv(OUT / f"{name}_public.csv", index=False)
    pd.DataFrame(np.asarray(fold_public).T, columns=[f"fold_{i}" for i in range(5)]).assign(user_id=public.user_id, trial_id=public.trial_id).to_csv(OUT / f"{name}_public_fold_probabilities.csv", index=False)
    return {"n_features": int(X.shape[1]), "result": summarize(y, oof, ref, folds), "public_refit": True}

def main():
    started = time.perf_counter(); config = json.loads((HERE / "config.json").read_text(encoding="utf-8")); OUT.mkdir(exist_ok=True)
    spectral = load_cache((HERE / config["spectral_cache"]).resolve()); temporal = load_cache((HERE / config["temporal_cache"]).resolve()); assert_cache_alignment(spectral, temporal)
    all_folds = make_subject_folds(spectral["subject_ids"], spectral["health_groups"], config["fold_seed"]); keep = spectral["subject_ids"] != config["excluded_subject"]
    subjects, groups, y = spectral["subject_ids"][keep], spectral["health_groups"][keep], spectral["y_train"][keep]; folds = all_folds[keep]; assert_subject_isolation(subjects, folds)
    if len(y) != 2760 or len(np.unique(subjects)) != 69: raise RuntimeError("unexpected filtered data")
    sx = normalize_by_subject(spectral["X_train"][keep, :150], subjects, "zscore"); sp = normalize_by_subject(spectral["X_public"][:, :150], spectral["public_user_ids"], "zscore")
    tx = normalize_by_subject(temporal["X_train"][keep], subjects, "zscore"); tp = normalize_by_subject(temporal["X_public"], temporal["public_user_ids"], "zscore")
    cx = consistency_features(sx, subjects, 5); cp = consistency_features(sp, spectral["public_user_ids"], 2)
    ref = pd.DataFrame({"subject_id": subjects, "health_group": groups, "original_trial_id": spectral["original_trial_ids"][keep], "fold": folds, "y_true": y}); ref.attrs["public_user_ids"] = spectral["public_user_ids"]; ref.attrs["public_trial_ids"] = spectral["public_trial_ids"]
    runs = {}; summary = []
    with threadpool_limits(limits=1):
        for name, X, XP in (("spectral_control", sx, sp), ("temporal_control", tx, tp), ("consistency_spectral", cx, cp)):
            runs[name] = evaluate(name, X, XP, y, ref, folds, config); r = runs[name]["result"]
            summary.append({"run_name": name, "n_features": X.shape[1], **r["overall"], "subject_auc_mean": r["subject_auc_mean"], "hc_auc": r["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": r["health_groups"]["DEP"]["pooled"]["auc"]})
        ps = runs["spectral_control"]; pt = runs["temporal_control"]
        # Recompute fixed fusion from fold-isolated models, not from public refit predictions.
        fs, ft = np.full(len(y), np.nan), np.full(len(y), np.nan); fps, fpt = [], []
        for fold in range(5):
            train, valid = folds != fold, folds == fold; ms = make_model(config, 2026 + fold); mt = make_model(config, 2026 + fold); ms.fit(sx[train], y[train]); mt.fit(tx[train], y[train]); fs[valid] = ms.predict_proba(sx[valid])[:, 1]; ft[valid] = mt.predict_proba(tx[valid])[:, 1]; fps.append(ms.predict_proba(sp)[:, 1]); fpt.append(mt.predict_proba(tp)[:, 1])
        fusion_oof = (fs + ft) / 2; ms = make_model(config, 2026); mt = make_model(config, 2026); ms.fit(sx, y); mt.fit(tx, y); fusion_public = (ms.predict_proba(sp)[:, 1] + mt.predict_proba(tp)[:, 1]) / 2
        fo = ref.copy(); fo["prob_positive"] = fusion_oof; fo.to_csv(OUT / "fixed_05_05_oof.csv", index=False); pub = pd.DataFrame({"user_id": spectral["public_user_ids"], "trial_id": spectral["public_trial_ids"], "prob. of positive emotion": fusion_public}); validate_submission(pub, [f"P_test{i}" for i in range(1, 11)]); pub.to_csv(OUT / "fixed_05_05_public.csv", index=False); pd.DataFrame((np.asarray(fps) + np.asarray(fpt)).T / 2, columns=[f"fold_{i}" for i in range(5)]).assign(user_id=pub.user_id, trial_id=pub.trial_id).to_csv(OUT / "fixed_05_05_public_fold_probabilities.csv", index=False)
        runs["fixed_05_05"] = {"n_features": 600, "result": summarize(y, fusion_oof, ref, folds), "public_refit": True}; r = runs["fixed_05_05"]["result"]; summary.append({"run_name": "fixed_05_05", "n_features": 600, **r["overall"], "subject_auc_mean": r["subject_auc_mean"], "hc_auc": r["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": r["health_groups"]["DEP"]["pooled"]["auc"]})
    result = {"experiment_config": config, "provenance": {"spectral_cache_sha256": sha256((HERE / config["spectral_cache"]).resolve()), "temporal_cache_sha256": sha256((HERE / config["temporal_cache"]).resolve()), "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}, "protocol": {"cache_alignment_checked": True, "excluded_subject": "HC43", "subjectwise_unlabeled_zscore": True, "block_statistics_unlabeled": True, "train_block_size": 5, "public_block_size": 2, "no_trial_id_mapping": True, "subject_isolation": True, "fold_training_seed": "2026+fold", "threshold": 0.5}, "runs": runs, "elapsed_seconds": time.perf_counter() - started}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    base = summary[0]["auc"]; report = "# 实验036：无标签同批次窗口一致性特征\n\n复用001 spectral 150维、009 temporal 450维缓存与002固定5个被试级折，排除HC43。每行先按被试无标签z-score；新增特征为原始150维加同一数据集被试按行顺序连续块的均值、标准差、当前行减块均值（训练每5行，公开每2行），共600维。公开8个窗口没有被伪装为5段trial，也没有用训练trial_id映射公开块；该特征仅是跨数据集结构假设下的无标签块统计，不等同真实trial标签。\n\n## 结果\n\n|方法|维度|AUC|F1|BAcc|subject宏AUC|HC AUC|DEP AUC|\n|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"|{r['run_name']}|{r['n_features']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['balanced_accuracy']:.6f}|{r['subject_auc_mean']:.6f}|{r['hc_auc']:.6f}|{r['dep_auc']:.6f}|" for r in summary) + f"\n\n一致性主方法相对spectral_control的AUC差为{summary[2]['auc'] - base:+.6f}。若低于对照，不将其解释为有效创新；块大小假设、行顺序依赖及公开块不是真实trial是主要限制。\n"
    report += "\n## 每折诊断\n\n" + "\n".join(f"- {name}: " + "; ".join(f"fold{row['fold']} AUC={row['auc']:.6f}, F1={row['f1']:.6f}, BAcc={row['balanced_accuracy']:.6f}" for row in runs[name]["result"]["folds"]) for name in runs)
    (HERE / "实验报告.md").write_text(report, encoding="utf-8"); print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__": main()
