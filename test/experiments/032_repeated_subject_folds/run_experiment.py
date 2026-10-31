import json
import hashlib
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
OOF_COLUMNS = ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"]


def load_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_cache_alignment(spectral, temporal):
    for key in ("subject_ids", "health_groups", "original_trial_ids", "y_train", "public_user_ids", "public_trial_ids"):
        if not np.array_equal(spectral[key], temporal[key]):
            raise RuntimeError(f"cache alignment mismatch: {key}")


def make_model(config, seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=seed))


def make_subject_folds(subject_ids, health_groups, seed, n_splits=5):
    subjects = np.unique(subject_ids)
    subject_groups = np.asarray([health_groups[np.flatnonzero(subject_ids == s)[0]] for s in subjects])
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignment = {}
    for fold, (_, validation) in enumerate(splitter.split(subjects, subject_groups)):
        for subject in subjects[validation]:
            assignment[subject] = fold
    return np.asarray([assignment[s] for s in subject_ids], dtype=np.int64)


def assert_subject_isolation(subject_ids, folds):
    for fold in range(5):
        train = set(subject_ids[folds != fold]); valid = set(subject_ids[folds == fold])
        if train & valid:
            raise RuntimeError(f"subject leakage in fold {fold}")


def classification_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": 0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def summarize(y, p, reference, folds):
    subject_rows = []
    for subject in np.unique(reference.subject_id):
        mask = reference.subject_id.to_numpy() == subject
        subject_rows.append({"subject_id": subject, "health_group": reference.loc[mask, "health_group"].iloc[0], **classification_metrics(y[mask], p[mask])})
    groups = {}
    for group in ("HC", "DEP"):
        mask = reference.health_group.to_numpy() == group
        rows = [r for r in subject_rows if r["health_group"] == group]
        groups[group] = {"n_subjects": len(rows), "pooled": classification_metrics(y[mask], p[mask]),
                         "subject_auc_mean": float(np.mean([r["auc"] for r in rows])),
                         "subject_auc_std": float(np.std([r["auc"] for r in rows], ddof=1))}
    fold_rows = [{"fold": f, "n_subjects": int(reference.loc[folds == f, "subject_id"].nunique()),
                  "n_samples": int((folds == f).sum()), **classification_metrics(y[folds == f], p[folds == f])} for f in range(5)]
    return {"overall": classification_metrics(y, p), "folds": fold_rows,
            "subject_auc_mean": float(np.mean([r["auc"] for r in subject_rows])),
            "subject_auc_std": float(np.std([r["auc"] for r in subject_rows], ddof=1)),
            "health_groups": groups, "subjects": subject_rows}


def write_predictions(name, reference, p, public_ids=None, public_trials=None, public_p=None):
    output = reference.copy(); output["prob_positive"] = p
    output.to_csv(OUT / f"{name}_oof.csv", index=False)
    if public_p is not None:
        frame = pd.DataFrame({"user_id": public_ids, "trial_id": public_trials, "prob. of positive emotion": public_p})
        validate_submission(frame, [f"P_test{i}" for i in range(1, 11)])
        frame.to_csv(OUT / f"{name}_public.csv", index=False)


def evaluate_method(name, X, X_public, y, reference, folds_by_repeat, config):
    repeat_results = []; all_public = []
    for repeat_index, folds in enumerate(folds_by_repeat):
        p = np.full(len(y), np.nan)
        for fold in range(5):
            train, valid = folds != fold, folds == fold
            fitted = make_model(config, 2026 + fold)
            fitted.fit(X[train], y[train])
            p[valid] = fitted.predict_proba(X[valid])[:, 1]
        if not np.isfinite(p).all(): raise RuntimeError(f"incomplete OOF predictions: {name}")
        repeat_reference = reference.copy(); repeat_reference["fold"] = folds
        write_predictions(f"{name}_repeat{repeat_index + 1:02d}", repeat_reference, p)
        result = summarize(y, p, repeat_reference, folds); result["repeat"] = repeat_index + 1; result["seed"] = config["random_seeds"][repeat_index]
        repeat_results.append(result)
        fitted = make_model(config, 2026); fitted.fit(X, y)
        all_public.append(fitted.predict_proba(X_public)[:, 1])
    metric_keys = ("auc", "f1", "balanced_accuracy")
    aggregate = {}
    for key in metric_keys:
        vals = [r["overall"][key] for r in repeat_results]
        aggregate[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "values": vals}
    for key in ("subject_auc_mean",):
        vals = [r[key] for r in repeat_results]; aggregate[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "values": vals}
    for group in ("HC", "DEP"):
        vals = [r["health_groups"][group]["pooled"]["auc"] for r in repeat_results]
        aggregate[f"{group.lower()}_auc"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "values": vals}
    public = np.mean(np.asarray(all_public), axis=0)
    return {"n_features": int(X.shape[1]), "repeats": repeat_results, "aggregate": aggregate, "public_prediction_repeats": len(all_public)}, public


def main():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8")); OUT.mkdir(parents=True, exist_ok=True)
    spectral = load_cache((HERE / config["spectral_cache"]).resolve()); temporal = load_cache((HERE / config["temporal_cache"]).resolve())
    assert_cache_alignment(spectral, temporal)
    keep = spectral["subject_ids"] != config["excluded_subject"]
    subjects = spectral["subject_ids"][keep]; groups = spectral["health_groups"][keep]; y = spectral["y_train"][keep]
    if len(y) != config["expected_train_rows"] or len(np.unique(subjects)) != config["expected_subjects"]: raise RuntimeError("unexpected filtered data")
    reference = pd.DataFrame({"subject_id": subjects, "health_group": groups, "original_trial_id": spectral["original_trial_ids"][keep], "fold": -1, "y_true": y})
    folds_by_repeat = [make_subject_folds(subjects, groups, seed, config["folds"]) for seed in config["random_seeds"]]
    for seed, folds in zip(config["random_seeds"], folds_by_repeat):
        assert_subject_isolation(subjects, folds)
        frame = pd.DataFrame({"repeat": np.repeat(seed, len(y)), "subject_id": subjects, "health_group": groups, "original_trial_id": reference.original_trial_id, "fold": folds})
        frame.to_csv(OUT / f"folds_repeat{seed}.csv", index=False)
    sx = normalize_by_subject(spectral["X_train"][keep, :150], subjects, "zscore"); tx = normalize_by_subject(temporal["X_train"][keep], subjects, "zscore")
    sp = normalize_by_subject(spectral["X_public"][:, :150], spectral["public_user_ids"], "zscore"); tp = normalize_by_subject(temporal["X_public"], temporal["public_user_ids"], "zscore")
    runs = {}; summary = []
    with threadpool_limits(limits=1):
        for name, X, XP in (("spectral_control", sx, sp), ("temporal_control", tx, tp)):
            runs[name], public = evaluate_method(name, X, XP, y, reference, folds_by_repeat, config)
            write_predictions(name, reference.iloc[:0].copy(), np.array([]), spectral["public_user_ids"], spectral["public_trial_ids"], public) if False else pd.DataFrame({"user_id": spectral["public_user_ids"], "trial_id": spectral["public_trial_ids"], "prob. of positive emotion": public}).to_csv(OUT / f"{name}_public.csv", index=False)
            summary.append({"run_name": name, **{k: runs[name]["aggregate"][k]["mean"] for k in ("auc", "f1", "balanced_accuracy", "subject_auc_mean", "hc_auc", "dep_auc")}})
        fusion_X = (sx, tx); fusion_P = (sp, tp); fusion_results = []
        for repeat_index, folds in enumerate(folds_by_repeat):
            ps = np.full(len(y), np.nan); pt = np.full(len(y), np.nan)
            for fold in range(5):
                train, valid = folds != fold, folds == fold
                ms = make_model(config, 2026 + fold); mt = make_model(config, 2026 + fold); ms.fit(sx[train], y[train]); mt.fit(tx[train], y[train])
                ps[valid] = ms.predict_proba(sx[valid])[:, 1]; pt[valid] = mt.predict_proba(tx[valid])[:, 1]
            p = .5 * ps + .5 * pt; rr = reference.copy(); rr["fold"] = folds
            write_predictions(f"fixed_05_05_repeat{repeat_index + 1:02d}", rr, p)
            result = summarize(y, p, rr, folds); result["repeat"] = repeat_index + 1; result["seed"] = config["random_seeds"][repeat_index]; fusion_results.append(result)
        vals = lambda key: [r["overall"][key] for r in fusion_results]
        agg = {key: {"mean": float(np.mean(vals(key))), "std": float(np.std(vals(key), ddof=1)), "values": vals(key)} for key in ("auc", "f1", "balanced_accuracy")}
        agg["subject_auc_mean"] = {"mean": float(np.mean([r["subject_auc_mean"] for r in fusion_results])), "std": float(np.std([r["subject_auc_mean"] for r in fusion_results], ddof=1)), "values": [r["subject_auc_mean"] for r in fusion_results]}
        for group in ("HC", "DEP"):
            v = [r["health_groups"][group]["pooled"]["auc"] for r in fusion_results]; agg[f"{group.lower()}_auc"] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "values": v}
        mf = make_model(config, 2026); mt = make_model(config, 2026); mf.fit(sx, y); mt.fit(tx, y); public = .5 * mf.predict_proba(sp)[:, 1] + .5 * mt.predict_proba(tp)[:, 1]
        pd.DataFrame({"user_id": spectral["public_user_ids"], "trial_id": spectral["public_trial_ids"], "prob. of positive emotion": public}).to_csv(OUT / "fixed_05_05_public.csv", index=False)
        runs["fixed_05_05"] = {"n_features": 600, "repeats": fusion_results, "aggregate": agg, "public_prediction_repeats": 1}; summary.append({"run_name": "fixed_05_05", **{k: agg[k]["mean"] for k in ("auc", "f1", "balanced_accuracy", "subject_auc_mean", "hc_auc", "dep_auc")}})
    result = {"experiment_config": config, "provenance": {"spectral_cache_sha256": sha256((HERE / config["spectral_cache"]).resolve()), "temporal_cache_sha256": sha256((HERE / config["temporal_cache"]).resolve()), "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}, "protocol": {"cache_alignment_checked": True, "excluded_subject": "HC43", "subjectwise_unlabeled_zscore": True, "subject_isolation": True, "independent_repeats": config["random_seeds"], "fold_training_seed": "2026+fold", "fixed_threshold": 0.5, "fusion_weights": [0.5, 0.5]}, "runs": runs}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    report = "# 实验032：独立重复被试级分层5折复核\n\n复用001 spectral 150维和009 temporal 450维缓存，先完成逐项缓存对齐检查，再排除HC43。每个重复用seed=2027..2031的被试级StratifiedKFold；每被试无标签z-score；HGB参数固定，训练折seed=2026+fold，阈值0.5，融合权重固定0.5/0.5。每个重复的OOF、公开预测和折分配分别保存，未将跨重复预测合并伪装成单次OOF。\n\n## 结果\n\n|方法|AUC均值±标准差|F1均值±标准差|BAcc均值±标准差|subject宏AUC均值±标准差|HC AUC均值±标准差|DEP AUC均值±标准差|\n|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"|{r['run_name']}|{runs[r['run_name']]['aggregate']['auc']['mean']:.6f}±{runs[r['run_name']]['aggregate']['auc']['std']:.6f}|{runs[r['run_name']]['aggregate']['f1']['mean']:.6f}±{runs[r['run_name']]['aggregate']['f1']['std']:.6f}|{runs[r['run_name']]['aggregate']['balanced_accuracy']['mean']:.6f}±{runs[r['run_name']]['aggregate']['balanced_accuracy']['std']:.6f}|{runs[r['run_name']]['aggregate']['subject_auc_mean']['mean']:.6f}±{runs[r['run_name']]['aggregate']['subject_auc_mean']['std']:.6f}|{runs[r['run_name']]['aggregate']['hc_auc']['mean']:.6f}±{runs[r['run_name']]['aggregate']['hc_auc']['std']:.6f}|{runs[r['run_name']]['aggregate']['dep_auc']['mean']:.6f}±{runs[r['run_name']]['aggregate']['dep_auc']['std']:.6f}|" for r in summary) + "\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8"); print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__": main()
