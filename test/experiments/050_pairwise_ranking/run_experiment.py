import json
import hashlib
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def make_subject_folds(subjects, groups, seed, n_splits=5):
    unique = np.unique(subjects)
    labels = np.asarray([groups[np.flatnonzero(subjects == s)[0]] for s in unique])
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignment = {}
    for fold, (_, valid) in enumerate(splitter.split(unique, labels)):
        for subject in unique[valid]: assignment[subject] = fold
    return np.asarray([assignment[s] for s in subjects], dtype=int)


def make_model(name, config, seed):
    if name == "pairwise_ranking":
        return make_pipeline(StandardScaler(), LogisticRegression(C=config["C"], max_iter=config["max_iter"], random_state=seed))
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=config["C"], max_iter=config["max_iter"], random_state=seed))
    if name == "spectral_hgb":
        return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config, random_state=seed))
    raise ValueError(name)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def build_pairs(X, y, subjects, train_mask, seed, pairs_per_subject):
    rng = np.random.default_rng(seed)
    left, right = [], []
    for subject in np.unique(subjects[train_mask]):
        idx = np.flatnonzero(train_mask & (subjects == subject))
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        if not len(pos) or not len(neg): raise RuntimeError(f"missing class for {subject}")
        count = pairs_per_subject
        left.append(rng.choice(pos, count, replace=True)); right.append(rng.choice(neg, count, replace=True))
    pos_left, neg_right = np.concatenate(left), np.concatenate(right)
    differences = X[pos_left] - X[neg_right]
    pair_X = np.concatenate([differences, -differences])
    pair_y = np.concatenate([np.ones(len(differences), dtype=int), np.zeros(len(differences), dtype=int)])
    return pair_X, pair_y, pos_left, neg_right


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": 0.5, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def summarize(y, p, ref, folds):
    subjects = []
    for s in np.unique(ref.subject_id):
        m = ref.subject_id.to_numpy() == s
        subjects.append({"subject_id": s, "health_group": ref.loc[m, "health_group"].iloc[0], **metrics(y[m], p[m])})
    groups = {}
    for g in ("HC", "DEP"):
        m = ref.health_group.to_numpy() == g; rows = [r for r in subjects if r["health_group"] == g]
        groups[g] = {"n_subjects": len(rows), "pooled": metrics(y[m], p[m]), "subject_auc_mean": float(np.mean([r["auc"] for r in rows])), "subject_auc_std": float(np.std([r["auc"] for r in rows], ddof=1))}
    fold_rows = [{"fold": f, "n_subjects": int(ref.loc[folds == f, "subject_id"].nunique()), "n_samples": int((folds == f).sum()), **metrics(y[folds == f], p[folds == f])} for f in range(5)]
    return {"overall": metrics(y, p), "folds": fold_rows, "subject_auc_mean": float(np.mean([r["auc"] for r in subjects])), "subject_auc_std": float(np.std([r["auc"] for r in subjects], ddof=1)), "health_groups": groups, "subjects": subjects}


def evaluate(name, X, X_public, y, ref, folds_list, config):
    repeats = []; public_all = []
    for ri, folds in enumerate(folds_list, 1):
        p = np.full(len(y), np.nan)
        for fold in range(5):
            train, valid = folds != fold, folds == fold
            if name == "pairwise_ranking":
                pair_X, pair_y, _, _ = build_pairs(X, y, ref.subject_id.to_numpy(), train, config["repeat_seeds"][ri - 1] + fold, config["pair_sampling"]["pairs_per_subject"])
                model = make_model(name, config["pairwise_model"], config["repeat_seeds"][ri - 1] + fold); model.fit(pair_X, pair_y)
                p[valid] = sigmoid(model.decision_function(X[valid]))
            else:
                model = make_model(name, config["control_models"][name], config["repeat_seeds"][ri - 1] + fold); model.fit(X[train], y[train]); p[valid] = model.predict_proba(X[valid])[:, 1]
        if not np.isfinite(p).all(): raise RuntimeError("incomplete OOF")
        rr = ref.copy(); rr["fold"] = folds; rr["prob_positive"] = p
        rr.to_csv(OUT / f"{name}_repeat{ri:02d}_oof.csv", index=False)
        result = summarize(y, p, rr, folds); result.update({"repeat": ri, "seed": config["repeat_seeds"][ri - 1]}); repeats.append(result)
        if name == "pairwise_ranking":
            pair_X, pair_y, _, _ = build_pairs(X, y, ref.subject_id.to_numpy(), np.ones(len(y), bool), config["repeat_seeds"][ri - 1], config["pair_sampling"]["pairs_per_subject"])
            model = make_model(name, config["pairwise_model"], config["repeat_seeds"][ri - 1]); model.fit(pair_X, pair_y); public_all.append(sigmoid(model.decision_function(X_public)))
        else:
            model = make_model(name, config["control_models"][name], config["repeat_seeds"][ri - 1]); model.fit(X, y); public_all.append(model.predict_proba(X_public)[:, 1])
    aggregate = {}
    for key in ("auc", "f1", "balanced_accuracy"):
        v = [r["overall"][key] for r in repeats]; aggregate[key] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "values": v}
    for key in ("subject_auc_mean",):
        v = [r[key] for r in repeats]; aggregate[key] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "values": v}
    for g in ("HC", "DEP"):
        v = [r["health_groups"][g]["pooled"]["auc"] for r in repeats]; aggregate[f"{g.lower()}_auc"] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "values": v}
    return {"n_features": int(X.shape[1]), "repeats": repeats, "aggregate": aggregate}, np.mean(public_all, axis=0)


def main():
    started = time.perf_counter(); config = json.loads((HERE / "config.json").read_text(encoding="utf-8")); OUT.mkdir(exist_ok=True)
    cache_path = (HERE / config["spectral_cache"]).resolve(); data = load_cache(cache_path)
    keep = data["subject_ids"] != config["excluded_subject"]
    subjects, groups, y = data["subject_ids"][keep], data["health_groups"][keep], data["y_train"][keep]
    X = normalize_by_subject(data["X_train"][keep, :150], subjects, "zscore")
    X_public = normalize_by_subject(data["X_public"][:, :150], data["public_user_ids"], "zscore")
    ref = pd.DataFrame({"subject_id": subjects, "health_group": groups, "original_trial_id": data["original_trial_ids"][keep], "fold": -1, "y_true": y})
    folds_list = [make_subject_folds(subjects, groups, s, config["folds"]) for s in config["repeat_seeds"]]
    for seed, folds in zip(config["repeat_seeds"], folds_list): pd.DataFrame({"subject_id": subjects, "health_group": groups, "fold": folds}).to_csv(OUT / f"folds_repeat{seed}.csv", index=False)
    runs = {}; summary = []
    with threadpool_limits(limits=1):
        for name in ("pairwise_ranking", "logistic_regression", "spectral_hgb"):
            runs[name], public = evaluate(name, X, X_public, y, ref, folds_list, config)
            public_frame = pd.DataFrame({"user_id": data["public_user_ids"], "trial_id": data["public_trial_ids"], "prob. of positive emotion": public})
            validate_submission(public_frame, [f"P_test{i}" for i in range(1, 11)]); public_frame.to_csv(OUT / f"{name}_public.csv", index=False)
            summary.append({"run_name": name, **{k: runs[name]["aggregate"][k]["mean"] for k in ("auc", "f1", "balanced_accuracy", "subject_auc_mean", "hc_auc", "dep_auc")}})
    result = {"experiment_config": config, "provenance": {"cache": str(cache_path), "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(), "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "elapsed_seconds": time.perf_counter() - started}, "protocol": {"excluded_subject": "HC43", "subjectwise_unlabeled_zscore": True, "subject_isolation": True, "within_subject_pairs_only": True, "validation_public_transform_only": True, "fixed_threshold": 0.5}, "runs": runs}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    report = "# 实验050：被试内成对排序\n\n复用001的150维频谱cache，排除HC43。训练/验证/公开按被试或用户独立无标签z-score；每个外层折只在训练被试内部固定采样正负窗口对，以差分特征训练线性Logistic ranking，验证与公开仅转为窗口分数。普通LR与频谱HGB为控制，002固定五折并使用032式五次独立OOF，阈值0.5。\n\n## 结果\n\n|方法|AUC|F1|BAcc|subject宏AUC|HC AUC|DEP AUC|\n|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"|{r['run_name']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['balanced_accuracy']:.6f}|{r['subject_auc_mean']:.6f}|{r['hc_auc']:.6f}|{r['dep_auc']:.6f}|" for r in summary) + "\n\n以频谱HGB为预注册控制判断是否提升，重点比较DEP AUC；若未提升则停止，不调参。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8"); print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__": main()
