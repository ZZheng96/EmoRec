import hashlib
import json
import platform
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
OOF_COLUMNS = ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"]


def load_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model(config, leaf, seed):
    params = dict(config["hgb"], max_leaf_nodes=int(leaf), random_state=seed)
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**params))


def lr_model(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed))


def metrics(y, p):
    pred = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": 0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def inner_oof(X, y, subjects, leaf, seed, folds=3):
    unique = np.unique(subjects)
    labels = np.asarray([y[np.flatnonzero(subjects == s)[0]] for s in unique])
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    p = np.full(len(y), np.nan); assignment = np.full(len(y), -1, dtype=int)
    for inner, (tr_s, va_s) in enumerate(splitter.split(unique, labels)):
        tr = np.isin(subjects, unique[tr_s]); va = np.isin(subjects, unique[va_s])
        fitted = model(CONFIG, leaf, seed); fitted.fit(X[tr], y[tr]); p[va] = fitted.predict_proba(X[va])[:, 1]
        assignment[va] = inner
    if not np.isfinite(p).all() or np.any(assignment < 0):
        raise RuntimeError("incomplete subject-isolated inner OOF")
    return p, assignment


def choose_leaf(X, y, subjects, seed):
    rows = []
    for leaf in CONFIG["hgb_candidates"]:
        p, assignment = inner_oof(X, y, subjects, leaf, seed, CONFIG["inner_folds"])
        m = metrics(y, p)
        rows.append({"leaf": int(leaf), "auc": m["auc"], "f1": m["f1"], "threshold": 0.5, "inner_assignment": assignment, "prob": p})
    chosen = max(rows, key=lambda r: (r["auc"], r["f1"], -abs(r["leaf"] - CONFIG["control_leaf"])))
    return chosen, rows


def fit_platt(p, y):
    eps = np.finfo(float).eps
    logits = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=2026)
    calibrator.fit(logits, y)
    return calibrator


def apply_platt(calibrator, p):
    eps = np.finfo(float).eps
    logits = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def write_oof(name, ref, p):
    out = ref.copy(); out["prob_positive"] = p; out.to_csv(OUT / f"{name}_oof.csv", index=False)


def summarize(y, p, ref, folds, selected_leaf=None, calibration_changed_auc=None):
    subject_rows = []
    for subject in np.unique(ref.subject_id):
        mask = ref.subject_id.to_numpy() == subject
        subject_rows.append({"subject_id": subject, "health_group": ref.loc[mask, "health_group"].iloc[0], **metrics(y[mask], p[mask])})
    groups = {}
    for group in ("HC", "DEP"):
        mask = ref.health_group.to_numpy() == group
        rows = [r for r in subject_rows if r["health_group"] == group]
        groups[group] = {"n_subjects": len(rows), "pooled": metrics(y[mask], p[mask]), "subject_auc_mean": float(np.mean([r["auc"] for r in rows]))}
    return {"overall": metrics(y, p), "folds": folds, "subject_auc_mean": float(np.mean([r["auc"] for r in subject_rows])),
            "health_groups": groups, "subjects": subject_rows, "selected_leaf": selected_leaf,
            "calibration_changed_auc": calibration_changed_auc}


def public_frame(ids, trials, p):
    frame = pd.DataFrame({"user_id": ids, "trial_id": trials, "prob. of positive emotion": p})
    validate_submission(frame, [f"P_test{i}" for i in range(1, 11)])
    return frame


def main():
    global CONFIG
    CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    spectral = load_cache((HERE / CONFIG["spectral_cache"]).resolve())
    temporal = load_cache((HERE / CONFIG["temporal_cache"]).resolve())
    for key in ("subject_ids", "health_groups", "original_trial_ids", "y_train", "public_user_ids", "public_trial_ids"):
        if not np.array_equal(spectral[key], temporal[key]): raise RuntimeError(f"cache alignment mismatch: {key}")
    keep = spectral["subject_ids"] != CONFIG["excluded_subject"]
    subjects = spectral["subject_ids"][keep]; groups = spectral["health_groups"][keep]; y = spectral["y_train"][keep]
    ref = pd.DataFrame({"subject_id": subjects, "health_group": groups, "original_trial_id": spectral["original_trial_ids"][keep], "fold": -1, "y_true": y})
    outer_path = (HERE / CONFIG["outer_reference"]).resolve(); outer = pd.read_csv(outer_path)
    if list(outer.columns) != OOF_COLUMNS: raise RuntimeError("unexpected outer reference columns")
    for key in ("subject_id", "health_group", "original_trial_id", "y_true"):
        if not np.array_equal(outer[key].to_numpy(), ref[key].to_numpy()): raise RuntimeError(f"outer metadata mismatch: {key}")
    folds = outer.fold.to_numpy(dtype=int); ref["fold"] = folds
    if len(y) != 2760 or len(np.unique(subjects)) != 69: raise RuntimeError("unexpected filtered data")
    ref.attrs["public_user_ids"] = spectral["public_user_ids"]; ref.attrs["public_trial_ids"] = spectral["public_trial_ids"]
    sx = normalize_by_subject(spectral["X_train"][keep, :150], subjects, "zscore")
    tx = normalize_by_subject(temporal["X_train"][keep], subjects, "zscore")
    sp = normalize_by_subject(spectral["X_public"][:, :150], spectral["public_user_ids"], "zscore")
    tp = normalize_by_subject(temporal["X_public"], temporal["public_user_ids"], "zscore")
    names = ["spectral_control", "temporal_control", "spectral_selected", "temporal_selected", "spectral_lr", "temporal_lr", "raw_fusion", "platt_fusion"]
    probs = {name: np.full(len(y), np.nan) for name in names}; selections = []; inner_rows = []
    with threadpool_limits(limits=1):
        for fold in range(CONFIG["outer_folds"]):
            tr, va = folds != fold, folds == fold; seed = 2026 + fold
            ss, sr = choose_leaf(sx[tr], y[tr], subjects[tr], seed); ts, trr = choose_leaf(tx[tr], y[tr], subjects[tr], seed)
            selections.append({"outer_fold": fold, "inner_seed": seed, "spectral_selected_leaf": ss["leaf"], "temporal_selected_leaf": ts["leaf"],
                               "spectral_control_auc": next(r["auc"] for r in sr if r["leaf"] == CONFIG["control_leaf"]), "temporal_control_auc": next(r["auc"] for r in trr if r["leaf"] == CONFIG["control_leaf"]),
                               "spectral_selected_auc": ss["auc"], "temporal_selected_auc": ts["auc"], "spectral_selected_threshold": 0.5, "temporal_selected_threshold": 0.5})
            for label, X, result, selected in (("spectral", sx, sr, ss), ("temporal", tx, trr, ts)):
                for row in result:
                    for i in np.flatnonzero(tr):
                        inner_rows.append({"outer_fold": fold, "inner_fold": int(row["inner_assignment"][np.flatnonzero(tr)[0] if False else np.flatnonzero(tr).tolist().index(i)]), "subject_id": subjects[i], "health_group": groups[i], "y_true": int(y[i]), "branch": label, "leaf": int(row["leaf"]), "probability": float(row["prob"][np.flatnonzero(tr).tolist().index(i)]), "is_selected": bool(row is selected)})
            models = {}
            for branch, X, selected in (("spectral", sx, ss), ("temporal", tx, ts)):
                control = model(CONFIG, CONFIG["control_leaf"], seed); control.fit(X[tr], y[tr])
                chosen = model(CONFIG, selected["leaf"], seed); chosen.fit(X[tr], y[tr])
                probs[f"{branch}_control"][va] = control.predict_proba(X[va])[:, 1]
                probs[f"{branch}_selected"][va] = chosen.predict_proba(X[va])[:, 1]
                lr = lr_model(seed); lr.fit(X[tr], y[tr]); probs[f"{branch}_lr"][va] = lr.predict_proba(X[va])[:, 1]
                models[branch] = chosen
            probs["raw_fusion"][va] = .5 * probs["spectral_selected"][va] + .5 * probs["temporal_selected"][va]
            cal_s, cal_t = fit_platt(ss["prob"], y[tr]), fit_platt(ts["prob"], y[tr])
            cs, ct = apply_platt(cal_s, probs["spectral_selected"][va]), apply_platt(cal_t, probs["temporal_selected"][va])
            probs["platt_fusion"][va] = .5 * cs + .5 * ct
            selections[-1].update({"raw_fusion_auc": metrics(y[tr], .5 * ss["prob"] + .5 * ts["prob"])["auc"], "platt_inner_auc": metrics(y[tr], .5 * apply_platt(cal_s, ss["prob"]) + .5 * apply_platt(cal_t, ts["prob"]))["auc"]})
            print(f"outer fold {fold + 1}/5: spectral leaf={ss['leaf']}, temporal leaf={ts['leaf']}", flush=True)
    if any(not np.isfinite(p).all() for p in probs.values()): raise RuntimeError("incomplete outer OOF")
    runs = {}; summary = []
    leaf_map = {"spectral_control": [r["spectral_control_auc"] for r in selections], "temporal_control": [r["temporal_control_auc"] for r in selections], "spectral_selected": [r["spectral_selected_leaf"] for r in selections], "temporal_selected": [r["temporal_selected_leaf"] for r in selections]}
    for name, p in probs.items():
        write_oof(name, ref, p)
        fold_metrics = [{"fold": f, "n_subjects": int(np.unique(subjects[folds == f]).size), **metrics(y[folds == f], p[folds == f])} for f in range(5)]
        changed = None if name != "platt_fusion" else bool(not np.isclose(metrics(y, probs["raw_fusion"])["auc"], metrics(y, p)["auc"]))
        runs[name] = summarize(y, p, ref, fold_metrics, leaf_map.get(name), changed)
        summary.append({"run_name": name, "auc": runs[name]["overall"]["auc"], "f1": runs[name]["overall"]["f1"], "balanced_accuracy": runs[name]["overall"]["balanced_accuracy"], "subject_auc_mean": runs[name]["subject_auc_mean"], "hc_auc": runs[name]["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": runs[name]["health_groups"]["DEP"]["pooled"]["auc"]})
    # Public predictions use one full-training fit and inner-only selection/calibration.
    ss, _ = choose_leaf(sx, y, subjects, 2026); ts, _ = choose_leaf(tx, y, subjects, 2026)
    public = {}
    for branch, X, XP, selected in (("spectral", sx, sp, ss), ("temporal", tx, tp, ts)):
        for suffix, fitted in (("control", model(CONFIG, CONFIG["control_leaf"], 2026)), ("selected", model(CONFIG, selected["leaf"], 2026)), ("lr", lr_model(2026))):
            fitted.fit(X, y); public[f"{branch}_{suffix}"] = fitted.predict_proba(XP)[:, 1]
    public["raw_fusion"] = .5 * public["spectral_selected"] + .5 * public["temporal_selected"]
    public["platt_fusion"] = .5 * apply_platt(fit_platt(ss["prob"], y), public["spectral_selected"]) + .5 * apply_platt(fit_platt(ts["prob"], y), public["temporal_selected"])
    for name, p in public.items(): public_frame(spectral["public_user_ids"], spectral["public_trial_ids"], p).to_csv(OUT / f"{name}_public.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "leaf_selection.csv", index=False); pd.DataFrame(inner_rows).to_csv(OUT / "inner_oof.csv", index=False)
    result = {"experiment_config": CONFIG, "provenance": {"spectral_cache_sha256": sha256((HERE / CONFIG["spectral_cache"]).resolve()), "temporal_cache_sha256": sha256((HERE / CONFIG["temporal_cache"]).resolve()), "outer_reference_sha256": sha256(outer_path), "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "threads": 1}, "protocol": {"subject_isolation": True, "outer_folds_reused_from_002": True, "inner_folds": 3, "platt_fit_only_inner_oof": True, "fusion_weight": 0.5, "threshold": 0.5, "outer_validation_never_used_for_selection": True}, "runs": runs}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    report = "# 实验031：嵌套分类器容量选择与Platt校准\n\n复用001 spectral 150维与009 temporal 450维，排除HC43，复用002固定5个外层被试折；两分支均按被试无标签zscore，单线程。每个外层训练集内使用seed=2026+fold的3折被试级StratifiedKFold产生inner OOF，仅以AUC、F1、接近control leaf=15的顺序选择leaf 7/15/31；另以LR(C=1)作低容量参照。Platt sigmoid为每个分支单独在inner OOF概率logit上拟合，外层验证只用于最终报告。融合权重固定0.5，主F1阈值固定0.5；inner选择阈值仅记录。\n\n## 结果\n\n|方法|AUC|F1|BAcc|subject宏AUC|HC AUC|DEP AUC|\n|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"|{r['run_name']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['balanced_accuracy']:.6f}|{r['subject_auc_mean']:.6f}|{r['hc_auc']:.6f}|{r['dep_auc']:.6f}|" for r in summary) + "\n\n每折选择、inner OOF、逐折/逐被试指标和校准AUC变化见metrics.json与predictions。固定外层折上的模型比较存在选择偏差，不能声称无偏；外层验证标签没有参与leaf、校准器或权重选择。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
