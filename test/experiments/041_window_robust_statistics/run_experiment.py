import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.data import load_public_test_segments, load_train_segments
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
CACHE = HERE / "cache" / "features.npz"
SPECTRAL = (HERE / "../001_spectral_logpower/cache/features.npz").resolve()
REFERENCE = (HERE / "../002_subject_normalization/predictions/subject_zscore_exclude_hc43_oof.csv").resolve()
PRED = HERE / "predictions"
BANDS = tuple(tuple(v) for v in CONFIG["eeg_bands"].values())
FS = CONFIG["sampling_rate"]


def robust_window_features(segments, sampling_rate=FS):
    x = np.asarray(segments, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (30, 2500) or not np.isfinite(x).all():
        raise ValueError("expected finite (n, 30, 2500) independent EEG windows")
    outputs = []
    for low, high in BANDS:
        sos = butter(CONFIG["eeg_filter_order"], [low, high], btype="bandpass", fs=sampling_rate, output="sos")
        band = sosfiltfilt(sos, x, axis=-1)
        median_abs = np.median(np.abs(band), axis=-1)
        iqr = np.percentile(band, 75, axis=-1) - np.percentile(band, 25, axis=-1)
        outputs.extend((median_abs, iqr))
    return np.concatenate(outputs, axis=1).reshape(len(x), 300)


def extract_spectral_control(segments):
    raise ValueError("control is reused from the 001 cache; raw extraction is intentionally not duplicated")


def load_cache():
    with np.load(CACHE, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def build_cache():
    train_r, labels, subjects, groups, trials = [], [], [], [], []
    for path in sorted((ROOT / "数据" / "训练集").glob("*/*.mat")):
        segments, y, trial = load_train_segments(path)
        train_r.append(robust_window_features(segments)); labels.append(y)
        subjects.extend([path.stem] * len(y)); groups.extend(["DEP" if path.parent.name == "抑郁症患者" else "HC"] * len(y)); trials.append(trial)
        print(f"features {path.stem}", flush=True)
    public_r, users, public_trials = [], [], []
    files = sorted((ROOT / "数据" / "公开测试集").glob("*.mat"), key=lambda p: int(p.stem.replace("P_test", "")))
    for path in files:
        seg = load_public_test_segments(path); public_r.append(robust_window_features(seg)); users.extend([path.stem] * len(seg)); public_trials.extend(range(1, 9))
    with np.load(SPECTRAL, allow_pickle=False) as d:
        control_train, control_public = d["X_train"][:, :150], d["X_public"][:, :150]
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, X_control=control_train, X_robust=np.concatenate(train_r), X_public_control=control_public, X_public_robust=np.concatenate(public_r), y_train=np.concatenate(labels), subject_ids=np.asarray(subjects), health_groups=np.asarray(groups), original_trial_ids=np.concatenate(trials), public_user_ids=np.asarray(users), public_trial_ids=np.asarray(public_trials), feature_names=np.asarray([f"{stat}_{band}_{channel}" for band in CONFIG["eeg_bands"] for stat in ("median_abs", "iqr") for channel in range(30)]))


def subject_folds(subjects, groups, seed):
    unique = np.unique(subjects); gl = np.asarray([groups[np.flatnonzero(subjects == s)[0]] for s in unique]); out = {}
    for fold, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(unique, gl)):
        for s in unique[valid]: out[s] = fold
    return np.asarray([out[s] for s in subjects], dtype=int)


def assert_subject_isolation(subjects, folds):
    for f in range(5):
        assert not (set(subjects[folds == f]) & set(subjects[folds != f]))


def model(seed): return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**CONFIG["hgb"], random_state=seed))


def metrics(y, p):
    pred = p >= .5; tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": .5, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def summarize(y, p, frame, folds):
    subject_rows = []
    for s in frame.subject_id.unique():
        m = frame.subject_id.to_numpy() == s; subject_rows.append({"subject_id": s, "health_group": frame.loc[m, "health_group"].iloc[0], **metrics(y[m], p[m])})
    groups = {}
    for g in ("HC", "DEP"):
        m = frame.health_group.to_numpy() == g; rows = [r for r in subject_rows if r["health_group"] == g]
        groups[g] = {"n_subjects": len(rows), "pooled": metrics(y[m], p[m]), "subject_auc_mean": float(np.mean([r["auc"] for r in rows]))}
    return {"overall": metrics(y, p), "folds": [{"fold": f, "n_subjects": int(frame.loc[folds == f, "subject_id"].nunique()), **metrics(y[folds == f], p[folds == f])} for f in range(5)], "subject_auc_mean": float(np.mean([r["auc"] for r in subject_rows])), "health_groups": groups, "subjects": subject_rows}


def run_single(name, X, XP, y, frame, folds, seed, users, trials):
    p = np.full(len(y), np.nan); pub_folds = []
    for f in range(5):
        tr, va = folds != f, folds == f; fitted = model(seed + f); fitted.fit(X[tr], y[tr]); p[va] = fitted.predict_proba(X[va])[:, 1]; pub_folds.append(fitted.predict_proba(XP)[:, 1])
    if not np.isfinite(p).all() or not np.all((p >= 0) & (p <= 1)): raise RuntimeError("invalid OOF probabilities")
    frame.assign(prob_positive=p).to_csv(PRED / f"{name}_oof.csv", index=False)
    fitted = model(seed); fitted.fit(X, y); public = fitted.predict_proba(XP)[:, 1]
    submission = pd.DataFrame({"user_id": users, "trial_id": trials, "prob. of positive emotion": public}); validate_submission(submission, [f"P_test{i}" for i in range(1, 11)]); submission.to_csv(PRED / f"{name}_public.csv", index=False)
    pd.DataFrame({"user_id": users, "trial_id": trials, **{f"fold_{i}_prob_positive": v for i, v in enumerate(pub_folds)}}).to_csv(PRED / f"{name}_public_folds.csv", index=False)
    return summarize(y, p, frame, folds)


def run_fusion(Xc, Xr, Pc, Pr, y, frame, folds, seed, users, trials):
    pc, pr = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for f in range(5):
        tr, va = folds != f, folds == f; mc, mr = model(seed + f), model(seed + f); mc.fit(Xc[tr], y[tr]); mr.fit(Xr[tr], y[tr]); pc[va] = mc.predict_proba(Xc[va])[:, 1]; pr[va] = mr.predict_proba(Xr[va])[:, 1]
    p = .5 * pc + .5 * pr; frame.assign(prob_positive=p).to_csv(PRED / "fixed_05_05_oof.csv", index=False)
    mc, mr = model(seed), model(seed); mc.fit(Xc, y); mr.fit(Xr, y); public = .5 * mc.predict_proba(Pc)[:, 1] + .5 * mr.predict_proba(Pr)[:, 1]
    submission = pd.DataFrame({"user_id": users, "trial_id": trials, "prob. of positive emotion": public}); validate_submission(submission, [f"P_test{i}" for i in range(1, 11)]); submission.to_csv(PRED / "fixed_05_05_public.csv", index=False)
    return summarize(y, p, frame, folds)


def main():
    PRED.mkdir(exist_ok=True)
    if not CACHE.exists(): build_cache()
    d = load_cache(); keep = d["subject_ids"] != CONFIG["excluded_subject"]; subjects, groups, y = d["subject_ids"][keep], d["health_groups"][keep], d["y_train"][keep]
    if len(y) != 2760 or len(np.unique(subjects)) != 69 or len(d["public_user_ids"]) != 80: raise RuntimeError("shape mismatch")
    ref = pd.read_csv(REFERENCE); folds = ref.fold.to_numpy(int)
    if not np.array_equal(ref.subject_id.to_numpy(), subjects): raise RuntimeError("reference alignment mismatch")
    assert_subject_isolation(subjects, folds)
    frame = ref.copy(); frame.attrs["public_users"] = d["public_user_ids"]; frame.attrs["public_trials"] = d["public_trial_ids"]
    Xc = normalize_by_subject(d["X_control"][keep], subjects, "zscore"); Xr = normalize_by_subject(d["X_robust"][keep], subjects, "zscore")
    Pc = normalize_by_subject(d["X_public_control"], d["public_user_ids"], "zscore"); Pr = normalize_by_subject(d["X_public_robust"], d["public_user_ids"], "zscore")
    runs, summary = {}, []
    with threadpool_limits(limits=1):
        for name, X, XP in (("spectral_control", Xc, Pc), ("robust_window", Xr, Pr)):
            r = run_single(name, X, XP, y, frame, folds, 2027, d["public_user_ids"], d["public_trial_ids"]); r["n_features"] = X.shape[1]; runs[name] = r; summary.append({"run_name": name, "n_features": X.shape[1], **r["overall"], "subject_auc_mean": r["subject_auc_mean"], "hc_auc": r["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": r["health_groups"]["DEP"]["pooled"]["auc"]})
        r = run_fusion(Xc, Xr, Pc, Pr, y, frame, folds, 2027, d["public_user_ids"], d["public_trial_ids"]); r["n_features"] = 450; runs["fixed_05_05"] = r; summary.append({"run_name": "fixed_05_05", "n_features": 450, **r["overall"], "subject_auc_mean": r["subject_auc_mean"], "hc_auc": r["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": r["health_groups"]["DEP"]["pooled"]["auc"]})
        for seed in CONFIG["repeated_fold_seeds"]:
            rf = subject_folds(subjects, groups, seed); assert_subject_isolation(subjects, rf); pd.DataFrame({"subject_id": subjects, "health_group": groups, "fold": rf, "y_true": y}).to_csv(PRED / f"folds_seed{seed}.csv", index=False)
            for name, X, XP in (("spectral_control", Xc, Pc), ("robust_window", Xr, Pr)):
                r = run_single(f"{name}_seed{seed}", X, XP, y, frame, rf, seed, d["public_user_ids"], d["public_trial_ids"]); r["n_features"] = X.shape[1]; r["seed"] = seed; runs[f"{name}_seed{seed}"] = r
    input_files = sorted((ROOT / "数据").glob("**/*.mat")); provenance = {"input_files": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size": p.stat().st_size} for p in input_files], "spectral_cache_sha256": hashlib.sha256(SPECTRAL.read_bytes()).hexdigest(), "feature_cache_sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest(), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    (HERE / "input_metadata.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    output = {"experiment_config": CONFIG, "provenance": provenance, "protocol": {"no_labels_or_metadata_in_features": True, "subjectwise_unlabeled_zscore": True, "fixed_002_folds": True, "repeated_oof_not_pooled": True, "public_performance_reported": False}, "runs": runs}
    (HERE / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"); pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    repeated = {n: {m: {"mean": float(np.mean([runs[f"{n}_seed{s}"]["overall"][m] for s in CONFIG["repeated_fold_seeds"]])), "std": float(np.std([runs[f"{n}_seed{s}"]["overall"][m] for s in CONFIG["repeated_fold_seeds"]], ddof=1))} for m in ("auc", "f1", "balanced_accuracy")} for n in CONFIG["methods_repeated"]}
    report = "# 实验041：逐10秒窗口稳健时域/频带统计\n\n从原始EEG逐个独立10秒窗口（250 Hz，2500点）以四阶Butterworth SOS带通和sosfiltfilt默认独立窗口边界提取1–4、4–8、8–13、13–30、30–45 Hz。每通道/频带计算median(abs(filtered))与IQR(filtered)，按频带、统计量、通道形成300维；spectral_control严格复用001缓存前150维150维频谱特征。训练与公开同构，每user独立无标签z-score。\n\n排除HC43后2760行/69被试，公开80行；固定复用002五折，HGB参数与阈值0.5。融合每折重训后0.5/0.5。重复seed=2027..2031分别保存，不拼接OOF、不做独立显著性检验。公开集无标签，仅保存预测。\n\n## 固定折实际结果\n\n|方法|维度|AUC|F1|BAcc|被试宏AUC|HC AUC|DEP AUC|\n|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"|{r['run_name']}|{r['n_features']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['balanced_accuracy']:.6f}|{r['subject_auc_mean']:.6f}|{r['subject_auc_mean'] if False else r['hc_auc']:.6f}|{r['dep_auc']:.6f}|" for r in summary) + "\n\n## 重复折均值±标准差\n\n" + "\n".join(f"- {n}: AUC {v['auc']['mean']:.6f} ± {v['auc']['std']:.6f}；F1 {v['f1']['mean']:.6f} ± {v['f1']['std']:.6f}；BAcc {v['balanced_accuracy']['mean']:.6f} ± {v['balanced_accuracy']['std']:.6f}" for n, v in repeated.items()) + "\n\n若robust_window低于control，按预注册规则停止该方向。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8"); print(pd.DataFrame(summary).to_string(index=False), flush=True)

if __name__ == "__main__": main()
