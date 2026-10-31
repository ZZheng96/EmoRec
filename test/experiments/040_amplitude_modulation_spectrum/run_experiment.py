import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import trapezoid
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.data import load_public_test_segments, load_train_segments
from emo_rec.features import FREQUENCY_BANDS
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
CACHE = HERE / "cache" / "features.npz"
PRED = HERE / "predictions"

EEG_BANDS = tuple(tuple(v) for v in CONFIG["eeg_bands"].values())
MOD_BANDS = tuple(tuple(v) for v in CONFIG["modulation_bands"].values())
FS = CONFIG["sampling_rate"]


def _integrated_power(values, fs, bands):
    frequencies, psd = welch(values, fs=fs, window="hann", nperseg=min(500, values.shape[-1]),
                              noverlap=min(500, values.shape[-1]) // 2, detrend="constant",
                              scaling="density", axis=-1)
    outputs = []
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        outputs.append(trapezoid(psd[..., mask], frequencies[mask], axis=-1))
    return np.stack(outputs, axis=-1)


def extract_spectral_control(segments, sampling_rate=FS):
    x = np.asarray(segments, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (30, 2500):
        raise ValueError("expected (n, 30, 2500) 10-second EEG windows")
    return np.log(np.maximum(_integrated_power(x, sampling_rate, EEG_BANDS), 1e-20)).reshape(len(x), 150)


def extract_modulation_features(segments, sampling_rate=FS):
    x = np.asarray(segments, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (30, 2500):
        raise ValueError("expected (n, 30, 2500) 10-second EEG windows")
    outputs = []
    for low, high in EEG_BANDS:
        sos = butter(CONFIG["eeg_filter_order"], [low, high], btype="bandpass", fs=sampling_rate,
                     output="sos")
        band = sosfiltfilt(sos, x, axis=-1)
        envelope_squared = np.abs(hilbert(band, axis=-1)) ** 2
        power = _integrated_power(envelope_squared, sampling_rate, MOD_BANDS)
        outputs.append(np.log1p(np.maximum(power, 0.0)))
    return np.concatenate(outputs, axis=-1).reshape(len(x), 450)


def build_cache():
    train_control, train_mod, labels, subjects, groups, trials = [], [], [], [], [], []
    names = None
    train_files = sorted((ROOT / "数据" / "训练集").glob("*/*.mat"))
    for path in train_files:
        segments, y, trial = load_train_segments(path)
        train_control.append(extract_spectral_control(segments))
        train_mod.append(extract_modulation_features(segments))
        labels.append(y); subjects.extend([path.stem] * len(y))
        groups.extend(["DEP" if path.parent.name == "抑郁症患者" else "HC"] * len(y)); trials.append(trial)
        print(f"features {path.stem}", flush=True)
    public_control, public_mod, users, public_trials = [], [], [], []
    test_files = sorted((ROOT / "数据" / "公开测试集").glob("*.mat"), key=lambda p: int(p.stem.replace("P_test", "")))
    for path in test_files:
        segments = load_public_test_segments(path)
        public_control.append(extract_spectral_control(segments)); public_mod.append(extract_modulation_features(segments))
        users.extend([path.stem] * len(segments)); public_trials.extend(range(1, 9))
    feature_names = np.asarray([f"modulation_{band}_{channel}_{mod}" for band in CONFIG["eeg_bands"] for channel in range(30) for mod in CONFIG["modulation_bands"]])
    np.savez_compressed(CACHE, X_control=np.concatenate(train_control), X_modulation=np.concatenate(train_mod),
                        X_public_control=np.concatenate(public_control), X_public_modulation=np.concatenate(public_mod),
                        y_train=np.concatenate(labels), subject_ids=np.asarray(subjects), health_groups=np.asarray(groups),
                        original_trial_ids=np.concatenate(trials), public_user_ids=np.asarray(users), public_trial_ids=np.asarray(public_trials),
                        feature_names=feature_names)


def load_cache():
    with np.load(CACHE, allow_pickle=False) as d: return {k: d[k] for k in d.files}


def subject_folds(subjects, groups, seed):
    unique = np.unique(subjects); labels = np.asarray([groups[np.flatnonzero(subjects == s)[0]] for s in unique])
    split = StratifiedKFold(5, shuffle=True, random_state=seed); assignment = {}
    for fold, (_, valid) in enumerate(split.split(unique, labels)):
        for s in unique[valid]: assignment[s] = fold
    return np.asarray([assignment[s] for s in subjects], dtype=int)


def assert_subject_isolation(subjects, folds):
    for f in range(5):
        assert not (set(subjects[folds == f]) & set(subjects[folds != f]))


def model(seed): return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**CONFIG["hgb"], random_state=seed))


def metrics(y, p):
    pred = p >= .5; tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": .5, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def summarize(y, p, frame, folds):
    subjects = []
    for s in frame.subject_id.unique():
        m = frame.subject_id.to_numpy() == s; subjects.append({"subject_id": s, "health_group": frame.loc[m, "health_group"].iloc[0], **metrics(y[m], p[m])})
    groups = {}
    for g in ("HC", "DEP"):
        m = frame.health_group.to_numpy() == g; rows = [r for r in subjects if r["health_group"] == g]
        groups[g] = {"n_subjects": len(rows), "pooled": metrics(y[m], p[m]), "subject_auc_mean": float(np.mean([r["auc"] for r in rows]))}
    return {"overall": metrics(y, p), "folds": [{"fold": f, "n_subjects": int(frame.loc[folds == f, "subject_id"].nunique()), **metrics(y[folds == f], p[folds == f])} for f in range(5)], "subject_auc_mean": float(np.mean([r["auc"] for r in subjects])), "health_groups": groups, "subjects": subjects}


def run_model(name, X, XP, y, frame, folds, seed):
    p = np.full(len(y), np.nan); public_folds = []
    for f in range(5):
        train, valid = folds != f, folds == f; fitted = model(seed + f); fitted.fit(X[train], y[train]); p[valid] = fitted.predict_proba(X[valid])[:, 1]; public_folds.append(fitted.predict_proba(XP)[:, 1])
    if not np.isfinite(p).all() or not np.all((p >= 0) & (p <= 1)): raise RuntimeError("invalid OOF probabilities")
    out = frame.copy(); out["prob_positive"] = p; out.to_csv(PRED / f"{name}_oof.csv", index=False)
    fitted = model(seed); fitted.fit(X, y); public = fitted.predict_proba(XP)[:, 1]
    submission = pd.DataFrame({"user_id": frame.attrs["public_users"], "trial_id": frame.attrs["public_trials"], "prob. of positive emotion": public})
    validate_submission(submission, [f"P_test{i}" for i in range(1, 11)]); submission.to_csv(PRED / f"{name}_public.csv", index=False)
    pd.DataFrame({"user_id": frame.attrs["public_users"], "trial_id": frame.attrs["public_trials"], **{f"fold_{i}_prob_positive": v for i, v in enumerate(public_folds)}}).to_csv(PRED / f"{name}_public_folds.csv", index=False)
    return summarize(y, p, frame, folds)


def run_fixed_fusion(control, modulation, public_control, public_modulation, y, frame, folds, seed):
    pc = np.full(len(y), np.nan); pm = np.full(len(y), np.nan); public_c = []; public_m = []
    for f in range(5):
        train, valid = folds != f, folds == f
        mc, mm = model(seed + f), model(seed + f)
        mc.fit(control[train], y[train]); mm.fit(modulation[train], y[train])
        pc[valid] = mc.predict_proba(control[valid])[:, 1]; pm[valid] = mm.predict_proba(modulation[valid])[:, 1]
        public_c.append(mc.predict_proba(public_control)[:, 1]); public_m.append(mm.predict_proba(public_modulation)[:, 1])
    p = .5 * pc + .5 * pm
    out = frame.copy(); out["prob_positive"] = p; out.to_csv(PRED / "fixed_05_05_oof.csv", index=False)
    mc, mm = model(seed), model(seed); mc.fit(control, y); mm.fit(modulation, y)
    public = pd.DataFrame({"user_id": frame.attrs["public_users"], "trial_id": frame.attrs["public_trials"], "prob. of positive emotion": .5 * mc.predict_proba(public_control)[:, 1] + .5 * mm.predict_proba(public_modulation)[:, 1]})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)]); public.to_csv(PRED / "fixed_05_05_public.csv", index=False)
    return summarize(y, p, frame, folds)


def main():
    PRED.mkdir(exist_ok=True)
    if not CACHE.exists(): build_cache()
    d = load_cache(); keep = d["subject_ids"] != CONFIG["excluded_subject"]
    if keep.sum() != 2760 or len(np.unique(d["subject_ids"][keep])) != 69: raise RuntimeError("filtered data shape mismatch")
    ref = pd.read_csv(HERE.parent / "002_subject_normalization" / "predictions" / "subject_zscore_exclude_hc43_oof.csv")
    subjects, groups, y = d["subject_ids"][keep], d["health_groups"][keep], d["y_train"][keep]
    if not np.array_equal(ref.subject_id.to_numpy(), subjects): raise RuntimeError("reference alignment mismatch")
    folds = ref.fold.to_numpy(int); assert_subject_isolation(subjects, folds)
    frame = ref.copy(); frame.attrs["public_users"] = d["public_user_ids"]; frame.attrs["public_trials"] = d["public_trial_ids"]
    control = normalize_by_subject(d["X_control"][keep], subjects, "zscore"); modulation = normalize_by_subject(d["X_modulation"][keep], subjects, "zscore")
    public_control = normalize_by_subject(d["X_public_control"], d["public_user_ids"], "zscore"); public_modulation = normalize_by_subject(d["X_public_modulation"], d["public_user_ids"], "zscore")
    runs, summary = {}, []
    with threadpool_limits(limits=1):
        for name, X, XP in (("spectral_control", control, public_control), ("modulation_spectral", modulation, public_modulation)):
            result = run_model(name, X, XP, y, frame, folds, 2032); result["n_features"] = X.shape[1]; runs[name] = result
            summary.append({"run_name": name, "n_features": X.shape[1], **result["overall"], "subject_auc_mean": result["subject_auc_mean"], "hc_auc": result["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": result["health_groups"]["DEP"]["pooled"]["auc"]})
        fusion = run_fixed_fusion(control, modulation, public_control, public_modulation, y, frame, folds, 2032); fusion["n_features"] = 600; runs["fixed_05_05"] = fusion
        summary.append({"run_name": "fixed_05_05", "n_features": 600, **fusion["overall"], "subject_auc_mean": fusion["subject_auc_mean"], "hc_auc": fusion["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": fusion["health_groups"]["DEP"]["pooled"]["auc"]})
        for seed in CONFIG["repeated_fold_seeds"]:
            rf = subject_folds(subjects, groups, seed); assert_subject_isolation(subjects, rf); pd.DataFrame({"subject_id": subjects, "health_group": groups, "fold": rf, "y_true": y}).to_csv(PRED / f"folds_seed{seed}.csv", index=False)
            for name, X, XP in (("spectral_control", control, public_control), ("modulation_spectral", modulation, public_modulation)):
                result = run_model(f"{name}_seed{seed}", X, XP, y, frame, rf, seed); result["n_features"] = X.shape[1]; result["seed"] = seed; runs[f"{name}_seed{seed}"] = result
    input_files = sorted((ROOT / "数据").glob("**/*.mat"))
    (HERE / "input_metadata.json").write_text(json.dumps({"files": [{"path": str(p.relative_to(ROOT)), "size": p.stat().st_size} for p in input_files], "cache_sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest()}, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = {"cache_sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest(), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    output = {"experiment_config": CONFIG, "provenance": provenance, "protocol": {"excluded_subject": "HC43", "subjectwise_unlabeled_zscore": True, "no_labels_or_row_neighbors_in_features": True, "fixed_002_folds": True, "repeated_oof_not_pooled": True, "filter_boundary": "sosfiltfilt default padding per independent 10-second window", "public_performance_reported": False}, "runs": runs}
    (HERE / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"); pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    repeated = {}
    for name in ("spectral_control", "modulation_spectral"):
        values = [runs[f"{name}_seed{s}"]["overall"] for s in CONFIG["repeated_fold_seeds"]]
        repeated[name] = {metric: {"mean": float(np.mean([v[metric] for v in values])), "std": float(np.std([v[metric] for v in values], ddof=1))} for metric in ("auc", "f1", "balanced_accuracy")}
    report = "# 实验040：逐10秒EEG频带包络调制谱\n\n" + "本实验从原始EEG逐独立10秒窗口计算。每通道以四阶零相位Butterworth带通提取1–4、4–8、8–13、13–30、30–45 Hz，Hilbert幅度包络平方后以Welch功率谱在0.2–0.5、0.5–1、1–2 Hz积分并log1p，得到30×5×3=450维；控制为同一窗口原始EEG五频带log功率150维。滤波使用sosfiltfilt默认边界填充，窗口间不拼接；250 Hz、2500采样点。训练与公开完全同一算法，每被试/公开user独立无标签z-score。\n\n排除HC43后使用69名被试2760训练窗口和80公开窗口；固定复用002五折，HGB沿用039/016参数，阈值0.5。固定融合是每折分别重训后的0.5/0.5概率融合；重复seed=2032、2033、2034仅分别保存control和modulation结果，不拼接OOF、不做显著性检验。\n\n## 固定折实际结果\n\n|方法|维度|AUC|F1|BAcc|被试宏AUC|HC AUC|DEP AUC|\n|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"|{r['run_name']}|{r['n_features']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['balanced_accuracy']:.6f}|{r['subject_auc_mean']:.6f}|{r['hc_auc']:.6f}|{r['dep_auc']:.6f}|" for r in summary) + "\n\n## 重复折均值±标准差\n\n" + "\n".join(f"- {name}: AUC {v['auc']['mean']:.6f} ± {v['auc']['std']:.6f}；F1 {v['f1']['mean']:.6f} ± {v['f1']['std']:.6f}；BAcc {v['balanced_accuracy']['mean']:.6f} ± {v['balanced_accuracy']['std']:.6f}" for name, v in repeated.items()) + "\n\n公开集无标签，仅保存预测，不报告公开性能。详细逐折、逐被试、重复折结果见metrics.json；缓存哈希及原始输入元数据一并保存。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8"); print(pd.DataFrame(summary).to_string(index=False), flush=True)

if __name__ == "__main__": main()
