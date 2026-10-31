import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import trapezoid
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.data import load_public_test_segments, load_train_segments
from emo_rec.features import ASYMMETRY_PAIRS, CHANNEL_NAMES, FREQUENCY_BANDS
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPECTRAL = (HERE / "../001_spectral_logpower/cache/features.npz").resolve()
REFERENCE = (HERE / "../002_subject_normalization/predictions/subject_zscore_exclude_hc43_oof.csv").resolve()
DIFF_CACHE = HERE / "cache" / "differential_features.npz"
OUT = HERE / "predictions"
META_KEYS = ("subject_ids", "health_groups", "original_trial_ids", "y_train", "public_user_ids", "public_trial_ids")


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_differential_features(segments, sampling_rate=250):
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (30, 2500) or len(segments) == 0:
        raise ValueError("Expected nonempty (n_windows, 30, 2500) EEG")
    if not np.isfinite(segments).all():
        raise ValueError("EEG must be finite")
    frequencies, psd = welch(segments, fs=sampling_rate, window="hann", nperseg=500,
                             noverlap=250, detrend="constant", scaling="density", axis=-1)
    powers = []
    for low, high in FREQUENCY_BANDS.values():
        mask = (frequencies >= low) & (frequencies < high)
        powers.append(trapezoid(psd[..., mask], frequencies[mask], axis=-1))
    log_power = np.log(np.maximum(np.stack(powers, axis=-1), 1e-20))
    indices = {name: i for i, name in enumerate(CHANNEL_NAMES)}
    features = np.stack([log_power[:, indices[left], :] - log_power[:, indices[right], :]
                         for left, right in ASYMMETRY_PAIRS], axis=1).reshape(len(segments), 60)
    if not np.isfinite(features).all():
        raise ValueError("Differential features must be finite")
    names = np.asarray([f"differential_log_bandpower_{left}_{right}_{band}"
                        for left, right in ASYMMETRY_PAIRS for band in FREQUENCY_BANDS])
    return features, names


def assert_cache_alignment(spectral, differential):
    for key in META_KEYS:
        if not np.array_equal(spectral[key], differential[key]):
            raise ValueError(f"cache metadata mismatch: {key}")
    if differential["X_train"].shape != (len(spectral["y_train"]), 60):
        raise ValueError("unexpected differential training shape")
    if differential["X_public"].shape != (len(spectral["public_user_ids"]), 60):
        raise ValueError("unexpected differential public shape")
    if not np.isfinite(differential["X_train"]).all() or not np.isfinite(differential["X_public"]).all():
        raise ValueError("non-finite differential cache")


def build_cache(spectral, config):
    train_x, labels, subjects, groups, trials = [], [], [], [], []
    public_x, users, public_trials = [], [], []
    paths = sorted((ROOT / "数据" / "训练集").glob("*/*.mat"))
    for path in paths:
        segments, y, ids = load_train_segments(path)
        x, names = extract_differential_features(segments, config["sampling_rate"])
        train_x.append(x); labels.extend(y); subjects.extend([path.stem] * len(y))
        groups.extend(["DEP" if path.parent.name == "抑郁症患者" else "HC"] * len(y)); trials.extend(ids)
    paths = sorted((ROOT / "数据" / "公开测试集").glob("*.mat"), key=lambda p: int(p.stem.replace("P_test", "")))
    for path in paths:
        x, _ = extract_differential_features(load_public_test_segments(path), config["sampling_rate"])
        public_x.append(x); users.extend([path.stem] * len(x)); public_trials.extend(range(1, len(x) + 1))
    data = dict(X_train=np.concatenate(train_x), y_train=np.asarray(labels), subject_ids=np.asarray(subjects),
                health_groups=np.asarray(groups), original_trial_ids=np.asarray(trials), X_public=np.concatenate(public_x),
                public_user_ids=np.asarray(users), public_trial_ids=np.asarray(public_trials), feature_names=names,
                metadata_json=np.asarray(json.dumps({"source": "raw EEG", "spectral_sha256": sha256(SPECTRAL),
                                                      "definition": "window-wise homologous left-minus-right log bandpower"})))
    assert_cache_alignment(spectral, data)
    DIFF_CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(DIFF_CACHE, **data)
    return data


def make_subject_folds(subject_ids, groups, seed):
    subjects = np.unique(subject_ids)
    subject_groups = np.asarray([groups[np.flatnonzero(subject_ids == s)[0]] for s in subjects])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    assignment = {}
    for fold, (_, valid) in enumerate(splitter.split(subjects, subject_groups)):
        for subject in subjects[valid]: assignment[subject] = fold
    return np.asarray([assignment[s] for s in subject_ids], dtype=int)


def assert_subject_isolation(subject_ids, folds):
    for fold in range(5):
        if set(subject_ids[folds == fold]) & set(subject_ids[folds != fold]):
            raise ValueError(f"subject leakage in fold {fold}")


def metrics(y, p, threshold=0.5):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, p >= threshold, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= threshold)), "threshold": threshold}


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
    return {"overall": metrics(y, p), "folds": fold_rows, "subject_auc_mean": float(np.mean([r["auc"] for r in subjects])),
            "health_groups": groups, "subjects": subjects}


def model(config, seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=seed))


def run_method(name, X, XP, data, folds, config, suffix=""):
    y = data["y_train"]; keep = data["subject_ids"] != config["excluded_subject"]
    y = y[keep]; X = X[keep]; subjects = data["subject_ids"][keep]; groups = data["health_groups"][keep]
    frame = pd.DataFrame({"subject_id": subjects, "health_group": groups, "original_trial_id": data["original_trial_ids"][keep], "fold": folds, "y_true": y})
    p = np.full(len(y), np.nan)
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        fitted = model(config, 2026 + fold); fitted.fit(X[train], y[train]); p[valid] = fitted.predict_proba(X[valid])[:, 1]
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)): raise ValueError("invalid OOF probability")
    frame["prob_positive"] = p; frame.to_csv(OUT / f"{name}{suffix}_oof.csv", index=False)
    fitted = model(config, 2026); fitted.fit(X, y); pp = fitted.predict_proba(XP)[:, 1]
    public = pd.DataFrame({"user_id": data["public_user_ids"], "trial_id": data["public_trial_ids"], "prob. of positive emotion": pp})
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)]); public.to_csv(OUT / f"{name}{suffix}_public.csv", index=False)
    return summarize(y, p, frame, folds), p


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--rebuild-features", action="store_true"); args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8")); OUT.mkdir(exist_ok=True)
    spectral = load_cache(SPECTRAL)
    differential = build_cache(spectral, config) if args.rebuild_features or not DIFF_CACHE.exists() else load_cache(DIFF_CACHE)
    assert_cache_alignment(spectral, differential)
    reference = pd.read_csv(REFERENCE); keep = spectral["subject_ids"] != config["excluded_subject"]
    if not np.array_equal(reference.subject_id.to_numpy(), spectral["subject_ids"][keep]): raise ValueError("reference subject alignment mismatch")
    fixed_folds = reference.fold.to_numpy(); assert_subject_isolation(reference.subject_id.to_numpy(), fixed_folds)
    sx = normalize_by_subject(spectral["X_train"][:, :150], spectral["subject_ids"], "zscore"); sp = normalize_by_subject(spectral["X_public"][:, :150], spectral["public_user_ids"], "zscore")
    dx = normalize_by_subject(differential["X_train"], differential["subject_ids"], "zscore"); dp = normalize_by_subject(differential["X_public"], differential["public_user_ids"], "zscore")
    runs = {}; summary = []
    with threadpool_limits(limits=1):
        for name, X, XP in (("spectral_control", sx, sp), ("differential_spectral", dx, dp)):
            result, _ = run_method(name, X, XP, {**spectral, "y_train": spectral["y_train"]}, fixed_folds, config)
            runs[name] = result; summary.append({"run_name": name, **result["overall"], "subject_auc_mean": result["subject_auc_mean"], "hc_auc": result["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": result["health_groups"]["DEP"]["pooled"]["auc"]})
        fusion_x = (sx[keep], dx[keep]); fusion_p = (sp, dp); y = spectral["y_train"][keep]
        p = np.full(len(y), np.nan)
        for fold in range(5):
            train, valid = fixed_folds != fold, fixed_folds == fold
            ms, md = model(config, 2026 + fold), model(config, 2026 + fold); ms.fit(fusion_x[0][train], y[train]); md.fit(fusion_x[1][train], y[train])
            p[valid] = .5 * ms.predict_proba(fusion_x[0][valid])[:, 1] + .5 * md.predict_proba(fusion_x[1][valid])[:, 1]
        ref = reference.copy(); ref["prob_positive"] = p; ref.to_csv(OUT / "fixed_05_05_oof.csv", index=False)
        ms, md = model(config, 2026), model(config, 2026); ms.fit(fusion_x[0], y); md.fit(fusion_x[1], y)
        public = .5 * ms.predict_proba(fusion_p[0])[:, 1] + .5 * md.predict_proba(fusion_p[1])[:, 1]
        pub = pd.DataFrame({"user_id": spectral["public_user_ids"], "trial_id": spectral["public_trial_ids"], "prob. of positive emotion": public}); validate_submission(pub, [f"P_test{i}" for i in range(1, 11)]); pub.to_csv(OUT / "fixed_05_05_public.csv", index=False)
        fusion_result = summarize(y, p, ref, fixed_folds); runs["fixed_05_05"] = fusion_result; summary.append({"run_name": "fixed_05_05", **fusion_result["overall"], "subject_auc_mean": fusion_result["subject_auc_mean"], "hc_auc": fusion_result["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": fusion_result["health_groups"]["DEP"]["pooled"]["auc"]})
        for seed in config["repeated_fold_seeds"]:
            folds = make_subject_folds(spectral["subject_ids"][keep], spectral["health_groups"][keep], seed); assert_subject_isolation(spectral["subject_ids"][keep], folds)
            for name, X, XP in (("spectral_control", sx, sp), ("differential_spectral", dx, dp)):
                result, _ = run_method(name, X, XP, {**spectral, "y_train": spectral["y_train"]}, folds, config, f"_repeat{seed}")
                runs[f"{name}_repeat{seed}"] = result
    output = {"config": config, "runs": runs, "provenance": {"spectral_cache_sha256": sha256(SPECTRAL), "differential_cache_sha256": sha256(DIFF_CACHE), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}, "protocol": {"raw_window_features": True, "metadata_alignment_checked": True, "train_windows_per_subject": 40, "public_windows_per_user": 8, "excluded_subject": "HC43", "fixed_threshold": 0.5, "no_labels_in_features": True, "repeated_folds_not_pooled": True}}
    (HERE / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"); pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__": main()
