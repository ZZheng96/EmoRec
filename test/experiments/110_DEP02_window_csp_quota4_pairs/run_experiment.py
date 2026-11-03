import hashlib
import json
import platform
import time
from pathlib import Path
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
REGS = (1e-8, 1e-6, 1e-4, 1e-2)
ENDS = (1, 2, 3)
CS = (0.01, 0.1, 1, 10, 100)
CANDIDATES = tuple((model, reg, end, c) for model in ("logistic_regression", "linear_svc") for reg in REGS for end in ENDS for c in CS)


FIXED_THRESHOLD = 0.5
POSITIVE_QUOTA = 4


def quota_predictions(probabilities):
    probabilities = np.asarray(probabilities)
    assert probabilities.shape == (8,) and np.isfinite(probabilities).all()
    predictions = np.zeros(8, dtype=int)
    predictions[np.argsort(-probabilities, kind="stable")[:POSITIVE_QUOTA]] = 1
    return predictions


def frozen_hashes():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("run_experiment.py", "config.json", "冻结协议.md")}


def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, p >= FIXED_THRESHOLD, zero_division=0))}


def load_subject(subject):
    candidates = [path for path in (ROOT / "数据" / "训练集").glob("*/*.mat") if path.stem == subject]
    assert len(candidates) == 1
    path = candidates[0]
    with h5py.File(path, "r") as mat:
        pieces = []
        for key in ("EEG_data_neu", "EEG_data_pos"):
            data = np.asarray(mat[key][()], dtype=np.float64).T
            assert data.shape == (30, 50000)
            pieces.append(data.reshape(30, 20, 2500).transpose(1, 0, 2))
    x = np.concatenate(pieces)
    y = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    trial = np.r_[np.repeat(np.arange(4), 5), 4 + np.repeat(np.arange(4), 5)]
    return x, y, trial, path


def covariances(x):
    all_bands = []
    for low, high in BANDS:
        sos = butter(4, [low, high], btype="bandpass", fs=250, output="sos")
        z = sosfiltfilt(sos, x, axis=-1)
        z -= z.mean(axis=-1, keepdims=True)
        all_bands.append(z @ z.transpose(0, 2, 1) / z.shape[-1])
    return np.stack(all_bands, axis=1)


def csp_fit_transform(cov_train, y_train, cov_valid, reg, end):
    filters = []
    eye = np.eye(30)
    for band in range(len(BANDS)):
        c = cov_train[:, band] + reg * eye
        c /= np.trace(c, axis1=1, axis2=2)[:, None, None]
        c0, c1 = c[y_train == 0].mean(axis=0), c[y_train == 1].mean(axis=0)
        _, vectors = eigh(c1, c0 + c1)
        take = np.r_[np.arange(end), np.arange(30 - end, 30)]
        filters.append(vectors[:, take].T)
    filters = np.asarray(filters)

    def transform(values):
        variance = np.einsum("bkc,nbcd,bkd->nbk", filters, values, filters)
        result = np.log(np.maximum(variance, 1e-30)).reshape(len(values), -1)
        assert np.isfinite(result).all()
        return result

    return transform(cov_train), transform(cov_valid)


def predict(cov_train, y_train, cov_valid, candidate):
    model_name, reg, end, c = candidate
    train_features, valid_features = csp_fit_transform(cov_train, y_train, cov_valid, reg, end)
    estimator = LogisticRegression(C=c, solver="liblinear", max_iter=3000, random_state=2026) if model_name == "logistic_regression" else LinearSVC(C=c, dual="auto", max_iter=5000, random_state=2026)
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(train_features, y_train)
    score = model.decision_function(valid_features)
    return 1 / (1 + np.exp(-np.clip(score, -60, 60)))


def trial_pairs(trial):
    pos = np.flatnonzero(np.asarray(trial) >= 4)
    neg = np.flatnonzero(np.asarray(trial) < 4)
    for p in pos:
        for n in neg:
            va = np.asarray([p, n])
            yield np.setdiff1d(np.arange(len(trial)), va), va


def window_indices(trial, selected_trials):
    return np.flatnonzero(np.isin(trial, selected_trials))


def aggregate(values, trial):
    unique = np.asarray(sorted(set(trial)))
    return unique, np.asarray([values[np.asarray(trial) == t].mean() for t in unique])


def select(cov, y, trial):
    inner = list(trial_pairs(np.unique(trial)))
    records = []
    for index, candidate in enumerate(CANDIDATES):
        values = []
        for tr_trials, va_trials in inner:
            tr = window_indices(trial, np.unique(trial)[tr_trials])
            va = window_indices(trial, np.unique(trial)[va_trials])
            values.append((predict(cov[tr], y[tr], cov[va], candidate), va, trial[va]))
        for direction_index, direction in enumerate(("raw", "reverse")):
            p_by_trial = {}
            y_by_trial = {}
            for value, va, va_trial in values:
                value = value if direction == "raw" else 1 - value
                for t in np.unique(va_trial):
                    mask = va_trial == t
                    p_by_trial.setdefault(int(t), []).append(float(value[mask].mean()))
                    y_by_trial[int(t)] = int(y[va][mask][0])
            keys = sorted(p_by_trial)
            p = np.asarray([np.mean(p_by_trial[k]) for k in keys])
            yy = np.asarray([y_by_trial[k] for k in keys])
            records.append({"candidate_index": index, "candidate": candidate, "direction": direction, "direction_index": direction_index, **metric(yy, p)})
    best = max(records, key=lambda r: (r["auc"], r["f1"], -r["candidate_index"], -r["direction_index"]))
    return (CANDIDATES[best["candidate_index"]], best["direction"]), records


def main():
    started = time.perf_counter()
    frozen = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "hashes": frozen_hashes()}
    (HERE / "input_metadata/freeze.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf8")
    x, y, trial, source = load_subject("DEP02")
    cov = covariances(x)
    assert cov.shape == (40, 5, 30, 30)
    values, masks, rows, selections = [], [], [], []
    with threadpool_limits(limits=1):
        for pair_id, (tr_trials, va_trials) in enumerate(trial_pairs(np.arange(8))):
            tr = window_indices(trial, tr_trials)
            va = window_indices(trial, va_trials)
            selected, scores = select(cov[tr], y[tr], trial[tr])
            candidate, direction = selected
            value = predict(cov[tr], y[tr], cov[va], candidate)
            if direction == "reverse":
                value = 1 - value
            va_unique, pair_trial_values = aggregate(value, trial[va])
            values.append(pair_trial_values); masks.append(va_unique)
            selections.append({"pair_id": pair_id, "train_trial_indices": tr_trials.tolist(), "valid_trial_indices": va_trials.tolist(), "train_window_count": int(len(tr)), "valid_window_count": int(len(va)), "selected_candidate": candidate, "selected_direction": direction, "inner_scores": scores})
            for t, probability in zip(va_unique, pair_trial_values):
                rows.append({"pair_id": pair_id, "trial_index": int(t), "trial_key": f"{int(y[trial == t][0])}::{int(t % 4) + 1}", "y": int(y[trial == t][0]), "prob_positive": float(probability), "window_count": 5})
    sums = np.zeros(8); counts = np.zeros(8, dtype=int)
    for value, va in zip(values, masks):
        sums[va] += value; counts[va] += 1
    assert np.all(counts == 4)
    p = sums / counts
    keys = np.asarray([f"{int(y[trial == t][0])}::{int(t % 4) + 1}" for t in range(8)])
    out = pd.DataFrame({"trial_index": np.arange(8), "trial_key": keys, "y": np.asarray([y[trial == t][0] for t in range(8)]), "prob_positive": p, "prediction_count": counts})
    out["prediction_quota4"] = quota_predictions(p)
    out["prediction_fixed05"] = (p >= FIXED_THRESHOLD).astype(int)
    out.to_csv(HERE / "predictions/DEP02_window_csp_pairs_oof.csv", index=False)
    pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf8")
    provenance = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256((HERE / "run_experiment.py").read_bytes()).hexdigest(), "python": platform.python_version(), "threads": 1}
    assert frozen["hashes"] == frozen_hashes()
    provenance.update({"freeze": frozen, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "data_description_sha256": hashlib.sha256((ROOT / "数据/数据集说明文档.md").read_bytes()).hexdigest(), "independent_refit": True})
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    pair_metrics = [metric(np.asarray([y[trial == t][0] for t in va]), value) for value, va in zip(values, masks)]
    result = {"subject": "DEP02", "pooled_oof": metric(out.y.to_numpy(), p), "pair_metrics": pair_metrics, "pair_mean": {k: float(np.mean([m[k] for m in pair_metrics])) for k in ("auc", "f1")}, "fixed_0.5_baseline": metric(out.y.to_numpy(), np.full(8, 0.5)), "n_trials": 8, "n_windows": 40, "windows_per_trial": 5, "prediction_count": 4, "candidate_count": len(CANDIDATES), "threshold": FIXED_THRESHOLD, "threshold_search": False, "elapsed_seconds": time.perf_counter() - started}
    result["fixed_0.5_comparison"] = result["pooled_oof"].copy()
    result["pooled_oof"]["f1"] = float(f1_score(out.y, out.prediction_quota4, zero_division=0))
    result.update({"positive_quota": POSITIVE_QUOTA, "quota_search": False, "primary_f1_rule": "top-4 pooled probabilities", "quota_source": "dataset description: four neutral and four positive physical trials", "primary_threshold": None})
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
