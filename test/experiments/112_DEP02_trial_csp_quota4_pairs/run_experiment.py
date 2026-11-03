import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
REGS = (1e-8, 1e-6, 1e-4, 1e-2)
ENDS = (1, 2, 3)
CS = (0.01, 0.1, 1, 10, 100)
CANDIDATES = tuple((m, r, e, c) for m in ("logistic_regression", "linear_svc") for r in REGS for e in ENDS for c in CS)
FIXED_THRESHOLD = 0.5
POSITIVE_QUOTA = 4


def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, np.asarray(p) >= FIXED_THRESHOLD, zero_division=0))}


def quota_predictions(p):
    p = np.asarray(p)
    out = np.zeros(8, dtype=int)
    out[np.argsort(-p, kind="stable")[:POSITIVE_QUOTA]] = 1
    return out


def load_subject(subject):
    paths = [p for p in (ROOT / "数据" / "训练集").glob("*/*.mat") if p.stem == subject]
    assert len(paths) == 1
    path = paths[0]
    with h5py.File(path, "r") as mat:
        pieces = []
        for key in ("EEG_data_neu", "EEG_data_pos"):
            data = np.asarray(mat[key][()], dtype=np.float64).T
            assert data.shape == (30, 50000)
            pieces.append(data.reshape(30, 20, 2500).transpose(1, 0, 2))
    return np.concatenate(pieces), np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)], np.r_[np.repeat(np.arange(4), 5), 4 + np.repeat(np.arange(4), 5)], path


def covariances(x):
    bands = []
    for low, high in BANDS:
        z = sosfiltfilt(butter(4, [low, high], btype="bandpass", fs=250, output="sos"), x, axis=-1)
        z -= z.mean(axis=-1, keepdims=True)
        bands.append(z @ z.transpose(0, 2, 1) / z.shape[-1])
    return np.stack(bands, axis=1)


def aggregate_trials(cov, y, trial):
    ids = np.arange(8)
    return np.asarray([cov[trial == t].mean(axis=0) for t in ids]), np.asarray([y[trial == t][0] for t in ids])


def csp_fit_transform(cov_train, y_train, cov_valid, reg, end):
    filters, eye = [], np.eye(30)
    for band in range(5):
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
    model = make_pipeline(StandardScaler(), estimator).fit(train_features, y_train)
    score = model.decision_function(valid_features)
    return 1 / (1 + np.exp(-np.clip(score, -60, 60)))


def trial_pairs(trial):
    trial = np.asarray(trial)
    for p in np.flatnonzero(trial >= 4):
        for n in np.flatnonzero(trial < 4):
            va = np.asarray([p, n])
            yield np.setdiff1d(np.arange(len(trial)), va), va


def window_indices(trial, selected_trials):
    return np.flatnonzero(np.isin(trial, selected_trials))


def select_inner(cov, y, trial_ids):
    scores = []
    for index, candidate in enumerate(CANDIDATES):
        folds = []
        for tr_pos, va_pos in trial_pairs(trial_ids):
            folds.append((predict(cov[tr_pos], y[tr_pos], cov[va_pos], candidate), va_pos))
        for direction_index, direction in enumerate(("raw", "reverse")):
            p, yy = [], []
            for value, va_pos in folds:
                value = value if direction == "raw" else 1 - value
                p.extend(value.tolist())
                yy.extend(y[va_pos].tolist())
            scores.append({"candidate_index": index, "candidate": candidate, "direction": direction, "direction_index": direction_index, **metric(np.asarray(yy), np.asarray(p))})
    ranked = sorted(scores, key=lambda r: (-r["auc"], -r["f1"], r["candidate_index"], r["direction_index"]))
    return ranked


def frozen_hashes():
    return {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("run_experiment.py", "config.json", "冻结协议.md")}


def main():
    started = time.perf_counter()
    x, y_window, window_trial, source = load_subject("DEP02")
    cov_window = covariances(x)
    cov, y = aggregate_trials(cov_window, y_window, window_trial)
    assert cov.shape == (8, 5, 30, 30) and y.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    values, rows, selections = [], [], []
    with threadpool_limits(limits=1):
        for pair_id, (tr, va) in enumerate(trial_pairs(np.arange(8))):
            ranked = select_inner(cov[tr], y[tr], tr)
            chosen = ranked[0]
            candidate = CANDIDATES[chosen["candidate_index"]]
            p = predict(cov[tr], y[tr], cov[va], candidate)
            if chosen["direction"] == "reverse": p = 1 - p
            values.append((va, p))
            selections.append({"pair_id": pair_id, "train_trial_indices": tr.tolist(), "valid_trial_indices": va.tolist(), "train_trial_count": 6, "valid_trial_count": 2, "window_covariance_count": 40, "selected_candidate": candidate, "selected_direction": chosen["direction"], "inner_scores": ranked, "selection_source": "inner OOF trial probabilities only"})
            for t, probability in zip(va, p):
                rows.append({"pair_id": pair_id, "trial_index": int(t), "trial_key": f"{int(y[t])}::{int(t % 4) + 1}", "y": int(y[t]), "prob_positive": float(probability), "window_count": 5})
    sums, counts = np.zeros(8), np.zeros(8, dtype=int)
    for va, p in values: sums[va] += p; counts[va] += 1
    p = sums / counts
    out = pd.DataFrame({"trial_index": np.arange(8), "trial_key": [f"{y[t]}::{t % 4 + 1}" for t in range(8)], "y": y, "prob_positive": p, "prediction_count": counts})
    out["prediction_quota4"] = quota_predictions(p)
    out["prediction_fixed05"] = (p >= FIXED_THRESHOLD).astype(int)
    out.to_csv(HERE / "predictions/DEP02_trial_csp_quota4_oof.csv", index=False)
    pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf8")
    frozen = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "hashes": frozen_hashes()}
    (HERE / "input_metadata/freeze.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf8")
    provenance = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256((HERE / "run_experiment.py").read_bytes()).hexdigest(), "python": platform.python_version(), "threads": 1, "protocol": "trial_mean_covariance_strict_nested_pairs", "outer_validation_labels_used_for_selection": False, "independent_refit": True, "freeze": frozen, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "data_description_sha256": hashlib.sha256((ROOT / "数据/数据集说明文档.md").read_bytes()).hexdigest()}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    pair_metrics = [metric(y[va], pv) for va, pv in values]
    result = {"experiment_id": "112_DEP02_trial_csp_quota4_pairs", "subject": "DEP02", "pooled_oof": {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, out.prediction_quota4, zero_division=0))}, "fixed_0.5": metric(y, p), "pair_mean": {k: float(np.mean([m[k] for m in pair_metrics])) for k in ("auc", "f1")}, "pair_metrics": pair_metrics, "n_trials": 8, "n_windows": 40, "windows_per_trial": 5, "candidate_count": 120, "positive_quota": 4, "quota_search": False, "threshold": 0.5, "threshold_search": False, "outer_validation_labels_used_for_selection": False, "elapsed_seconds": time.perf_counter() - started}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
