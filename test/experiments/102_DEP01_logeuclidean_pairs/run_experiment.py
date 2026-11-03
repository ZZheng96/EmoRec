import hashlib
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
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
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf8"))
assert tuple(map(tuple, CONFIG["bands"])) == BANDS
REGS = tuple(CONFIG["regularization"])
BAND_SETS = tuple(map(tuple, CONFIG["band_sets"]))
CS = tuple(CONFIG["C"])
CANDIDATES = tuple(
    (model, reg, bands, c)
    for model in CONFIG["classifier"]
    for reg in REGS
    for bands in BAND_SETS
    for c in CS
)


def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, p >= 0.5, zero_division=0))}


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
    trial = np.tile(np.repeat(np.arange(1, 5), 5), 2)
    return x, y, trial, path


def covariances(x):
    all_bands = []
    for low, high in BANDS:
        sos = butter(4, [low, high], btype="bandpass", fs=250, output="sos")
        z = sosfiltfilt(sos, x, axis=-1)
        z -= z.mean(axis=-1, keepdims=True)
        all_bands.append(z @ z.transpose(0, 2, 1) / z.shape[-1])
    return np.stack(all_bands, axis=1)


def trial_covariances(cov, y, trial):
    keys = np.asarray([f"{int(a)}::{int(b)}" for a, b in zip(y, trial)], dtype=object)
    unique = list(dict.fromkeys(keys))
    return (np.asarray([cov[keys == key].mean(axis=0) for key in unique]),
            np.asarray([y[keys == key][0] for key in unique]), unique)


def spd_log(cov, reg):
    regularized = (cov + cov.swapaxes(-1, -2)) / 2 + reg * np.eye(cov.shape[-1])
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    assert np.all(eigenvalues > 0)
    return (eigenvectors * np.log(eigenvalues)[..., None, :]) @ eigenvectors.swapaxes(-1, -2)


def logeuclidean_fit_transform(cov_train, cov_valid, reg, bands):
    reference = spd_log(cov_train[:, bands].mean(axis=0), reg)
    upper = np.triu_indices(cov_train.shape[-1])

    def transform(values):
        centered = spd_log(values[:, bands], reg) - reference
        result = centered[..., upper[0], upper[1]].reshape(len(values), -1)
        assert np.isfinite(result).all()
        return result

    return transform(cov_train), transform(cov_valid)


def predict(cov_train, y_train, cov_valid, candidate):
    model_name, reg, bands, c = candidate
    train_features, valid_features = logeuclidean_fit_transform(cov_train, cov_valid, reg, bands)
    if model_name == "logistic_regression":
        estimator = LogisticRegression(C=c, solver="liblinear", max_iter=3000, random_state=2026)
    else:
        estimator = LinearSVC(C=c, dual="auto", max_iter=5000, random_state=2026)
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(train_features, y_train)
    score = model.decision_function(valid_features)
    return 1 / (1 + np.exp(-np.clip(score, -60, 60)))


def pairs(y):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    for p in pos:
        for n in neg:
            va = np.asarray([p, n])
            tr = np.setdiff1d(np.arange(len(y)), va)
            yield tr, va


def select(cov, y):
    inner = list(pairs(y))
    records = []
    for index, candidate in enumerate(CANDIDATES):
        values = [predict(cov[tr], y[tr], cov[va], candidate) for tr, va in inner]
        for direction_index, direction in enumerate(("raw", "reverse")):
            p = np.zeros(len(y)); count = np.zeros(len(y), dtype=int)
            for value, (_, va) in zip(values, inner):
                p[va] += value if direction == "raw" else 1 - value; count[va] += 1
            p /= count
            records.append({"candidate_index": index, "candidate": candidate, "direction": direction, "direction_index": direction_index, **metric(y, p)})
    best = max(records, key=lambda r: (r["auc"], r["f1"], -r["candidate_index"], -r["direction_index"]))
    return (CANDIDATES[best["candidate_index"]], best["direction"]), records


def main():
    started = time.perf_counter()
    x, y_windows, trial_windows, source = load_subject("DEP01")
    cov = covariances(x)
    cov, y, keys = trial_covariances(cov, y_windows, trial_windows)
    assert cov.shape == (8, 5, 30, 30)
    values, masks, rows, selections = [], [], [], []
    with threadpool_limits(limits=1):
        for pair_id, (tr, va) in enumerate(pairs(y)):
            selected, scores = select(cov[tr], y[tr])
            candidate, direction = selected
            value = predict(cov[tr], y[tr], cov[va], candidate)
            if direction == "reverse":
                value = 1 - value
            values.append(value); masks.append(va)
            selections.append({"pair_id": pair_id, "train_indices": tr.tolist(), "valid_indices": va.tolist(), "selected_candidate": candidate, "selected_direction": direction, "inner_scores": scores})
            for index, probability in zip(va, value):
                rows.append({"pair_id": pair_id, "trial_index": int(index), "trial_key": keys[index], "y": int(y[index]), "prob_positive": float(probability)})
    sums = np.zeros(8); counts = np.zeros(8, dtype=int)
    for value, va in zip(values, masks):
        sums[va] += value; counts[va] += 1
    assert np.all(counts == 4)
    p = sums / counts
    out = pd.DataFrame({"trial_index": np.arange(8), "trial_key": keys, "y": y, "prob_positive": p, "prediction_count": counts})
    out.to_csv(HERE / "predictions/DEP01_logeuclidean_pairs_oof.csv", index=False)
    pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf8")
    provenance = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256((HERE / "run_experiment.py").read_bytes()).hexdigest(), "config_sha256": hashlib.sha256((HERE / "config.json").read_bytes()).hexdigest(), "protocol_sha256": hashlib.sha256((HERE / "冻结协议.md").read_bytes()).hexdigest(), "python": platform.python_version(), "threads": 1, "device": "cpu", "trial_keys": keys, "trial_covariance_shape": list(cov.shape), "reference_fit": "training trial arithmetic mean per band only", "source_experiment": "101_DEP01_filterbank_csp_direction_pairs"}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    pair_metrics = [metric(y[va], value) for value, (_, va) in zip(values, pairs(y))]
    result = {"subject": "DEP01", "pooled_oof": metric(y, p), "pair_mean": {k: float(np.mean([m[k] for m in pair_metrics])) for k in ("auc", "f1")}, "constant_0.5": metric(y, np.full(8, 0.5)), "n_trials": 8, "n_windows": 40, "prediction_count": 4, "candidate_count": len(CANDIDATES), "elapsed_seconds": time.perf_counter() - started}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
