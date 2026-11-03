import hashlib, json, platform, time
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
LAMBDAS = (0.0, 0.1, 0.3, 0.5)
N_COMPONENTS = (2, 4)
# Five frozen blocks are all bands together plus the first four individual bands.
BLOCKS = ("all", "band_1", "band_2", "band_3", "band_4")
MODELS = ("linear_svc", "logistic_regression")
CS = (0.01, 0.1, 1, 10)
CANDIDATES = tuple((lam, comp, block, model, c) for lam in LAMBDAS for comp in N_COMPONENTS for block in BLOCKS for model in MODELS for c in CS)
FIXED_THRESHOLD = 0.5
POSITIVE_QUOTA = 4


def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, np.asarray(p) >= FIXED_THRESHOLD, zero_division=0))}


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
    x = np.concatenate(pieces)
    y = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    trial = np.r_[np.repeat(np.arange(4), 5), 4 + np.repeat(np.arange(4), 5)]
    return x, y, trial, path


def covariances(x):
    bands = []
    for low, high in BANDS:
        z = sosfiltfilt(butter(4, [low, high], btype="bandpass", fs=250, output="sos"), x, axis=-1)
        z -= z.mean(axis=-1, keepdims=True)
        bands.append(z @ z.transpose(0, 2, 1) / z.shape[-1])
    return np.stack(bands, axis=1)


def trial_pairs(trial):
    trial = np.asarray(trial)
    for p in np.flatnonzero(trial >= 4):
        for n in np.flatnonzero(trial < 4):
            va = np.asarray([p, n])
            yield np.setdiff1d(np.arange(len(trial)), va), va


def window_indices(trial, selected):
    return np.flatnonzero(np.isin(trial, selected))


def csp_fit_transform(cov_train, y_train, cov_valid, lam, components):
    filters = []
    eye = np.eye(30)
    for band in range(5):
        c = cov_train[:, band]
        scale = np.trace(c, axis1=1, axis2=2) / 30.0
        c = (1 - lam) * c + lam * scale[:, None, None] * eye
        c /= np.trace(c, axis1=1, axis2=2)[:, None, None]
        c0, c1 = c[y_train == 0].mean(axis=0), c[y_train == 1].mean(axis=0)
        _, vectors = eigh(c1, c0 + c1)
        take = np.r_[np.arange(components), np.arange(30 - components, 30)]
        filters.append(vectors[:, take].T)
    filters = np.asarray(filters)

    def transform(values):
        variance = np.einsum("bkc,nbcd,bkd->nbk", filters, values, filters)
        features = np.log(np.maximum(variance, 1e-30))
        assert np.isfinite(features).all()
        return features
    return transform(cov_train), transform(cov_valid)


def block_features(features, block):
    band = None if block == "all" else int(block[-1]) - 1
    selected = features if band is None else features[:, band:band + 1]
    return selected.reshape(len(selected), -1)


def predict(train_features, y_train, valid_features, candidate):
    _, _, _, model_name, c = candidate
    if model_name == "logistic_regression":
        estimator = LogisticRegression(C=c, solver="liblinear", max_iter=3000, random_state=2026)
    else:
        estimator = LinearSVC(C=c, dual="auto", max_iter=5000, random_state=2026)
    model = make_pipeline(StandardScaler(), estimator).fit(train_features, y_train)
    score = model.decision_function(valid_features)
    return 1 / (1 + np.exp(-np.clip(score, -60, 60)))


def select_inner(cov, y, trial):
    folds = list(trial_pairs(np.unique(trial)))
    scores = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        fold_values = []
        for tr_trials, va_trials in folds:
            tr = window_indices(trial, np.unique(trial)[tr_trials])
            va = window_indices(trial, np.unique(trial)[va_trials])
            ft, fv = csp_fit_transform(cov[tr], y[tr], cov[va], candidate[0], candidate[1])
            p = predict(block_features(ft, candidate[2]), y[tr], block_features(fv, candidate[2]), candidate)
            # Aggregate each validation trial's five windows before scoring.
            keys = np.unique(trial[va])
            fold_values.append((np.asarray([p[trial[va] == t].mean() for t in keys]), np.asarray([y[va][trial[va] == t][0] for t in keys])))
        for direction_index, direction in enumerate(("raw", "reverse")):
            pp = np.concatenate([v if direction == "raw" else 1 - v for v, _ in fold_values])
            yy = np.concatenate([yy for _, yy in fold_values])
            scores.append({"candidate_index": candidate_index, "candidate": candidate, "direction": direction, "direction_index": direction_index, **metric(yy, pp)})
    return sorted(scores, key=lambda r: (-r["auc"], -r["f1"], r["candidate_index"], r["direction_index"])), scores


def fit_outer(cov_train, y_train, cov_valid, candidate):
    ft, fv = csp_fit_transform(cov_train, y_train, cov_valid, candidate[0], candidate[1])
    return predict(block_features(ft, candidate[2]), y_train, block_features(fv, candidate[2]), candidate)


def main():
    started = time.perf_counter()
    x, y, trial, source = load_subject("DEP02")
    cov = covariances(x)
    assert cov.shape == (40, 5, 30, 30)
    values, rows, selections = [], [], []
    with threadpool_limits(limits=1):
        for pair_id, (tr_trials, va_trials) in enumerate(trial_pairs(np.arange(8))):
            tr = window_indices(trial, tr_trials)
            va = window_indices(trial, va_trials)
            ranked, scores = select_inner(cov[tr], y[tr], trial[tr])
            chosen = ranked[0]
            p = fit_outer(cov[tr], y[tr], cov[va], CANDIDATES[chosen["candidate_index"]])
            if chosen["direction"] == "reverse": p = 1 - p
            keys = np.unique(trial[va])
            trial_p = np.asarray([p[trial[va] == t].mean() for t in keys])
            values.append((keys, trial_p))
            selections.append({"pair_id": pair_id, "train_trial_indices": tr_trials.tolist(), "valid_trial_indices": va_trials.tolist(), "train_window_count": int(len(tr)), "valid_window_count": int(len(va)), "selected_candidate": chosen["candidate"], "selected_direction": chosen["direction"], "inner_scores": scores, "selection_source": "inner OOF trial probabilities only"})
            for t, probability in zip(keys, trial_p):
                rows.append({"pair_id": pair_id, "trial_index": int(t), "trial_key": f"{int(y[trial == t][0])}::{int(t % 4) + 1}", "y": int(y[trial == t][0]), "prob_positive": float(probability), "window_count": 5})
    sums, counts = np.zeros(8), np.zeros(8, dtype=int)
    for keys, probs in values: sums[keys] += probs; counts[keys] += 1
    assert np.all(counts == 4)
    p = sums / counts
    yy = np.asarray([y[trial == t][0] for t in range(8)])
    out = pd.DataFrame({"trial_index": np.arange(8), "trial_key": [f"{yy[t]}::{t % 4 + 1}" for t in range(8)], "y": yy, "prob_positive": p, "prediction_count": counts})
    out["prediction_quota4"] = 0
    out.loc[np.argsort(-p, kind="stable")[:4], "prediction_quota4"] = 1
    out["prediction_fixed05"] = (p >= 0.5).astype(int)
    out.to_csv(HERE / "predictions/DEP02_window_regularized_csp_pairs_oof.csv", index=False)
    pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    frozen = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "hashes": {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("run_experiment.py", "config.json", "冻结协议.md")}}
    (HERE / "input_metadata/freeze.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf8")
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf8")
    provenance = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256((HERE / "run_experiment.py").read_bytes()).hexdigest(), "python": platform.python_version(), "threads": 1, "outer_validation_labels_used_for_selection": False, "independent_refit": True, "freeze": frozen, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "data_description_sha256": hashlib.sha256((ROOT / "数据/数据集说明文档.md").read_bytes()).hexdigest()}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    pair_metrics = [metric(np.asarray([yy[t] for t in keys]), probs) for keys, probs in values]
    result = {"experiment_id": "122_DEP02_window_regularized_csp_pairs", "subject": "DEP02", "pooled_oof": {"auc": float(roc_auc_score(yy, p)), "f1": float(f1_score(yy, out.prediction_quota4, zero_division=0))}, "fixed_0.5": metric(yy, p), "pair_mean": {k: float(np.mean([m[k] for m in pair_metrics])) for k in ("auc", "f1")}, "pair_metrics": pair_metrics, "n_trials": 8, "n_windows": 40, "windows_per_trial": 5, "candidate_count": len(CANDIDATES), "candidate_direction_count": len(CANDIDATES) * 2, "positive_quota": 4, "threshold": 0.5, "outer_validation_labels_used_for_selection": False, "elapsed_seconds": time.perf_counter() - started}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
