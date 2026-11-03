import hashlib, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
LAMBDAS = (0.0, 0.1, 0.3, 0.5, 0.8)
N_COMPONENTS = (2, 4)
BLOCKS = ("all", "band_1", "band_2", "band_3", "band_4", "band_5")
MODELS = (("lda", "auto"), ("lda", 0.1), ("logistic_regression", 0.1), ("logistic_regression", 1.0), ("logistic_regression", 10.0))
CANDIDATES = tuple((lam, nc, block, model, value) for lam in LAMBDAS for nc in N_COMPONENTS for block in BLOCKS for model, value in MODELS)
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
    return np.concatenate(pieces), np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)], np.r_[np.repeat(np.arange(4), 5), 4 + np.repeat(np.arange(4), 5)], path


def covariances(x):
    bands = []
    for low, high in BANDS:
        z = sosfiltfilt(butter(4, [low, high], btype="bandpass", fs=250, output="sos"), x, axis=-1)
        z -= z.mean(axis=-1, keepdims=True)
        bands.append(z @ z.transpose(0, 2, 1) / z.shape[-1])
    return np.stack(bands, axis=1)


def aggregate_trials(cov, y, trial):
    return np.asarray([cov[trial == t].mean(axis=0) for t in range(8)]), np.asarray([y[trial == t][0] for t in range(8)])


def trial_pairs(trial):
    trial = np.asarray(trial)
    for p in np.flatnonzero(trial >= 4):
        for n in np.flatnonzero(trial < 4):
            va = np.asarray([p, n])
            yield np.setdiff1d(np.arange(len(trial)), va), va


def window_indices(trial, selected):
    return np.flatnonzero(np.isin(trial, selected))


def feature_fit_transform(cov_train, y_train, cov_valid, lam, n_components):
    filters = []
    eye = np.eye(30)
    for band in range(5):
        c = cov_train[:, band]
        scale = np.trace(c, axis1=1, axis2=2) / 30.0
        c = (1 - lam) * c + lam * scale[:, None, None] * eye
        c /= np.trace(c, axis1=1, axis2=2)[:, None, None]
        c0, c1 = c[y_train == 0].mean(axis=0), c[y_train == 1].mean(axis=0)
        _, vectors = eigh(c1, c0 + c1)
        take = np.r_[np.arange(n_components), np.arange(30 - n_components, 30)]
        filters.append(vectors[:, take].T)
    filters = np.asarray(filters)

    def transform(values):
        variance = np.einsum("bkc,nbcd,bkd->nbk", filters, values, filters)
        result = np.log(np.maximum(variance, 1e-30))
        assert np.isfinite(result).all()
        return result
    return transform(cov_train), transform(cov_valid)


def predict(features_train, y_train, features_valid, candidate):
    _, _, _, model, value = candidate
    estimator = (LinearDiscriminantAnalysis(solver="lsqr", shrinkage=value) if model == "lda" else LogisticRegression(C=value, solver="liblinear", max_iter=3000, random_state=2026))
    fitted = make_pipeline(StandardScaler(), estimator).fit(features_train, y_train)
    return fitted.predict_proba(features_valid)[:, 1]


def select_inner(cov, y, trial_ids):
    folds = list(trial_pairs(trial_ids))
    cache = {}
    for fold_id, (tr, va) in enumerate(folds):
        for lam in LAMBDAS:
            for nc in N_COMPONENTS:
                train, valid = feature_fit_transform(cov[tr], y[tr], cov[va], lam, nc)
                cache[(fold_id, lam, nc)] = (train, valid)
    scores = []
    for index, candidate in enumerate(CANDIDATES):
        values = []
        _, _, block, _, _ = candidate
        for fold_id, (_, va) in enumerate(folds):
            train, valid = cache[(fold_id, candidate[0], candidate[1])]
            bi = None if block == "all" else int(block[-1]) - 1
            ft = train.reshape(len(train), 5, -1) if bi is None else train[:, bi:bi + 1]
            fv = valid.reshape(len(valid), 5, -1) if bi is None else valid[:, bi:bi + 1]
            ft, fv = ft.reshape(len(ft), -1), fv.reshape(len(fv), -1)
            values.append((predict(ft, y[tr], fv, candidate), va))
        for direction_index, direction in enumerate(("raw", "reverse")):
            p, yy = [], []
            for value, va in values:
                value = value if direction == "raw" else 1 - value
                p.extend(value.tolist()); yy.extend(y[va].tolist())
            scores.append({"candidate_index": index, "candidate": candidate, "direction": direction, "direction_index": direction_index, **metric(np.asarray(yy), np.asarray(p))})
    return sorted(scores, key=lambda r: (-r["auc"], -r["f1"], r["candidate_index"], r["direction_index"]))


def fit_outer(cov_train, y_train, cov_valid, candidate):
    train, valid = feature_fit_transform(cov_train, y_train, cov_valid, candidate[0], candidate[1])
    block = candidate[2]
    bi = None if block == "all" else int(block[-1]) - 1
    ft = train.reshape(len(train), 5, -1) if bi is None else train[:, bi:bi + 1]
    fv = valid.reshape(len(valid), 5, -1) if bi is None else valid[:, bi:bi + 1]
    return predict(ft.reshape(len(ft), -1), y_train, fv.reshape(len(fv), -1), candidate)


def main():
    started = time.perf_counter()
    x, y_window, trial, source = load_subject("DEP02")
    cov_window = covariances(x)
    cov, y = aggregate_trials(cov_window, y_window, trial)
    assert cov.shape == (8, 5, 30, 30) and y.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    values, rows, selections = [], [], []
    with threadpool_limits(limits=1):
        for pair_id, (tr, va) in enumerate(trial_pairs(np.arange(8))):
            ranked = select_inner(cov[tr], y[tr], tr)
            chosen = ranked[0]
            p = fit_outer(cov[tr], y[tr], cov[va], CANDIDATES[chosen["candidate_index"]])
            if chosen["direction"] == "reverse": p = 1 - p
            values.append((va, p))
            selections.append({"pair_id": pair_id, "train_trial_indices": tr.tolist(), "valid_trial_indices": va.tolist(), "train_trial_count": 6, "valid_trial_count": 2, "window_covariance_count": 40, "selected_candidate": chosen["candidate"], "selected_direction": chosen["direction"], "inner_scores": ranked, "selection_source": "inner OOF trial probabilities only"})
            for t, probability in zip(va, p):
                rows.append({"pair_id": pair_id, "trial_index": int(t), "trial_key": f"{int(y[t])}::{int(t % 4) + 1}", "y": int(y[t]), "prob_positive": float(probability), "window_count": 5})
    sums, counts = np.zeros(8), np.zeros(8, dtype=int)
    for va, p in values: sums[va] += p; counts[va] += 1
    p = sums / counts
    out = pd.DataFrame({"trial_index": np.arange(8), "trial_key": [f"{y[t]}::{t % 4 + 1}" for t in range(8)], "y": y, "prob_positive": p, "prediction_count": counts})
    out["prediction_quota4"] = 0
    out.loc[np.argsort(-p, kind="stable")[:4], "prediction_quota4"] = 1
    out["prediction_fixed05"] = (p >= 0.5).astype(int)
    (HERE / "predictions/DEP02_trial_regularized_csp_oof.csv").parent.mkdir(exist_ok=True)
    out.to_csv(HERE / "predictions/DEP02_trial_regularized_csp_oof.csv", index=False)
    pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf8")
    frozen = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "hashes": {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("run_experiment.py", "config.json", "冻结协议.md")}}
    (HERE / "input_metadata/freeze.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf8")
    provenance = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256((HERE / "run_experiment.py").read_bytes()).hexdigest(), "python": platform.python_version(), "threads": 1, "protocol": "trial_mean_covariance_regularized_filterbank_csp_nested_pairs", "outer_validation_labels_used_for_selection": False, "independent_refit": True, "freeze": frozen, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "data_description_sha256": hashlib.sha256((ROOT / "数据/数据集说明文档.md").read_bytes()).hexdigest()}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    pair_metrics = [metric(y[va], pv) for va, pv in values]
    result = {"experiment_id": "121_DEP02_regularized_csp_ensemble_pairs", "subject": "DEP02", "pooled_oof": {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, out.prediction_quota4, zero_division=0))}, "fixed_0.5": metric(y, p), "pair_mean": {k: float(np.mean([m[k] for m in pair_metrics])) for k in ("auc", "f1")}, "pair_metrics": pair_metrics, "n_trials": 8, "n_windows": 40, "windows_per_trial": 5, "candidate_count": len(CANDIDATES), "candidate_direction_count": len(CANDIDATES) * 2, "positive_quota": 4, "threshold": 0.5, "outer_validation_labels_used_for_selection": False, "elapsed_seconds": time.perf_counter() - started}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
