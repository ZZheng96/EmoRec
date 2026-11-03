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

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[2]
HERE = ROOT / "test" / "experiments" / "137_HC11_csp_transfer_pairs"
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
REGS = (1e-8, 1e-6, 1e-4, 1e-2)
ENDS = (1, 2, 3)
CS = (0.01, 0.1, 1, 10, 100)
CANDIDATES = tuple((m, r, e, c) for m in ("logistic_regression", "linear_svc") for r in REGS for e in ENDS for c in CS)
FIXED_THRESHOLD = 0.5
POSITIVE_QUOTA = 4


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def quota_predictions(p):
    p = np.asarray(p, dtype=float)
    assert p.shape == (8,) and np.isfinite(p).all()
    out = np.zeros(8, dtype=int)
    out[np.argsort(-p, kind="stable")[:POSITIVE_QUOTA]] = 1
    return out


def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, p >= .5, zero_division=0))}


def load_hc11():
    path = ROOT / "数据" / "训练集" / "正常人" / "HC11.mat"
    with h5py.File(path, "r") as mat:
        pieces = [np.asarray(mat[k][()], dtype=np.float64).T.reshape(30, 20, 2500).transpose(1, 0, 2) for k in ("EEG_data_neu", "EEG_data_pos")]
    x = np.concatenate(pieces)
    y = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    trial = np.r_[np.repeat(np.arange(4), 5), 4 + np.repeat(np.arange(4), 5)]
    return x, y, trial, path


def load_720():
    cfg = json.loads((HERE.parent / "091_within_subject_trial_oof" / "config.json").read_text(encoding="utf8"))
    base = HERE.parent / "091_within_subject_trial_oof"
    arrays = []
    for key in ("spectral_cache", "temporal_cache", "pe_cache"):
        with np.load((base / cfg[key]).resolve(), allow_pickle=False) as z:
            arrays.append({k: z[k] for k in ("X_train", "y_train", "subject_ids", "original_trial_ids", "feature_names")})
    s, t, pe = arrays
    keep = s["subject_ids"] == "HC11"
    for cache in arrays:
        for key in ("subject_ids", "original_trial_ids", "y_train"):
            assert np.array_equal(cache[key], s[key]), key
        assert np.array_equal(cache["y_train"][keep], np.repeat([0, 1], 20))
        assert np.array_equal(cache["original_trial_ids"][keep], np.tile(np.repeat(np.arange(1, 5), 5), 2))
    mask = np.asarray([str(n).startswith("log_bandpower_") for n in s["feature_names"]])
    x = np.column_stack((s["X_train"][keep][:, mask], t["X_train"][keep], pe["X_train"][keep]))
    y = s["y_train"][keep].astype(int)
    ref = pd.DataFrame({"row_index": np.arange(40), "subject": s["subject_ids"][keep], "trial": s["original_trial_ids"][keep], "y": y})
    assert x.shape == (40, 720)
    return x, ref


def trial_pairs(trials):
    trials = np.asarray(trials)
    pos = np.flatnonzero(trials >= 4)
    neg = np.flatnonzero(trials < 4)
    for p in pos:
        for n in neg:
            valid = np.asarray([p, n])
            yield np.setdiff1d(np.arange(len(trials)), valid), valid


def window_indices(trial, selected):
    return np.flatnonzero(np.isin(trial, selected))


def covariances(x):
    result = []
    for low, high in BANDS:
        sos = butter(4, [low, high], btype="bandpass", fs=250, output="sos")
        z = sosfiltfilt(sos, x, axis=-1)
        z -= z.mean(axis=-1, keepdims=True)
        result.append(z @ z.transpose(0, 2, 1) / z.shape[-1])
    return np.stack(result, axis=1)


def csp_fit_transform(cov_train, y_train, cov_valid, reg, end):
    filters = []
    eye = np.eye(30)
    for band in range(5):
        c = cov_train[:, band] + reg * eye
        c /= np.trace(c, axis1=1, axis2=2)[:, None, None]
        c0, c1 = c[y_train == 0].mean(0), c[y_train == 1].mean(0)
        _, vec = eigh(c1, c0 + c1)
        take = np.r_[np.arange(end), np.arange(30 - end, 30)]
        filters.append(vec[:, take].T)
    filters = np.asarray(filters)
    def transform(v):
        variance = np.einsum("bkc,nbcd,bkd->nbk", filters, v, filters)
        out = np.log(np.maximum(variance, 1e-30)).reshape(len(v), -1)
        assert np.isfinite(out).all()
        return out
    return transform(cov_train), transform(cov_valid)


def predict(cov_train, y_train, cov_valid, candidate):
    name, reg, end, c = candidate
    a, b = csp_fit_transform(cov_train, y_train, cov_valid, reg, end)
    est = LogisticRegression(C=c, solver="liblinear", max_iter=3000, random_state=2026) if name == "logistic_regression" else LinearSVC(C=c, dual="auto", max_iter=5000, random_state=2026)
    model = make_pipeline(StandardScaler(), est).fit(a, y_train)
    return sigmoid(model.decision_function(b))


def select(cov, y, trial):
    records = []
    unique = np.unique(trial)
    for ci, candidate in enumerate(CANDIDATES):
        vals = []
        for tr_i, va_i in trial_pairs(unique):
            tr = window_indices(trial, unique[tr_i]); va = window_indices(trial, unique[va_i])
            vals.append((predict(cov[tr], y[tr], cov[va], candidate), va))
        for di, direction in enumerate(("raw", "reverse")):
            by_trial = {}
            labels = {}
            for p, va in vals:
                if direction == "reverse": p = 1 - p
                for t in np.unique(trial[va]):
                    m = trial[va] == t
                    by_trial.setdefault(int(t), []).append(float(p[m].mean()))
                    labels[int(t)] = int(y[va][m][0])
            keys = sorted(by_trial)
            pp = np.asarray([np.mean(by_trial[k]) for k in keys])
            yy = np.asarray([labels[k] for k in keys])
            records.append({"candidate_index": ci, "candidate": candidate, "direction": direction, "direction_index": di, **metric(yy, pp)})
    best = max(records, key=lambda r: (r["auc"], r["f1"], -r["candidate_index"], -r["direction_index"]))
    return (CANDIDATES[best["candidate_index"]], best["direction"]), records


def main():
    started = time.perf_counter()
    (HERE / "predictions").mkdir(exist_ok=True); (HERE / "input_metadata").mkdir(exist_ok=True); (HERE / "selections").mkdir(exist_ok=True)
    freeze = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "hashes": frozen_hashes() if (HERE / "冻结协议.md").exists() else {}}
    eeg, y, trial, source = load_hc11()
    x720, ref = load_720()
    assert np.array_equal(ref.y.to_numpy(), y)
    assert np.array_equal(ref.trial.to_numpy() - 1 + 4 * ref.y.to_numpy(), trial)
    ref.to_csv(HERE / "input_metadata/cache_alignment.csv", index=False)
    p_base = np.full(40, np.nan); folds = np.full(40, -1, dtype=int)
    for fold, held in enumerate(np.unique(trial)):
        tr, va = trial != held, trial == held
        model = make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=2026)).fit(x720[tr], y[tr])
        p_base[va] = sigmoid(model.decision_function(x720[va])); folds[va] = fold
    pd.DataFrame({"row_index": np.arange(40), "trial_index": trial, "y": y, "fold": folds, "prob_positive": p_base}).to_csv(HERE / "predictions/baseline_720_linearsvc_oof.csv", index=False)
    cov = covariances(eeg)
    rows, selections, values, masks = [], [], [], []
    with threadpool_limits(limits=1):
        for pair_id, (tr_trials, va_trials) in enumerate(trial_pairs(np.arange(8))):
            tr = window_indices(trial, tr_trials); va = window_indices(trial, va_trials)
            selected, scores = select(cov[tr], y[tr], trial[tr])
            candidate, direction = selected
            p = predict(cov[tr], y[tr], cov[va], candidate)
            if direction == "reverse": p = 1 - p
            vu = np.unique(trial[va]); pv = np.asarray([p[trial[va] == t].mean() for t in vu])
            values.append(pv); masks.append(vu)
            selections.append({"pair_id": pair_id, "train_trial_indices": tr_trials.tolist(), "valid_trial_indices": va_trials.tolist(), "train_window_count": int(len(tr)), "valid_window_count": int(len(va)), "selected_candidate": candidate, "selected_direction": direction, "inner_scores": scores})
            for t, prob in zip(vu, pv): rows.append({"pair_id": pair_id, "trial_index": int(t), "y": int(y[trial == t][0]), "prob_positive": float(prob), "window_count": 5})
    sums = np.zeros(8); counts = np.zeros(8, dtype=int)
    for p, vu in zip(values, masks): sums[vu] += p; counts[vu] += 1
    p = sums / counts
    out = pd.DataFrame({"trial_index": np.arange(8), "trial_key": [f"{int(y[trial == t][0])}::{t % 4 + 1}" for t in range(8)], "y": [int(y[trial == t][0]) for t in range(8)], "prob_positive": p, "prediction_count": counts})
    out["prediction_quota4"] = quota_predictions(p); out["prediction_fixed05"] = (p >= .5).astype(int)
    out.to_csv(HERE / "predictions/HC11_window_csp_pairs_oof.csv", index=False); pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf8")
    (HERE / "selections/provenance.json").write_text(json.dumps({"selection": "inner training trials only", "outer_validation_labels_used_for_selection": False, "candidate_count": len(CANDIDATES)}, ensure_ascii=False, indent=2), encoding="utf8")
    assert freeze["hashes"] == frozen_hashes(), "Frozen files changed during execution"
    provenance = {"source": str(source), "source_sha256": sha(source), "freeze": freeze, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "threads": 1, "public_arrays_loaded": False}
    (HERE / "input_metadata/freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf8"); (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    baseline = {"auc": float(roc_auc_score(y, p_base)), "f1": float(f1_score(y, p_base >= .5, zero_division=0))}
    csp = {"auc": float(roc_auc_score(out.y, out.prob_positive)), "f1_fixed05": float(f1_score(out.y, out.prediction_fixed05, zero_division=0)), "f1_quota4": float(f1_score(out.y, out.prediction_quota4, zero_division=0))}
    metrics = {"subject": "HC11", "baseline_720_linearsvc_fixed05": baseline, "csp_window_pairs": csp, "n_trials": 8, "windows_per_trial": 5, "candidate_count": len(CANDIDATES), "positive_quota": 4, "quota_search": False, "threshold_search": False, "outer_validation_labels_used_for_selection": False, "elapsed_seconds": time.perf_counter() - started, "provenance": provenance}
    (HERE / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def frozen_hashes():
    paths = (SOURCE, HERE / "config.json", HERE / "冻结协议.md")
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
if __name__ == "__main__": main()

