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
TOP_KS = (1, 3, 5, 10, 20)
FIXED_THRESHOLD = 0.5


def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, p >= FIXED_THRESHOLD, zero_division=0))}


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


def csp_fit_transform(cov_train, y_train, cov_valid, reg, end):
    filters, eye = [], np.eye(30)
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
    model = make_pipeline(StandardScaler(), estimator).fit(train_features, y_train)
    score = model.decision_function(valid_features)
    return 1 / (1 + np.exp(-np.clip(score, -60, 60)))


def trial_pairs(trial):
    pos, neg = np.flatnonzero(np.asarray(trial) >= 4), np.flatnonzero(np.asarray(trial) < 4)
    for p in pos:
        for n in neg:
            va = np.asarray([p, n])
            yield np.setdiff1d(np.arange(len(trial)), va), va


def window_indices(trial, selected_trials):
    return np.flatnonzero(np.isin(trial, selected_trials))


def aggregate(values, trial):
    unique = np.asarray(sorted(set(trial)))
    return unique, np.asarray([values[np.asarray(trial) == t].mean() for t in unique])


def select_inner(cov, y, trial):
    inner = list(trial_pairs(np.unique(trial)))
    candidate_oof = {}
    for index, candidate in enumerate(CANDIDATES):
        folds = []
        for tr_trials, va_trials in inner:
            tr = window_indices(trial, np.unique(trial)[tr_trials])
            va = window_indices(trial, np.unique(trial)[va_trials])
            folds.append((predict(cov[tr], y[tr], cov[va], candidate), va, trial[va]))
        for direction_index, direction in enumerate(("raw", "reverse")):
            by_trial, labels = {}, {}
            for value, va, va_trial in folds:
                value = value if direction == "raw" else 1 - value
                for t in np.unique(va_trial):
                    mask = va_trial == t
                    by_trial.setdefault(int(t), []).append(float(value[mask].mean()))
                    labels[int(t)] = int(y[va][mask][0])
            keys = sorted(by_trial)
            p = np.asarray([np.mean(by_trial[k]) for k in keys])
            yy = np.asarray([labels[k] for k in keys])
            candidate_oof[(index, direction)] = (p, yy)
    scores = []
    for (index, direction), (p, yy) in candidate_oof.items():
        scores.append({"candidate_index": index, "candidate": CANDIDATES[index], "direction": direction, "direction_index": int(direction == "reverse"), **metric(yy, p)})
    ranked = sorted(scores, key=lambda r: (-r["auc"], -r["f1"], r["candidate_index"], r["direction_index"]))
    k_scores = []
    for k in TOP_KS:
        chosen = ranked[:k]
        p = np.mean([candidate_oof[(r["candidate_index"], r["direction"])] [0] for r in chosen], axis=0)
        yy = candidate_oof[(chosen[0]["candidate_index"], chosen[0]["direction"])][1]
        k_scores.append({"K": k, **metric(yy, p)})
    selected_k = max(k_scores, key=lambda r: (-r["auc"], -r["f1"], -r["K"]))["K"]
    return ranked, scores, k_scores, selected_k


def frozen_hashes():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in ("run_experiment.py", "config.json", "冻结协议.md")}


def main():
    started = time.perf_counter()
    x, y, trial, source = load_subject("DEP02")
    cov = covariances(x)
    assert cov.shape == (40, 5, 30, 30)
    values, masks, rows, selections = [], [], [], []
    with threadpool_limits(limits=1):
        for pair_id, (tr_trials, va_trials) in enumerate(trial_pairs(np.arange(8))):
            tr, va = window_indices(trial, tr_trials), window_indices(trial, va_trials)
            ranked, scores, k_scores, selected_k = select_inner(cov[tr], y[tr], trial[tr])
            selected = ranked[:selected_k]
            outer_predictions = {}
            for item in selected:
                key = (item["candidate_index"], item["direction"])
                p = predict(cov[tr], y[tr], cov[va], CANDIDATES[item["candidate_index"]])
                outer_predictions[key] = p if item["direction"] == "raw" else 1 - p
            p = np.mean([outer_predictions[(r["candidate_index"], r["direction"])] for r in selected], axis=0)
            va_unique, pair_values = aggregate(p, trial[va])
            values.append(pair_values); masks.append(va_unique)
            selections.append({"pair_id": pair_id, "train_trial_indices": tr_trials.tolist(), "valid_trial_indices": va_trials.tolist(), "train_window_count": len(tr), "valid_window_count": len(va), "top_ks": list(TOP_KS), "selected_K": selected_k, "ranking_source": "inner_oof_only", "ranked_candidates": ranked, "inner_scores": scores, "inner_k_scores": k_scores})
            for t, probability in zip(va_unique, pair_values):
                rows.append({"pair_id": pair_id, "trial_index": int(t), "trial_key": f"{int(y[trial == t][0])}::{int(t % 4) + 1}", "y": int(y[trial == t][0]), "prob_positive": float(probability), "window_count": 5, "selected_K": selected_k})
    sums, counts = np.zeros(8), np.zeros(8, dtype=int)
    for p, va in zip(values, masks):
        sums[va] += p; counts[va] += 1
    assert np.all(counts == 4)
    p = sums / counts
    out = pd.DataFrame({"trial_index": np.arange(8), "trial_key": [f"{int(y[trial == t][0])}::{int(t % 4) + 1}" for t in range(8)], "y": [int(y[trial == t][0]) for t in range(8)], "prob_positive": p, "prediction_count": counts, "prediction_fixed05": (p >= FIXED_THRESHOLD).astype(int)})
    out.to_csv(HERE / "predictions/DEP02_window_csp_inner_topk_oof.csv", index=False)
    pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf8")
    frozen = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "hashes": frozen_hashes()}
    (HERE / "input_metadata/freeze.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf8")
    provenance = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256((HERE / "run_experiment.py").read_bytes()).hexdigest(), "python": platform.python_version(), "threads": 1, "protocol": "strict_inner_only_topk_selection", "outer_validation_labels_used_for_selection": False, "freeze": frozen, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "data_description_sha256": hashlib.sha256((ROOT / "数据/数据集说明文档.md").read_bytes()).hexdigest(), "independent_refit": True}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    pair_metrics = [metric(np.asarray([y[trial == t][0] for t in va]), p_) for p_, va in zip(values, masks)]
    result = {"experiment_id": "111_DEP02_window_csp_inner_topk_pairs", "subject": "DEP02", "pooled_oof": metric(out.y.to_numpy(), p), "pair_metrics": pair_metrics, "pair_mean": {k: float(np.mean([m[k] for m in pair_metrics])) for k in ("auc", "f1")}, "n_trials": 8, "n_windows": 40, "windows_per_trial": 5, "candidate_count": 120, "candidate_direction_count": 240, "top_ks": list(TOP_KS), "threshold": FIXED_THRESHOLD, "threshold_search": False, "k_selection": "inner OOF pooled AUC then F1 then smaller K", "outer_validation_labels_used_for_selection": False, "elapsed_seconds": time.perf_counter() - started}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
