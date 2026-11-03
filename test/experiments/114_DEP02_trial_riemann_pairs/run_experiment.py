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
CS = (0.01, 0.1, 1, 10, 100)
BLOCKS = ("all", "band0", "band1", "band2", "band3", "band4")
CANDIDATES = tuple((m, block, c) for m in ("logistic_regression", "linear_svc") for block in BLOCKS for c in CS)
FIXED_THRESHOLD = 0.5
POSITIVE_QUOTA = 4
TRIU = np.triu_indices(30)


def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, np.asarray(p) >= FIXED_THRESHOLD, zero_division=0))}


def quota_predictions(p):
    out = np.zeros(8, dtype=int)
    out[np.argsort(-np.asarray(p), kind="stable")[:POSITIVE_QUOTA]] = 1
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
        cov = z @ z.transpose(0, 2, 1) / z.shape[-1]
        scale = np.trace(cov, axis1=1, axis2=2)[:, None, None]
        bands.append(cov / np.maximum(scale, 1e-30) + 1e-8 * np.eye(30))
    return np.stack(bands, axis=1)


def aggregate_trials(cov, y, trial):
    return np.asarray([cov[trial == t].mean(axis=0) for t in range(8)]), np.asarray([y[trial == t][0] for t in range(8)])


def logeuclidean_features(cov_train, cov_valid):
    all_cov = np.concatenate([cov_train, cov_valid])
    logs = []
    for band in range(5):
        vals, vecs = eigh(all_cov[:, band])
        logs.append(np.einsum("nci,ni,nji->ncj", vecs, np.log(np.maximum(vals, 1e-30)), vecs))
    logs = np.stack(logs, axis=1)
    reference = logs[:len(cov_train)].mean(axis=0)
    centered = logs - reference[None]
    return centered[:, :, TRIU[0], TRIU[1]][:len(cov_train)].reshape(len(cov_train), -1), centered[:, :, TRIU[0], TRIU[1]][len(cov_train):].reshape(len(cov_valid), -1)


def select_block(values, block):
    if block == "all":
        return values
    return values[:, int(block[-1]) * 465:(int(block[-1]) + 1) * 465]


def predict(cov_train, y_train, cov_valid, candidate):
    model_name, block, c = candidate
    train, valid = logeuclidean_features(cov_train, cov_valid)
    train, valid = select_block(train, block), select_block(valid, block)
    estimator = LogisticRegression(C=c, solver="liblinear", max_iter=3000, random_state=2026) if model_name == "logistic_regression" else LinearSVC(C=c, dual="auto", max_iter=5000, random_state=2026)
    model = make_pipeline(StandardScaler(), estimator).fit(train, y_train)
    score = model.decision_function(valid)
    return 1 / (1 + np.exp(-np.clip(score, -60, 60)))


def trial_pairs(trial):
    trial = np.asarray(trial)
    for p in np.flatnonzero(trial >= 4):
        for n in np.flatnonzero(trial < 4):
            va = np.asarray([p, n])
            yield np.setdiff1d(np.arange(len(trial)), va), va


def select_inner(cov, y, trial_ids):
    scores = []
    for index, candidate in enumerate(CANDIDATES):
        folds = [(predict(cov[tr], y[tr], cov[va], candidate), va) for tr, va in trial_pairs(trial_ids)]
        for direction_index, direction in enumerate(("raw", "reverse")):
            pp, yy = [], []
            for value, va in folds:
                pp.extend((value if direction == "raw" else 1 - value).tolist())
                yy.extend(y[va].tolist())
            scores.append({"candidate_index": index, "candidate": candidate, "direction": direction, "direction_index": direction_index, **metric(np.asarray(yy), np.asarray(pp))})
    return sorted(scores, key=lambda r: (-r["auc"], -r["f1"], r["candidate_index"], r["direction_index"]))


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
            p = predict(cov[tr], y[tr], cov[va], tuple(chosen["candidate"]))
            if chosen["direction"] == "reverse": p = 1 - p
            values.append((va, p))
            selections.append({"pair_id": pair_id, "train_trial_indices": tr.tolist(), "valid_trial_indices": va.tolist(), "train_trial_count": 6, "valid_trial_count": 2, "raw_window_count": 40, "selected_candidate": chosen["candidate"], "selected_direction": chosen["direction"], "inner_scores": ranked, "selection_source": "inner OOF trial probabilities only; train-fold log-Euclidean references"})
            for t, probability in zip(va, p):
                rows.append({"pair_id": pair_id, "trial_index": int(t), "trial_key": f"{int(y[t])}::{int(t % 4) + 1}", "y": int(y[t]), "prob_positive": float(probability), "window_count": 5})
    sums, counts = np.zeros(8), np.zeros(8, dtype=int)
    for va, pv in values: sums[va] += pv; counts[va] += 1
    p = sums / counts
    out = pd.DataFrame({"trial_index": np.arange(8), "trial_key": [f"{y[t]}::{t % 4 + 1}" for t in range(8)], "y": y, "prob_positive": p, "prediction_count": counts})
    out["prediction_quota4"] = quota_predictions(p)
    out["prediction_fixed05"] = (p >= FIXED_THRESHOLD).astype(int)
    out.to_csv(HERE / "predictions/DEP02_trial_riemann_oof.csv", index=False)
    pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf8")
    frozen = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "hashes": frozen_hashes()}
    (HERE / "input_metadata/freeze.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf8")
    provenance = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256((HERE / "run_experiment.py").read_bytes()).hexdigest(), "python": platform.python_version(), "threads": 1, "protocol": "trial_mean_five_band_log_euclidean_spd_nested_pairs", "outer_validation_labels_used_for_selection": False, "reference_fit_on_training_trials_only": True, "standardization_fit_on_training_features_only": True, "freeze": frozen, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "data_description_sha256": hashlib.sha256((ROOT / "数据/数据集说明文档.md").read_bytes()).hexdigest()}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    pair_metrics = [metric(y[va], pv) for va, pv in values]
    result = {"experiment_id": "114_DEP02_trial_riemann_pairs", "subject": "DEP02", "pooled_oof": {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, out.prediction_quota4, zero_division=0))}, "fixed_0.5": metric(y, p), "pair_mean": {k: float(np.mean([m[k] for m in pair_metrics])) for k in ("auc", "f1")}, "pair_metrics": pair_metrics, "n_trials": 8, "n_windows": 40, "windows_per_trial": 5, "feature_dimensions": {"per_band": 465, "all": 2325}, "candidate_count": len(CANDIDATES), "positive_quota": 4, "quota_search": False, "threshold": 0.5, "threshold_search": False, "outer_validation_labels_used_for_selection": False, "elapsed_seconds": time.perf_counter() - started}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    report = f"# 实验114：DEP02 trial-level五频带Riemannian log-Euclidean表示\n\n- pooled OOF AUC：**{result['pooled_oof']['auc']:.6f}**\n- 主F1（预冻结top-4 quota）：**{result['pooled_oof']['f1']:.6f}**\n- fixed 0.5 F1：**{result['fixed_0.5']['f1']:.6f}**；pair均值 AUC/F1：**{result['pair_mean']['auc']:.6f} / {result['pair_mean']['f1']:.6f}**。\n- 结论：{'达到' if result['pooled_oof']['auc'] >= .95 and result['pooled_oof']['f1'] >= .95 else '未达到'}双指标0.95要求。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
