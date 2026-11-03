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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
N_CHANNELS = 30
TRI = np.triu_indices(N_CHANNELS, 1)
BLOCKS = ("connectivity_all", "bandpower", "concat")
AGGS = ("mean", "median")
MODELS = ("logistic_regression", "linear_svc")
CS = (0.01, 0.1, 1, 10)
CANDIDATES = tuple((agg, block, model, c) for agg in AGGS for block in BLOCKS for model in MODELS for c in CS)


def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, np.asarray(p) >= 0.5, zero_division=0))}


def quota(p):
    out = np.zeros(len(p), dtype=int)
    out[np.argsort(-np.asarray(p), kind="stable")[:4]] = 1
    return out


def load_subject(subject):
    paths = [p for p in (ROOT / "数据" / "训练集").glob("*/*.mat") if p.stem == subject]
    assert len(paths) == 1
    with h5py.File(paths[0], "r") as mat:
        chunks = []
        for key in ("EEG_data_neu", "EEG_data_pos"):
            data = np.asarray(mat[key][()], dtype=np.float64).T
            assert data.shape == (30, 50000) and np.isfinite(data).all()
            chunks.append(data.reshape(30, 20, 2500).transpose(1, 0, 2))
    x = np.concatenate(chunks)
    y = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    trial = np.r_[np.repeat(np.arange(4), 5), 4 + np.repeat(np.arange(4), 5)]
    return x, y, trial, paths[0]


def window_features(x):
    powers, conns = [], []
    for low, high in BANDS:
        z = sosfiltfilt(butter(4, [low, high], btype="bandpass", fs=250, output="sos"), x, axis=-1)
        powers.append(np.log(np.maximum(np.var(z, axis=-1), 1e-30)))
        centered = z - z.mean(axis=-1, keepdims=True)
        denom = np.sqrt(np.sum(centered * centered, axis=-1, keepdims=True))
        normalized = centered / np.maximum(denom, 1e-30)
        corr = np.einsum("nct,ndt->ncd", normalized, normalized)
        conns.append(corr[:, TRI[0], TRI[1]])
    power = np.concatenate(powers, axis=1)
    conn = np.concatenate(conns, axis=1)
    features = np.concatenate([conn, power], axis=1)
    assert features.shape == (len(x), 2325) and np.isfinite(features).all()
    return features


def block_x(x, block):
    if block == "connectivity_all": return x[:, :2175]
    if block == "bandpower": return x[:, 2175:]
    return x


def aggregate(x, trial_ids, method):
    return np.asarray([x[np.asarray(trial_ids) == t].mean(0) if method == "mean" else np.median(x[np.asarray(trial_ids) == t], axis=0) for t in np.unique(trial_ids)])


def make_model(candidate):
    _, _, name, c = candidate
    est = LogisticRegression(C=c, solver="liblinear", max_iter=3000, random_state=2026) if name == "logistic_regression" else LinearSVC(C=c, dual="auto", max_iter=5000, random_state=2026)
    return make_pipeline(StandardScaler(), est)


def predict(xtr, ytr, idtr, xva, idva, candidate):
    agg, block, _, _ = candidate
    model = make_model(candidate)
    train_agg = aggregate(block_x(xtr, block), idtr, agg)
    train_labels = np.asarray([ytr[idtr == i][0] for i in np.unique(idtr)])
    model.fit(train_agg, train_labels)
    score = model.decision_function(aggregate(block_x(xva, block), idva, agg))
    return 1 / (1 + np.exp(-np.clip(score, -60, 60)))


def pair_splits(y):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for p in pos:
        for n in neg:
            va = np.sort(np.asarray([p, n]))
            yield np.setdiff1d(np.arange(len(y)), va), va


def select(x, y, ids):
    unique = np.unique(ids)
    trial_y = np.asarray([y[ids == i][0] for i in unique])
    folds = []
    for tr_i, va_i in pair_splits(trial_y):
        tr_ids, va_ids = unique[tr_i], unique[va_i]
        tr = np.flatnonzero(np.isin(ids, tr_ids)); va = np.flatnonzero(np.isin(ids, va_ids))
        local_tr = np.searchsorted(tr_ids, ids[tr]); local_va = np.searchsorted(va_ids, ids[va])
        folds.append((tr, va, local_tr, local_va, va_ids))
    trial_y = np.asarray([y[ids == i][0] for i in unique])
    scores = []
    for ci, candidate in enumerate(CANDIDATES):
        raw = np.zeros(len(unique)); count = np.zeros(len(unique), dtype=int)
        for tr, va, local_tr, local_va, va_ids in folds:
            pp = predict(x[tr], y[tr], local_tr, x[va], local_va, candidate)
            for j, tid in enumerate(va_ids): raw[tid] += pp[j]; count[tid] += 1
        raw /= count
        for di, direction in enumerate(("raw", "reverse")):
            p = raw if direction == "raw" else 1 - raw
            scores.append({"candidate_index": ci, "candidate": candidate, "direction": direction, "direction_index": di, **metric(y, p)})
    return min(scores, key=lambda r: (-r["auc"], -r["f1"], r["candidate_index"], r["direction_index"])), scores


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    x, yw, trial, source = load_subject("DEP02")
    features = window_features(x)
    keys = np.asarray([f"{int(y)}::{int(t) + 1}" for y, t in zip(yw, trial)], dtype=object)
    unique = list(dict.fromkeys(keys)); assert len(unique) == 8 and all((keys == k).sum() == 5 for k in unique)
    ids = np.asarray([unique.index(k) for k in keys])
    yt = np.asarray([yw[ids == i][0] for i in range(8)])
    values, masks, rows, selections = [], [], [], []
    with threadpool_limits(limits=1):
        for pair_id, (tr_i, va_i) in enumerate(pair_splits(yt)):
            tr_ids, va_ids = tr_i, va_i
            tr = np.flatnonzero(np.isin(ids, tr_ids)); va = np.flatnonzero(np.isin(ids, va_ids))
            local_tr = np.searchsorted(tr_ids, ids[tr]); local_va = np.searchsorted(va_ids, ids[va])
            chosen, scores = select(features[tr], yw[tr], local_tr)
            p = predict(features[tr], yt[tr_ids], local_tr, features[va], local_va, tuple(chosen["candidate"]))
            if chosen["direction"] == "reverse": p = 1 - p
            trial_p = np.asarray([p[local_va == j].mean() for j in range(len(va_ids))])
            values.append((va_ids, trial_p)); masks.append(va_ids)
            selections.append({"pair_id": pair_id, "train_indices": tr_ids.tolist(), "valid_indices": va_ids.tolist(), "selected_candidate": chosen["candidate"], "selected_direction": chosen["direction"], "inner_scores": scores, "selection_source": "inner OOF only"})
            for t, prob in zip(va_ids, trial_p): rows.append({"pair_id": pair_id, "trial_index": int(t), "trial_key": unique[t], "y": int(yt[t]), "prob_positive": float(prob), "window_count": 5})
    sums, counts = np.zeros(8), np.zeros(8, dtype=int)
    for va, p in values: sums[va] += p; counts[va] += 1
    p = sums / counts
    out = pd.DataFrame({"trial_index": np.arange(8), "trial_key": unique, "y": yt, "prob_positive": p, "prediction_count": counts, "prediction_quota4": quota(p), "prediction_fixed05": p >= .5, "window_rows": [np.flatnonzero(ids == i).tolist() for i in range(8)]})
    out.to_csv(HERE / "predictions/DEP02_trial_connectivity_oof.csv", index=False)
    pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2, default=str), encoding="utf8")
    prov = {"source": str(source), "source_sha256": sha(source), "data_description_sha256": sha(ROOT / "数据/数据集说明文档.md"), "script_sha256": sha(HERE / "run_experiment.py"), "python": platform.python_version(), "threads": 1, "metadata_rows": 40, "feature_dimensions": {"connectivity_all": 2175, "bandpower": 150, "concat": 2325}, "trial_window_counts": {k: int((keys == k).sum()) for k in unique}, "outer_validation_labels_used_for_selection": False, "connectivity": "bandpass Pearson correlation per window"}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf8")
    pair_metrics = [metric(yt[va], pv) for va, pv in values]
    result = {"experiment_id": "124_DEP02_connectivity_pairs", "subject": "DEP02", "pooled_oof": {"auc": float(roc_auc_score(yt, p)), "f1": float(f1_score(yt, out.prediction_quota4, zero_division=0))}, "fixed_0.5": metric(yt, p), "pair_mean": {k: float(np.mean([m[k] for m in pair_metrics])) for k in ("auc", "f1")}, "pair_metrics": pair_metrics, "n_trials": 8, "n_windows": 40, "windows_per_trial": 5, "candidate_count": 96, "model_candidate_count": 48, "feature_dimensions": {"connectivity_all": 2175, "bandpower": 150, "concat": 2325}, "positive_quota": 4, "outer_validation_labels_used_for_selection": False, "simultaneously_reached_0.95": False, "elapsed_seconds": time.perf_counter() - started}
    result["simultaneously_reached_0.95"] = result["pooled_oof"]["auc"] >= .95 and result["pooled_oof"]["f1"] >= .95
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    report = f"# 实验124：DEP02频带通道连接严格配对\n\n- 连接实现：每窗口带通Pearson相关上三角（未使用标签、无跨折拟合），并保留log bandpower block。\n- pooled OOF AUC/F1：**{result['pooled_oof']['auc']:.6f} / {result['pooled_oof']['f1']:.6f}**\n- fixed 0.5 AUC/F1：**{result['fixed_0.5']['auc']:.6f} / {result['fixed_0.5']['f1']:.6f}**；pair均值 AUC/F1：**{result['pair_mean']['auc']:.6f} / {result['pair_mean']['f1']:.6f}**。\n- 结论：{'达到' if result['simultaneously_reached_0.95'] else '未达到'}双指标0.95要求。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
