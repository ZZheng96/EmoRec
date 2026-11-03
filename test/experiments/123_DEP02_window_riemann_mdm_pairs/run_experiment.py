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
from sklearn.metrics import f1_score, roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
LAMBDAS = (0.0, 0.1, 0.3, 0.5)
BLOCKS = ("all", "band_1", "band_2", "band_3", "band_4", "band_5")
DISTANCES = ("riemann", "euclidean_logm")
CANDIDATES = tuple((lam, block, distance) for lam in LAMBDAS for block in BLOCKS for distance in DISTANCES)
FIXED_THRESHOLD = 0.5
POSITIVE_QUOTA = 4


def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, np.asarray(p) >= FIXED_THRESHOLD, zero_division=0))}


def quota_predictions(p):
    out = np.zeros(len(p), dtype=int)
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
        bands.append(z @ z.transpose(0, 2, 1) / z.shape[-1])
    return np.stack(bands, axis=1)


def shrink(cov, lam):
    scale = np.trace(cov, axis1=2, axis2=3) / 30.0
    eye = np.eye(30)[None, None]
    out = (1 - lam) * cov + lam * scale[:, :, None, None] * eye
    return out / np.maximum(np.trace(out, axis1=2, axis2=3)[:, :, None, None], 1e-30)


def logm(cov):
    vals, vecs = eigh(cov)
    return np.einsum("...ik,...k,...jk->...ij", vecs, np.log(np.maximum(vals, 1e-30)), vecs)


def karcher_mean(cov, iterations=20, step=0.5):
    mean = cov.mean(axis=0)
    for _ in range(iterations):
        vals, vecs = eigh(mean)
        root = np.einsum("bci,bi,bji->bcj", vecs, np.sqrt(np.maximum(vals, 1e-30)), vecs)
        invroot = np.einsum("bci,bi,bji->bcj", vecs, 1 / np.sqrt(np.maximum(vals, 1e-30)), vecs)
        whitened = np.einsum("bij,nbjk,bkl->nbil", invroot, cov, invroot)
        tangent = logm(whitened)
        delta = np.einsum("bij,nbjk,bkl->nbil", root, tangent, root).mean(axis=0)
        mean = mean + step * delta
        mean = (mean + mean.transpose(0, 2, 1)) / 2
    return mean


def distances(cov, prototypes, distance):
    if distance == "euclidean_logm":
        values = logm(cov)
        refs = logm(prototypes)
        return np.sqrt(np.sum((values - refs[None]) ** 2, axis=(2, 3)))
    vals, vecs = eigh(prototypes)
    root = np.einsum("bci,bi,bji->bcj", vecs, np.sqrt(np.maximum(vals, 1e-30)), vecs)
    invroot = np.einsum("bci,bi,bji->bcj", vecs, 1 / np.sqrt(np.maximum(vals, 1e-30)), vecs)
    whitened = np.einsum("bij,nbjk,bkl->nbil", invroot, cov, invroot)
    return np.sqrt(np.sum(logm(whitened) ** 2, axis=(2, 3)))


def build_prediction_cache(cov_train, y_train, cov_valid):
    """Fit each lambda's training-only geometry once and cache valid transforms/distances."""
    cache = {}
    for lam in LAMBDAS:
        train = shrink(cov_train, lam)
        valid = shrink(cov_valid, lam)
        prototypes = np.stack([karcher_mean(train[y_train == cls]) for cls in (0, 1)])
        valid_logm = logm(valid)
        prototype_logm = logm(prototypes)
        riemann = np.stack([
            distances(valid, prototypes[cls], "riemann") for cls in (0, 1)
        ], axis=0)
        euclidean = np.stack([
            np.sqrt(np.sum((valid_logm - prototype_logm[cls][None]) ** 2, axis=(2, 3)))
            for cls in (0, 1)
        ], axis=0)
        cache[lam] = {"riemann": riemann, "euclidean_logm": euclidean}
    return cache


def predict(cov_train, y_train, cov_valid, candidate, cache=None):
    lam, block, distance = candidate
    if cache is None:
        cache = build_prediction_cache(cov_train, y_train, cov_valid)
    class_distances = cache[lam][distance]
    band_score = class_distances[0] - class_distances[1]
    selected = band_score.mean(axis=1) if block == "all" else band_score[:, int(block[-1]) - 1]
    return 1 / (1 + np.exp(-np.clip(selected, -60, 60)))


def trial_pairs(trial):
    trial = np.asarray(trial)
    for p in np.flatnonzero(trial >= 4):
        for n in np.flatnonzero(trial < 4):
            va = np.sort(np.asarray([p, n]))
            yield np.setdiff1d(np.arange(len(trial)), va), va


def select_inner(cov, y, trial_ids):
    scores = []
    unique = np.unique(trial_ids)
    fold_data = []
    for tr_pos, va_pos in trial_pairs(unique):
        tr_trials, va_trials = unique[tr_pos], unique[va_pos]
        tr = np.flatnonzero(np.isin(trial_ids, tr_trials))
        va = np.flatnonzero(np.isin(trial_ids, va_trials))
        fold_data.append((va_trials, trial_ids[va], y, build_prediction_cache(cov[tr], y[tr], cov[va])))
    for index, candidate in enumerate(CANDIDATES):
        pp, yy = [], []
        for va_trials, va_trial_ids, all_y, fold_cache in fold_data:
            p = predict(None, None, None, candidate, fold_cache)
            for t in va_trials:
                pp.append(float(p[va_trial_ids == t].mean()))
                yy.append(int(all_y[trial_ids == t][0]))
        for direction_index, direction in enumerate(("raw", "reverse")):
            values = np.asarray(pp) if direction == "raw" else 1 - np.asarray(pp)
            scores.append({"candidate_index": index, "candidate": candidate, "direction": direction, "direction_index": direction_index, **metric(np.asarray(yy), values)})
    return sorted(scores, key=lambda r: (-r["auc"], -r["f1"], r["candidate_index"], r["direction_index"]))


def main():
    started = time.perf_counter()
    x, y, trial, source = load_subject("DEP02")
    cov = covariances(x)
    assert cov.shape == (40, 5, 30, 30)
    values, rows, selections = [], [], []
    with threadpool_limits(limits=1):
        for pair_id, (tr_trials, va_trials) in enumerate(trial_pairs(np.arange(8))):
            tr = np.flatnonzero(np.isin(trial, tr_trials))
            va = np.flatnonzero(np.isin(trial, va_trials))
            ranked = select_inner(cov[tr], y[tr], trial[tr])
            chosen = ranked[0]
            outer_cache = build_prediction_cache(cov[tr], y[tr], cov[va])
            p = predict(None, None, None, tuple(chosen["candidate"]), outer_cache)
            if chosen["direction"] == "reverse": p = 1 - p
            trial_p = np.asarray([p[trial[va] == t].mean() for t in va_trials])
            values.append((va_trials, trial_p))
            selections.append({"pair_id": pair_id, "train_trial_indices": tr_trials.tolist(), "valid_trial_indices": va_trials.tolist(), "train_window_count": int(len(tr)), "valid_window_count": int(len(va)), "selected_candidate": chosen["candidate"], "selected_direction": chosen["direction"], "inner_scores": ranked, "selection_source": "inner OOF trial probabilities only; training-fold prototypes only"})
            for t, probability in zip(va_trials, trial_p):
                rows.append({"pair_id": pair_id, "trial_index": int(t), "trial_key": f"{int(y[trial == t][0])}::{int(t % 4) + 1}", "y": int(y[trial == t][0]), "prob_positive": float(probability), "window_count": 5})
    sums, counts = np.zeros(8), np.zeros(8, dtype=int)
    for keys, probs in values: sums[keys] += probs; counts[keys] += 1
    assert np.all(counts == 4)
    yy = np.asarray([y[trial == t][0] for t in range(8)])
    p = sums / counts
    out = pd.DataFrame({"trial_index": np.arange(8), "trial_key": [f"{yy[t]}::{t % 4 + 1}" for t in range(8)], "y": yy, "prob_positive": p, "prediction_count": counts})
    out["prediction_quota4"] = quota_predictions(p)
    out["prediction_fixed05"] = (p >= FIXED_THRESHOLD).astype(int)
    (HERE / "predictions/DEP02_window_riemann_mdm_pairs_oof.csv").write_text(out.to_csv(index=False), encoding="utf8")
    pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf8")
    frozen = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "hashes": {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("run_experiment.py", "config.json", "冻结协议.md")}}
    (HERE / "input_metadata/freeze.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf8")
    provenance = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256((HERE / "run_experiment.py").read_bytes()).hexdigest(), "python": platform.python_version(), "threads": 1, "compute": "single-thread CPU", "outer_validation_labels_used_for_selection": False, "prototype_fit_on_training_windows_only": True, "shrinkage_fit_on_training_windows_only": True, "freeze": frozen, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "data_description_sha256": hashlib.sha256((ROOT / "数据/数据集说明文档.md").read_bytes()).hexdigest()}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    pair_metrics = [metric(np.asarray([yy[t] for t in keys]), probs) for keys, probs in values]
    result = {"experiment_id": "123_DEP02_window_riemann_mdm_pairs", "subject": "DEP02", "pooled_oof": {"auc": float(roc_auc_score(yy, p)), "f1": float(f1_score(yy, out.prediction_quota4, zero_division=0))}, "fixed_0.5": metric(yy, p), "pair_mean": {k: float(np.mean([m[k] for m in pair_metrics])) for k in ("auc", "f1")}, "pair_metrics": pair_metrics, "n_trials": 8, "n_windows": 40, "windows_per_trial": 5, "candidate_count": len(CANDIDATES), "candidate_direction_count": len(CANDIDATES) * 2, "positive_quota": 4, "threshold": 0.5, "outer_validation_labels_used_for_selection": False, "elapsed_seconds": time.perf_counter() - started}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    report = f"# 实验123：DEP02窗口级Riemannian MDM严格配对\n\n## 真实结果\n\n- pooled OOF（top-4 quota）AUC **{result['pooled_oof']['auc']:.6f}**，主F1 **{result['pooled_oof']['f1']:.6f}**。\n- fixed 0.5：AUC **{result['fixed_0.5']['auc']:.6f}**，F1 **{result['fixed_0.5']['f1']:.6f}**。\n- 16个outer pair均值：AUC **{result['pair_mean']['auc']:.6f}**，F1 **{result['pair_mean']['f1']:.6f}**。\n- 候选{len(CANDIDATES)}个，raw/reverse共{len(CANDIDATES)*2}项；单线程CPU。\n\n## 协议\n\n原始DEP02 40窗口、五频带30×30协方差；outer 4×4、inner 3×3按物理trial严格配对，验证trial排序后按窗口概率均值聚合。每训练折窗口协方差使用lambda={list(LAMBDAS)}收缩并trace归一化；每band以训练类Karcher均值（20次、步长0.5、算术均值初始化）作Riemannian prototype。候选为lambda×all/band_1..band_5×riemann/euclidean_logm；inner按trial均值概率AUC→F1→序号→方向选择，outer标签禁用。\n\n单线程CPU运行，完整协议/config/报告/metrics/predictions/selections/provenance及测试均保存。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
