import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BLOCKS = ("all", "spectral", "temporal", "PE")
DISTANCES = ("euclidean", "cosine", "diagonal_mahalanobis")
NORMALIZATIONS = ("raw", "StandardScaler")
CANDIDATES = tuple((b, d, n) for b in BLOCKS for d in DISTANCES for n in NORMALIZATIONS)
FIXED_THRESHOLD = 0.5
POSITIVE_QUOTA = 4


def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, np.asarray(p) >= FIXED_THRESHOLD, zero_division=0))}


def quota_predictions(p):
    out = np.zeros(8, dtype=int)
    out[np.argsort(-np.asarray(p), kind="stable")[:POSITIVE_QUOTA]] = 1
    return out


def block_x(x, block):
    return x if block == "all" else x[:, {"spectral": slice(0, 150), "temporal": slice(150, 600), "PE": slice(600, 720)}[block]]


def load_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        return {k: cache[k] for k in ("X_train", "subject_ids", "health_groups", "original_trial_ids", "y_train", "feature_names")}


def load_subject(config):
    caches = [load_cache((HERE / config[k]).resolve()) for k in ("spectral_cache", "temporal_cache", "pe_cache")]
    for other in caches[1:]:
        for key in ("subject_ids", "health_groups", "original_trial_ids", "y_train"):
            np.testing.assert_array_equal(caches[0][key], other[key])
    s, t, pe = caches
    mask = s["subject_ids"] == config["target_subject"]
    assert s["X_train"].shape[1] == 210 and t["X_train"].shape[1] == 450 and pe["X_train"].shape[1] == 120
    spectral_mask = np.asarray([str(n).startswith("log_bandpower_") for n in s["feature_names"]])
    assert spectral_mask.sum() == 150
    x = np.column_stack([s["X_train"][mask][:, spectral_mask], t["X_train"][mask], pe["X_train"][mask]])
    ref = pd.DataFrame({"subject": s["subject_ids"][mask], "group": s["health_groups"][mask], "trial": s["original_trial_ids"][mask], "y": s["y_train"][mask]})
    assert x.shape == (40, 720) and ref.groupby(["y", "trial"]).size().eq(5).all()
    return x, ref, [str((HERE / config[k]).resolve()) for k in ("spectral_cache", "temporal_cache", "pe_cache")]


def trial_pairs(y):
    pos, neg = np.flatnonzero(np.asarray(y) == 1), np.flatnonzero(np.asarray(y) == 0)
    for p in pos:
        for n in neg:
            valid = np.asarray([p, n])
            yield np.setdiff1d(np.arange(len(y)), valid), valid


def centroid_predict(x_train, y_train, x_valid, candidate):
    block, distance, normalization = candidate
    a, b = block_x(x_train, block), block_x(x_valid, block)
    if normalization == "StandardScaler":
        scaler = StandardScaler().fit(a)
        a, b = scaler.transform(a), scaler.transform(b)
    pos, neg = a[y_train == 1], a[y_train == 0]
    c_pos, c_neg = pos.mean(axis=0), neg.mean(axis=0)
    if distance == "euclidean":
        d_pos = np.linalg.norm(b - c_pos, axis=1)
        d_neg = np.linalg.norm(b - c_neg, axis=1)
    elif distance == "cosine":
        def cosine(u, v):
            return 1.0 - (u @ v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v), 1e-12)
        d_pos, d_neg = cosine(b, c_pos), cosine(b, c_neg)
    else:
        variance = np.var(a, axis=0) + 1e-8
        d_pos = np.sqrt(np.sum((b - c_pos) ** 2 / variance, axis=1))
        d_neg = np.sqrt(np.sum((b - c_neg) ** 2 / variance, axis=1))
    score = d_neg - d_pos
    return 1.0 / (1.0 + np.exp(-np.clip(score, -60, 60)))


def select_inner(x, y):
    inner = list(trial_pairs(y)); records = []
    for index, candidate in enumerate(CANDIDATES):
        values = [centroid_predict(x[tr], y[tr], x[va], candidate) for tr, va in inner]
        p = np.zeros(len(y)); count = np.zeros(len(y), dtype=int)
        for value, (_, va) in zip(values, inner): p[va] += value; count[va] += 1
        raw = p / count
        for direction_index, direction in enumerate(("raw", "reverse")):
            pp = raw if direction == "raw" else 1 - raw
            records.append({"candidate_index": index, "candidate": candidate, "direction": direction, "direction_index": direction_index, **metric(y, pp)})
    best = min(records, key=lambda r: (-r["auc"], -r["f1"], r["candidate_index"], r["direction_index"]))
    return CANDIDATES[best["candidate_index"]], best["direction"], records


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.perf_counter(); config = json.loads((HERE / "config.json").read_text(encoding="utf8"))
    x, ref, inputs = load_subject(config)
    keys = np.asarray([f"{int(a)}::{int(b)}" for a, b in zip(ref.y, ref.trial)], dtype=object)
    unique = list(dict.fromkeys(keys)); assert all((keys == k).sum() == 5 for k in unique)
    trial_x = np.asarray([x[keys == k].mean(axis=0) for k in unique]); trial_y = np.asarray([ref.y.to_numpy()[keys == k][0] for k in unique])
    trial_ref = pd.DataFrame([{"trial_index": i, "trial_key": k, "y": int(trial_y[i]), "window_rows": np.flatnonzero(keys == k).tolist()} for i, k in enumerate(unique)])
    values, masks, rows, selections = [], [], [], []
    with threadpool_limits(limits=1):
        for pair_id, (tr, va) in enumerate(trial_pairs(trial_y)):
            candidate, direction, scores = select_inner(trial_x[tr], trial_y[tr])
            p = centroid_predict(trial_x[tr], trial_y[tr], trial_x[va], candidate)
            if direction == "reverse": p = 1 - p
            values.append(p); masks.append(va)
            selections.append({"pair_id": pair_id, "train_indices": tr.tolist(), "valid_indices": va.tolist(), "train_trial_count": 6, "valid_trial_count": 2, "selected_candidate": candidate, "selected_direction": direction, "inner_scores": scores})
            for i, prob in zip(va, p): rows.append({"pair_id": pair_id, "trial_index": int(i), "trial_key": unique[i], "y": int(trial_y[i]), "prob_positive": float(prob), "window_count": 5})
    sums, counts = np.zeros(8), np.zeros(8, dtype=int)
    for value, va in zip(values, masks): sums[va] += value; counts[va] += 1
    p = sums / counts; trial_ref["prob_positive"] = p; trial_ref["prediction_count"] = counts; trial_ref["prediction_quota4"] = quota_predictions(p); trial_ref["prediction_fixed05"] = (p >= .5).astype(int)
    trial_ref.to_csv(HERE / "predictions/DEP02_trial_centroid_distance_oof.csv", index=False); pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf8")
    provenance = {"inputs": {path: sha(path) for path in inputs}, "sources": {str(HERE / n): sha(HERE / n) for n in ("run_experiment.py", "config.json", "冻结协议.md")}, "data_description_sha256": sha(ROOT / "数据/数据集说明文档.md"), "python": platform.python_version(), "threads": 1, "metadata_rows": 40, "trial_window_counts": {k: int((keys == k).sum()) for k in unique}, "outer_validation_labels_used_for_selection": False}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    pair_metrics = [metric(trial_y[va], pv) for va, pv in zip(masks, values)]
    result = {"experiment_id": config["experiment_id"], "subject": "DEP02", "pooled_oof": {"auc": float(roc_auc_score(trial_y, p)), "f1": float(f1_score(trial_y, trial_ref.prediction_quota4, zero_division=0))}, "fixed_0.5": metric(trial_y, p), "pair_mean": {k: float(np.mean([m[k] for m in pair_metrics])) for k in ("auc", "f1")}, "pair_metrics": pair_metrics, "n_trials": 8, "n_windows": 40, "windows_per_trial": 5, "feature_dimensions": config["blocks"], "candidate_count": len(CANDIDATES) * 2, "positive_quota": 4, "outer_validation_labels_used_for_selection": False, "simultaneously_reached_0.95": bool(result if False else False), "elapsed_seconds": time.perf_counter() - started}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    status = "达到" if result["pooled_oof"]["auc"] >= .95 and result["pooled_oof"]["f1"] >= .95 else "未达到"
    (HERE / "实验报告.md").write_text(f"# 实验117：DEP02 trial质心距离配对OOF\n\n- 主 pooled AUC：**{result['pooled_oof']['auc']:.6f}**\n- 主F1（预冻结top-4 quota）：**{result['pooled_oof']['f1']:.6f}**\n- fixed 0.5 F1：**{result['fixed_0.5']['f1']:.6f}**；pair均值 AUC/F1：**{result['pair_mean']['auc']:.6f} / {result['pair_mean']['f1']:.6f}**。\n- 结论：双指标0.95要求**{status}**；若任一指标低于0.95，不宣称达标。\n", encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
