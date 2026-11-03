import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BLOCKS = ("all", "spectral", "temporal", "PE")
CS = (0.1, 1, 10)
AGGREGATIONS = ("mean",)
COMPONENTS = (1, 2, 3)
CANDIDATES = tuple(
    ("mean", "logistic_regression", block, c, components)
    for block in BLOCKS
    for components in COMPONENTS
    for c in CS
)


def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, np.asarray(p) >= 0.5, zero_division=0))}


def block_x(x, block):
    return x if block == "all" else x[:, {"spectral": slice(0, 150), "temporal": slice(150, 600), "PE": slice(600, 720)}[block]]


def aggregate_windows(x, trial_ids, aggregation):
    assert aggregation == "mean"
    return np.asarray([x[trial_ids == trial].mean(axis=0) for trial in np.unique(trial_ids)])


def make_model(candidate):
    _, _, _, c, components = candidate
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=components, svd_solver="full", whiten=False),
        LogisticRegression(C=c, solver="liblinear", max_iter=3000, random_state=2026),
    )


def predict(x_train, y_train, train_trial_ids, x_valid, valid_trial_ids, candidate):
    aggregation, _, block, _, _ = candidate
    model = make_model(candidate)
    train_features = aggregate_windows(block_x(x_train, block), train_trial_ids, aggregation)
    model.fit(train_features, y_train)
    valid_features = aggregate_windows(block_x(x_valid, block), valid_trial_ids, aggregation)
    score = model.decision_function(valid_features)
    return 1.0 / (1.0 + np.exp(-np.clip(score, -60, 60)))


def pair_splits(y):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for p in pos:
        for n in neg:
            valid = np.sort(np.asarray([p, n]))
            yield np.setdiff1d(np.arange(len(y)), valid), valid


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
    spectral_mask = np.asarray([str(n).startswith("log_bandpower_") for n in s["feature_names"]])
    assert spectral_mask.sum() == 150 and t["X_train"].shape[1] == 450 and pe["X_train"].shape[1] == 120
    x = np.column_stack([s["X_train"][mask][:, spectral_mask], t["X_train"][mask], pe["X_train"][mask]])
    ref = pd.DataFrame({"subject": s["subject_ids"][mask], "group": s["health_groups"][mask], "trial": s["original_trial_ids"][mask], "y": s["y_train"][mask]})
    assert x.shape == (40, 720) and len(ref) == 40 and ref.subject.eq(config["target_subject"]).all()
    assert ref.groupby(["y", "trial"]).size().eq(5).all() and np.isfinite(x).all()
    ref["trial_key"] = [f"{int(y)}::{int(t)}" for y, t in zip(ref.y, ref.trial)]
    assert ref.trial_key.nunique() == 8
    assert ref.trial_key.tolist() == np.repeat(ref.trial_key.drop_duplicates().to_numpy(), 5).tolist()
    return x, ref, [str((HERE / config[k]).resolve()) for k in ("spectral_cache", "temporal_cache", "pe_cache")]


def select(x, y, trial_ids):
    inner = list(pair_splits(y))
    records = []
    for index, candidate in enumerate(CANDIDATES):
        p = np.zeros(len(y)); count = np.zeros(len(y), dtype=int)
        for tr, va in inner:
            train_trials = np.repeat(np.arange(len(tr)), 5)
            valid_trials = np.repeat(np.arange(len(va)), 5)
            value = predict(x[trials_to_rows(tr, trial_ids)], y[tr], train_trials, x[trials_to_rows(va, trial_ids)], valid_trials, candidate)
            p[va] += value; count[va] += 1
        raw = p / count
        for direction_index, direction in enumerate(("raw", "reverse")):
            pp = raw if direction == "raw" else 1 - raw
            records.append({"candidate_index": index, "candidate": candidate, "direction": direction, "direction_index": direction_index, **metric(y, pp)})
    best = min(records, key=lambda r: (-r["auc"], -r["f1"], r["candidate_index"], r["direction_index"]))
    return CANDIDATES[best["candidate_index"]], best["direction"], records


def trials_to_rows(trials, trial_ids):
    return np.flatnonzero(np.isin(trial_ids, trials))


def quota_predictions(p):
    out = np.zeros(8, dtype=int); out[np.argsort(-np.asarray(p), kind="stable")[:4]] = 1; return out


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.perf_counter(); config = json.loads((HERE / "config.json").read_text(encoding="utf8"))
    x, ref, inputs = load_subject(config)
    window_trial = ref.trial_key.to_numpy()
    trial_keys = list(dict.fromkeys(ref.trial_key.tolist()))
    trial_y = np.asarray([int(ref.loc[ref.trial_key == trial, "y"].iloc[0]) for trial in trial_keys])
    assert len(trial_keys) == 8 and trial_y.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    values, masks, rows, selections = [], [], [], []
    with threadpool_limits(limits=1):
        for pair_id, (tr, va) in enumerate(pair_splits(trial_y)):
            window_trial_index = np.asarray([trial_keys.index(t) for t in window_trial])
            train_rows = trials_to_rows(tr, window_trial_index)
            valid_rows = trials_to_rows(va, window_trial_index)
            train_trial_ids = np.repeat(np.arange(len(tr)), 5)
            valid_trial_ids = np.repeat(np.arange(len(va)), 5)
            candidate, direction, scores = select(x[train_rows], trial_y[tr], train_trial_ids)
            p = predict(x[train_rows], trial_y[tr], train_trial_ids, x[valid_rows], valid_trial_ids, candidate)
            p = p if direction == "raw" else 1 - p
            values.append(p); masks.append(va)
            selections.append({"pair_id": pair_id, "train_indices": tr.tolist(), "valid_indices": va.tolist(), "selected_candidate": candidate, "selected_direction": direction, "inner_scores": scores})
            for i, prob in zip(va, p): rows.append({"pair_id": pair_id, "trial_index": int(i), "trial_key": trial_keys[i], "y": int(trial_y[i]), "prob_positive": float(prob), "window_count": 5})
    sums, counts = np.zeros(8), np.zeros(8, dtype=int)
    for value, va in zip(values, masks): sums[va] += value; counts[va] += 1
    p = sums / counts
    trial_ref = pd.DataFrame({"trial_index": np.arange(8), "trial_key": trial_keys, "window_rows": [np.flatnonzero(window_trial == key).tolist() for key in trial_keys], "y": trial_y, "prob_positive": p, "prediction_count": counts, "prediction_quota4": quota_predictions(p), "prediction_fixed05": p >= 0.5})
    trial_ref.to_csv(HERE / "predictions/DEP02_trial_lowdim_pca_oof.csv", index=False); pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2, default=str), encoding="utf8")
    provenance = {"inputs": {path: sha(path) for path in inputs}, "sources": {str(HERE / n): sha(HERE / n) for n in ("run_experiment.py", "config.json", "冻结协议.md")}, "data_description_sha256": sha(ROOT / "数据/数据集说明文档.md"), "python": platform.python_version(), "threads": 1, "metadata_rows": 40, "feature_dimensions": config["blocks"], "trial_window_counts": {str(k): int((window_trial == k).sum()) for k in trial_keys}, "outer_validation_labels_used_for_selection": False}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    pair_metrics = [metric(trial_y[va], pv) for va, pv in zip(masks, values)]
    result = {"experiment_id": config["experiment_id"], "subject": "DEP02", "pooled_oof": {"auc": float(roc_auc_score(trial_y, p)), "f1": float(f1_score(trial_y, trial_ref.prediction_quota4, zero_division=0))}, "fixed_0.5": metric(trial_y, p), "pair_mean": {k: float(np.mean([m[k] for m in pair_metrics])) for k in ("auc", "f1")}, "pair_metrics": pair_metrics, "n_trials": 8, "n_windows": 40, "windows_per_trial": 5, "model_candidate_count": len(CANDIDATES), "candidate_count": len(CANDIDATES) * 2, "aggregations": AGGREGATIONS, "positive_quota": 4, "simultaneously_reached_0.95": False, "outer_validation_labels_used_for_selection": False, "elapsed_seconds": time.perf_counter() - started}
    result["simultaneously_reached_0.95"] = all(result["pooled_oof"][k] >= 0.95 for k in ("auc", "f1"))
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
