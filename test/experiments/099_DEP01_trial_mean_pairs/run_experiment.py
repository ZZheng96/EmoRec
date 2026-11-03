import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score, roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("e097", HERE.parent / "097_DEP01_nonlinear_pairs" / "run_experiment.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

BLOCKS = ("all", "spectral", "temporal", "PE")
CS = (0.01, 0.1, 1, 10, 100)
CANDIDATES = tuple(
    [("linear_svc", b, c, None) for b in BLOCKS for c in CS]
    + [("logistic_regression", b, c, None) for b in BLOCKS for c in CS]
    + [("rbf_svc", b, c, g) for b in BLOCKS for c in CS for g in ("scale", 0.01)]
)


def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, p >= 0.5, zero_division=0))}


def block_x(x, block):
    return x if block == "all" else x[:, {"spectral": slice(0, 150), "temporal": slice(150, 600), "PE": slice(600, 720)}[block]]


def make_model(candidate):
    name, _, c, gamma = candidate
    if name == "linear_svc":
        model = LinearSVC(C=c, dual="auto", max_iter=5000, random_state=2026)
    elif name == "logistic_regression":
        model = LogisticRegression(C=c, solver="liblinear", max_iter=3000, random_state=2026)
    else:
        model = SVC(kernel="rbf", C=c, gamma=gamma)
    return make_pipeline(StandardScaler(), model)


def predict(x_train, y_train, x_valid, candidate):
    model = make_model(candidate)
    model.fit(block_x(x_train, candidate[1]), y_train)
    score = model.decision_function(block_x(x_valid, candidate[1]))
    return 1.0 / (1.0 + np.exp(-np.clip(score, -60, 60)))


def pair_splits(y):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    for p in pos:
        for n in neg:
            valid = np.asarray([p, n])
            train = np.setdiff1d(np.arange(len(y)), valid)
            yield train, valid


def select(x, y):
    records = []
    inner = list(pair_splits(y))
    for index, candidate in enumerate(CANDIDATES):
        values = [predict(x[tr], y[tr], x[va], candidate) for tr, va in inner]
        p = np.zeros(len(y), dtype=float)
        count = np.zeros(len(y), dtype=int)
        for value, (_, va) in zip(values, inner):
            p[va] += value
            count[va] += 1
        p /= count
        records.append({"candidate_index": index, "candidate": candidate, **metric(y, p)})
    best = max(records, key=lambda r: (r["auc"], r["f1"], -r["candidate_index"]))
    return CANDIDATES[best["candidate_index"]], records


def main():
    start = time.perf_counter()
    config = json.loads((HERE / "config.json").read_text(encoding="utf8"))
    x, ref = base.load_subject({**config, "target_subject": "DEP01"})
    y = ref.y.to_numpy()
    keys = np.asarray([f"{int(a)}::{int(b)}" for a, b in zip(y, ref.trial)], dtype=object)
    unique = list(dict.fromkeys(keys))
    assert x.shape == (40, 720) and len(unique) == 8
    trial_x = np.asarray([x[keys == key].mean(axis=0) for key in unique])
    trial_y = np.asarray([y[keys == key][0] for key in unique])
    trial_ref = pd.DataFrame([{"trial_index": i, "trial_key": key, "y": int(trial_y[i]), "window_rows": np.flatnonzero(keys == key).tolist()} for i, key in enumerate(unique)])
    values, masks, rows, selections = [], [], [], []
    with threadpool_limits(limits=1):
        for pair_id, (tr, va) in enumerate(pair_splits(trial_y)):
            candidate, scores = select(trial_x[tr], trial_y[tr])
            values.append(predict(trial_x[tr], trial_y[tr], trial_x[va], candidate))
            masks.append(va)
            selections.append({"pair_id": pair_id, "train_indices": tr.tolist(), "valid_indices": va.tolist(), "selected_candidate": candidate, "inner_scores": scores})
            for i, p in zip(va, values[-1]):
                rows.append({"pair_id": pair_id, "trial_index": int(i), "trial_key": unique[i], "y": int(trial_y[i]), "prob_positive": float(p)})
    sums = np.zeros(8)
    counts = np.zeros(8, dtype=int)
    for value, va in zip(values, masks):
        sums[va] += value
        counts[va] += 1
    assert np.all(counts == 4)
    p = sums / counts
    trial_ref["prob_positive"] = p
    trial_ref["prediction_count"] = counts
    trial_ref.to_csv(HERE / "predictions/DEP01_trial_mean_oof.csv", index=False)
    pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf8")
    (HERE / "input_metadata/trials.json").write_text(trial_ref.to_json(orient="records", force_ascii=False, indent=2), encoding="utf8")
    inputs = {name: (HERE / "../097_DEP01_nonlinear_pairs" / config[name]).resolve() for name in ("spectral_cache", "temporal_cache", "pe_cache")}
    provenance = {"sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE / "run_experiment.py", HERE / "config.json", HERE / "冻结协议.md", ROOT / "数据/数据集说明文档.md"]}, "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.values()}, "python": platform.python_version(), "threads": 1}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    result = {"subject": "DEP01", "pooled_oof": metric(trial_y, p), "pair_mean": {k: float(np.mean([metric(trial_y[va], v)[k] for v, (_, va) in zip(values, pair_splits(trial_y))])) for k in ("auc", "f1")}, "constant_0.5": metric(trial_y, np.full(8, 0.5)), "n_trials": 8, "n_windows": 40, "prediction_count": 4, "candidate_count": len(CANDIDATES), "elapsed_seconds": time.perf_counter() - start}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
