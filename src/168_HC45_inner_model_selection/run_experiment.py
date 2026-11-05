import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve()
ROOT = next(p for p in SOURCE.parents if (p / "数据").is_dir())
HERE = ROOT / "test" / "experiments" / "168_HC45_inner_model_selection"
BASE_DIR = ROOT / "test" / "experiments" / "091_within_subject_trial_oof"
_spec = importlib.util.spec_from_file_location("exp091", BASE_DIR / "run_experiment.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

METHODS = ("linear_svc", "logistic_regression", "hgb")
DIRECTIONS = ("raw", "reverse")
CANDIDATES = tuple(f"{m}:{d}" for m in METHODS for d in DIRECTIONS)
FIXED_THRESHOLD = 0.5
RANDOM_STATE = 2026


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -60, 60)))


def make_model(method):
    if method == "linear_svc":
        estimator = LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=RANDOM_STATE)
    elif method == "logistic_regression":
        estimator = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)
    else:
        estimator = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, random_state=RANDOM_STATE)
    return make_pipeline(StandardScaler(), estimator)


def score(model, x, method, direction):
    p = sigmoid(model.decision_function(x)) if method == "linear_svc" else model.predict_proba(x)[:, 1]
    return p if direction == "raw" else 1.0 - p


def classification(y, p):
    return {"auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None, "f1": float(f1_score(y, p >= FIXED_THRESHOLD, zero_division=0))}


def load_data(config):
    x, ref = base.load_data(config)
    keep = ref.subject.to_numpy() == "HC45"
    return x[keep], ref.loc[keep].reset_index(drop=True)


def trial_keys(ref):
    return np.asarray([f"{y}::{t}" for y, t in zip(ref.y, ref.trial)], dtype=object)


def evaluate_candidate(x, y, keys, train_keys, candidate):
    method, direction = candidate.split(":")
    unique = list(dict.fromkeys(train_keys))
    inner_p, inner_y = [], []
    for held in unique:
        tr = np.isin(keys, unique) & (keys != held)
        va = keys == held
        model = make_model(method)
        model.fit(x[tr], y[tr])
        inner_p.extend(score(model, x[va], method, direction))
        inner_y.extend(y[va])
    return classification(np.asarray(inner_y), np.asarray(inner_p))


def choose_candidate(x, y, keys, train_keys):
    scores = {candidate: evaluate_candidate(x, y, keys, train_keys, candidate) for candidate in CANDIDATES}
    best = max(CANDIDATES, key=lambda c: ((-np.inf if scores[c]["auc"] is None else scores[c]["auc"]), scores[c]["f1"], -CANDIDATES.index(c)))
    return best, scores


def subject_run(x, ref):
    y, keys = ref.y.to_numpy(dtype=int), trial_keys(ref)
    unique = list(dict.fromkeys(keys))
    assert len(unique) == 8 and all(np.sum(keys == k) == 5 for k in unique)
    p = np.full(len(y), np.nan)
    folds = np.full(len(y), -1, dtype=int)
    fold_meta = []
    for fold, held in enumerate(unique):
        train, valid = keys != held, keys == held
        selected, inner = choose_candidate(x, y, keys, keys[train])
        method, direction = selected.split(":")
        model = make_model(method)
        model.fit(x[train], y[train])
        p[valid] = score(model, x[valid], method, direction)
        folds[valid] = fold
        fold_meta.append({"subject": "HC45", "fold": fold, "held_out_trial_key": str(held), "train_rows": int(train.sum()), "valid_rows": int(valid.sum()), "selected_candidate": selected, "inner_scores": inner})
    assert np.isfinite(p).all() and np.array_equal(np.sort(folds), np.repeat(np.arange(8), 5))
    return p, folds, fold_meta


def main():
    started = time.perf_counter()
    config = json.loads((BASE_DIR / "config.json").read_text(encoding="utf8"))
    (HERE / "predictions").mkdir(parents=True, exist_ok=True)
    (HERE / "input_metadata").mkdir(parents=True, exist_ok=True)
    x, ref = load_data(config)
    y, keys = ref.y.to_numpy(dtype=int), trial_keys(ref)
    with threadpool_limits(limits=1):
        p, folds, fold_meta = subject_run(x, ref)
    unique = list(dict.fromkeys(keys))
    trial_p = np.asarray([p[keys == k].mean() for k in unique])
    trial_y = np.asarray([y[keys == k][0] for k in unique])
    pred = ref.copy()
    pred["fold"], pred["prob_positive"] = folds, p
    pred[["row_index", "subject", "group", "trial", "position", "y", "fold", "prob_positive"]].to_csv(HERE / "predictions/window_oof.csv", index=False)
    trial_out = pd.DataFrame({"trial_index": range(8), "trial_key": unique, "y": trial_y, "prob_positive": trial_p, "prediction_fixed05": (trial_p >= FIXED_THRESHOLD).astype(int), "window_count": 5})
    trial_out.to_csv(HERE / "predictions/trial_oof.csv", index=False)
    metrics = {"subject": "HC45", "method": "inner-only model:direction selection", "window": classification(y, p), "trial_mean": classification(trial_y, trial_p), "threshold": FIXED_THRESHOLD, "outer_validation_labels_used_for_selection": False, "candidate_order": list(CANDIDATES), "trial_aggregation": "mean of five window probabilities", "elapsed_seconds": time.perf_counter() - started}
    pd.DataFrame(fold_meta).to_csv(HERE / "input_metadata/folds.csv", index=False)
    inputs = [(BASE_DIR / config[k]).resolve() for k in ("spectral_cache", "temporal_cache", "pe_cache")]
    sources = [SOURCE, HERE / "test_168.py", HERE / "config.json", HERE / "test_protocol.txt", BASE_DIR / "run_experiment.py", ROOT / "数据" / "数据集说明文档.md"]
    sources = [p for p in sources if p.exists()]
    provenance = {"inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, "raw_shape": list(x.shape), "public_arrays_loaded": False, "subjects": ["HC45"], "rows": int(len(ref)), "normalization": config["normalization"], "threads": 1, "python": platform.python_version(), "sklearn": __import__("sklearn").__version__}
    metrics["provenance"] = provenance
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    (HERE / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
