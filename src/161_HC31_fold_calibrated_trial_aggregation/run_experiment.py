import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[2]
HERE = ROOT / "test" / "experiments" / "161_HC31_fold_calibrated_trial_aggregation"
FIXED_THRESHOLD = 0.5


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def load_data():
    base = ROOT / "test" / "experiments" / "091_within_subject_trial_oof"
    cfg = json.loads((base / "config.json").read_text(encoding="utf8"))
    arrays = []
    for key in ("spectral_cache", "temporal_cache", "pe_cache"):
        with np.load((base / cfg[key]).resolve(), allow_pickle=False) as z:
            arrays.append({k: z[k] for k in ("X_train", "y_train", "subject_ids", "original_trial_ids", "feature_names")})
    s, t, pe = arrays
    keep = s["subject_ids"] == "HC31"
    mask = np.asarray([str(n).startswith("log_bandpower_") for n in s["feature_names"]])
    x = np.column_stack((s["X_train"][keep][:, mask], t["X_train"][keep], pe["X_train"][keep]))
    y = s["y_train"][keep].astype(int)
    trial = np.arange(8).repeat(5)
    assert x.shape == (40, 720) and np.array_equal(y, np.repeat([0, 1], 20))
    return x, y, trial


def fit_score(x, y, train, valid):
    model = make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=2026))
    model.fit(x[train], y[train])
    return model.decision_function(x[valid]), model


def calibrate_outer(x, y, trial, train, valid):
    train_trials = np.unique(trial[train])
    train_positions = np.flatnonzero(train)
    inner_oof = np.empty(len(train_positions), dtype=float)
    for held in train_trials:
        inner_train = train & (trial != held)
        inner_valid = train & (trial == held)
        raw, _ = fit_score(x, y, inner_train, inner_valid)
        local_valid = np.flatnonzero(inner_valid)
        inner_oof[np.searchsorted(train_positions, local_valid)] = raw
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=2026)
    calibrator.fit(inner_oof.reshape(-1, 1), y[train])
    raw_valid, model = fit_score(x, y, train, valid)
    return calibrator.predict_proba(raw_valid.reshape(-1, 1))[:, 1], inner_oof, model


def main():
    started = time.perf_counter()
    (HERE / "predictions").mkdir(exist_ok=True)
    (HERE / "input_metadata").mkdir(exist_ok=True)
    x, y, trial = load_data()
    rows = []
    p = np.full(40, np.nan)
    with threadpool_limits(limits=1):
        for fold, held in enumerate(range(8)):
            train, valid = trial != held, trial == held
            calibrated, _, _ = calibrate_outer(x, y, trial, train, valid)
            p[valid] = calibrated
            for row, prob in zip(np.flatnonzero(valid), calibrated):
                rows.append({"row_index": int(row), "trial_index": int(held), "y": int(y[row]), "fold": fold, "prob_positive": float(prob)})
    window = pd.DataFrame(rows)
    window.to_csv(HERE / "predictions/window_oof.csv", index=False)
    trial_p = np.asarray([p[trial == t].mean() for t in range(8)])
    trial_y = np.asarray([y[trial == t][0] for t in range(8)])
    out = pd.DataFrame({"trial_index": range(8), "y": trial_y, "prob_positive": trial_p, "prediction_fixed05": (trial_p >= FIXED_THRESHOLD).astype(int), "window_count": 5})
    out.to_csv(HERE / "predictions/trial_oof.csv", index=False)
    provenance = {"source": "091 cached 720 features aligned to HC31", "outer_validation_labels_used_for_selection": False, "calibration": "each outer training fold only, 7 leave-one-trial-out inner folds", "trial_aggregation": "mean of five window probabilities", "python": platform.python_version(), "threads": 1}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    metrics = {"subject": "HC31", "method": "fold-inner Platt calibration + mean trial aggregation", "auc": float(roc_auc_score(trial_y, trial_p)), "f1": float(f1_score(trial_y, trial_p >= FIXED_THRESHOLD)), "window_auc": float(roc_auc_score(y, p)), "window_f1": float(f1_score(y, p >= FIXED_THRESHOLD)), "threshold": FIXED_THRESHOLD, "outer_validation_labels_used_for_selection": False, "elapsed_seconds": time.perf_counter() - started, "provenance": provenance}
    (HERE / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
