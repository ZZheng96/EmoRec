import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("e163", ROOT / "src/163_HC16_fold_calibrated_trial_aggregation/run_experiment.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def test_data_alignment_and_shape():
    x, y, trial = e.load_data()
    assert x.shape == (40, 720)
    assert y.tolist() == [0] * 20 + [1] * 20
    assert np.bincount(trial).tolist() == [5] * 8


def test_inner_calibration_isolation():
    x, y, trial = e.load_data()
    train, valid = trial != 0, trial == 0
    p, inner, _ = e.calibrate_outer(x, y, trial, train, valid)
    assert len(inner) == 35 and np.isfinite(inner).all()
    assert len(p) == 5 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_real_outputs_and_fixed_metrics():
    window = pd.read_csv(HERE / "predictions/window_oof.csv")
    trial = pd.read_csv(HERE / "predictions/trial_oof.csv")
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    assert len(window) == 40 and len(trial) == 8
    assert window.prob_positive.between(0, 1).all()
    assert np.isclose(metrics["auc"], roc_auc_score(trial.y, trial.prob_positive))
    assert np.isclose(metrics["f1"], f1_score(trial.y, trial.prediction_fixed05))
    assert metrics["threshold"] == 0.5
    assert metrics["outer_validation_labels_used_for_selection"] is False
