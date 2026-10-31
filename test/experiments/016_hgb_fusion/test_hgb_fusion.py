import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("experiment016", Path(__file__).with_name("run_experiment.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_choose_weight_uses_auc_then_f1():
    y = np.array([0, 0, 1, 1])
    baseline = np.array([0.1, 0.2, 0.8, 0.9])
    hgb = np.array([0.9, 0.8, 0.2, 0.1])
    weight, _ = MODULE.choose_weight(y, baseline, hgb, np.ones(4, dtype=bool), [0.0, 1.0])
    assert weight == 1.0


def test_load_oof_rejects_key_mismatch(tmp_path):
    frame = pd.DataFrame({"subject_id": ["A"], "health_group": ["HC"], "original_trial_id": [1], "fold": [0], "y_true": [0], "prob_positive": [0.2]})
    path = tmp_path / "x.csv"
    frame.to_csv(path, index=False)
    reference = frame.copy()
    reference.loc[0, "subject_id"] = "B"
    with pytest.raises(RuntimeError, match="alignment"):
        MODULE.load_oof(path, reference)


def test_hgb_parameters_are_fixed():
    config = {"hgb": {"max_iter": 200, "learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1, "early_stopping": False}}
    estimator = MODULE.model(config, 2026)[-1]
    assert estimator.get_params()["max_iter"] == 200
    assert estimator.get_params()["early_stopping"] is False


def test_baseline_model_matches_lr_protocol():
    config = {"baseline_lr": {"C": 1.0, "penalty": "l2", "max_iter": 2000, "solver": "lbfgs"}}
    estimator = MODULE.baseline_model(config, 2026)[-1]
    assert estimator.get_params()["C"] == 1.0
    assert estimator.get_params()["max_iter"] == 2000
    assert estimator.get_params()["solver"] == "lbfgs"
