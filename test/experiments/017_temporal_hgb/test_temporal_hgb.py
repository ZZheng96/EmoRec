import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("experiment017", Path(__file__).with_name("run_experiment.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def cache(meta, rows=2):
    return {key: np.asarray(values) for key, values in meta.items()}


def test_cache_alignment_rejects_misaligned_metadata():
    base = {"subject_ids": ["DEP01", "DEP01"], "health_groups": ["DEP", "DEP"],
            "original_trial_ids": [1, 2], "y_train": [1, 0], "public_user_ids": ["P_test1"], "public_trial_ids": [1]}
    other = cache(base)
    other["y_train"][1] = 1
    with pytest.raises(RuntimeError, match="cache alignment"):
        MODULE.assert_cache_alignment(cache(base), other)


def test_outer_reference_rejects_misaligned_subject():
    reference = pd.DataFrame({"subject_id": ["DEP01"], "health_group": ["DEP"], "original_trial_id": [1], "fold": [-1], "y_true": [1], "prob_positive": [0.5]})
    frame = reference.copy()
    frame["subject_id"] = "HC01"
    path = Path(__file__).parent / "_bad_outer_reference.csv"
    frame.to_csv(path, index=False)
    try:
        with pytest.raises(RuntimeError, match="outer metadata alignment"):
            MODULE.load_outer_reference(path, reference)
    finally:
        path.unlink()


def test_subject_isolation_rejects_leakage():
    with pytest.raises(RuntimeError, match="subject leakage"):
        MODULE.assert_subject_isolation(np.asarray(["A", "A", "B", "B"]), np.asarray([0, 1, 0, 1]))


def test_hgb_parameters_are_fixed():
    config = {"hgb": {"max_iter": 200, "learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1, "early_stopping": False}}
    estimator = MODULE.model(config, 2026)[-1]
    assert estimator.get_params()["max_iter"] == 200
    assert estimator.get_params()["early_stopping"] is False
    assert estimator.get_params()["max_leaf_nodes"] == 15
