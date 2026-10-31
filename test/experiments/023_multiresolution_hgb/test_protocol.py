import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("experiment023", Path(__file__).with_name("run_experiment.py"))
MODULE = importlib.util.module_from_spec(spec); spec.loader.exec_module(MODULE)


def test_multiresolution_shape_and_order():
    segments = np.zeros((2, 30, 2500))
    features, names = MODULE.raw_multiresolution_features(segments)
    assert features.shape == (2, 750)
    assert len(names) == 750


def test_cache_alignment_and_label_leakage_rejection():
    base = {"subject_ids": np.array(["A"]), "health_groups": np.array(["HC"]), "original_trial_ids": np.array([1]), "y_train": np.array([0]), "public_user_ids": np.array(["P_test1"]), "public_trial_ids": np.array([1])}
    other = {k: v.copy() for k, v in base.items()}; other["y_train"][0] = 1
    with pytest.raises(RuntimeError, match="cache alignment"):
        MODULE.assert_cache_alignment(base, other)


def test_subject_isolation_and_transform_locality():
    with pytest.raises(RuntimeError, match="subject leakage"):
        MODULE.assert_subject_isolation(np.array(["A", "A", "B", "B"]), np.array([0, 1, 0, 1]))
    from emo_rec.normalization import normalize_by_subject
    x = np.array([[1., 2.], [3., 4.], [100., 200.]])
    out = normalize_by_subject(x, np.array(["A", "A", "B"]), "zscore")
    assert np.allclose(out[:2].mean(0), 0)
    assert np.allclose(out[2], 0)


def test_fixed_model_parameters_and_output_completeness(tmp_path):
    config = {"hgb": {"max_iter": 200, "learning_rate": .03, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1., "early_stopping": False}}
    estimator = MODULE.model(config, 2026)[-1]
    assert estimator.get_params()["max_iter"] == 200
    assert estimator.get_params()["early_stopping"] is False
    assert estimator.get_params()["max_leaf_nodes"] == 15
    frame = pd.DataFrame({"subject_id": ["A"] * 2, "health_group": ["HC"] * 2, "original_trial_id": [1, 1], "fold": [0, 0], "y_true": [0, 1], "prob_positive": [.2, .8]})
    assert len(frame) == 2 and frame.prob_positive.notna().all()
