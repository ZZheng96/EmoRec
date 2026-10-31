import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

SPEC = importlib.util.spec_from_file_location("experiment018", Path(__file__).with_name("run_experiment.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_subject_isolation_rejects_inner_leakage():
    with pytest.raises(RuntimeError, match="subject leakage"):
        MODULE.assert_subject_isolation(np.asarray(["A", "A", "B", "B"]), np.asarray([0, 1, 0, 1]), 2)


def test_inner_folds_keep_subjects_together_and_health_stratified():
    subjects = np.repeat(["A", "B", "C", "D", "E", "F"], 2)
    health = np.repeat(["HC", "HC", "HC", "DEP", "DEP", "DEP"], 2)
    folds = MODULE.make_inner_folds(subjects, health, 2026)
    for subject in np.unique(subjects):
        assert len(set(folds[subjects == subject])) == 1
    for fold in range(3):
        assert set(health[folds == fold]) == {"HC", "DEP"}


def test_svc_probability_is_false():
    estimator = MODULE.model(1.0, 0.006666666666666667)[-1]
    assert estimator.probability is False


def test_calibration_uses_inner_margins():
    margins = np.asarray([-2.0, -1.0, 1.0, 2.0])
    y = np.asarray([0, 0, 1, 1])
    fitted = MODULE.calibrate(margins, y)
    assert fitted.coef_.shape == (1, 1)
    assert fitted.coef_[0, 0] > 0
    np.testing.assert_allclose(fitted.predict_proba(margins.reshape(-1, 1))[:, 1], fitted.predict_proba(margins[:, None])[:, 1])


def test_selection_does_not_accept_or_read_outer_labels():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(18, 3))
    y = np.asarray([0, 1] * 9)
    subjects = np.repeat([f"S{i}" for i in range(9)], 2)
    health = np.repeat(["HC", "DEP", "HC", "DEP", "HC", "DEP", "HC", "DEP", "HC"], 2)
    selected = MODULE.inner_select(X, y, subjects, health, 2026)
    assert set(selected) == {"folds", "candidates", "selected", "selected_margins"}
    assert len(selected["candidates"]) == 9
