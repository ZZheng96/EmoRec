import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

PATH = Path(__file__).with_name("run_experiment.py")
spec = importlib.util.spec_from_file_location("experiment008", PATH)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


@pytest.mark.parametrize("condition", ["baseline_zscore", "subject_residual", "coral_identity"])
def test_validation_cannot_change_training_parameters(condition):
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(32, 6))
    ids = np.repeat(["a", "b"], 16)
    valid = rng.normal(size=(16, 6))
    valid_ids = np.repeat("c", 16)
    train1, _ = experiment.prepare_features(X, ids, valid, valid_ids, condition)
    train2, _ = experiment.prepare_features(X, ids, valid ** 3 + 10, valid_ids, condition)
    np.testing.assert_array_equal(train1, train2)
    model1 = experiment.protocol.make_model({"name": "logistic_regression", "C": 1, "penalty": "l2", "solver": "lbfgs", "max_iter": 2000}, 2026)
    model2 = experiment.protocol.make_model({"name": "logistic_regression", "C": 1, "penalty": "l2", "solver": "lbfgs", "max_iter": 2000}, 2026)
    y = np.tile([0, 1], 16)
    model1.fit(train1, y)
    model2.fit(train2, y)
    np.testing.assert_array_equal(model1[-1].coef_, model2[-1].coef_)
    np.testing.assert_array_equal(model1[0].mean_, StandardScaler().fit(train1).mean_)


def test_residual_removes_subject_mean_and_coral_only_transforms_training():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 5))
    ids = np.repeat(["a", "b"], 12)
    valid, valid_ids = rng.normal(size=(8, 5)), np.repeat("c", 8)
    train, evaluation = experiment.prepare_features(X, ids, valid, valid_ids, "subject_residual")
    for subject in np.unique(ids):
        np.testing.assert_allclose(train[ids == subject].mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(evaluation, valid - valid.mean(axis=0))
    _, evaluation = experiment.prepare_features(X, ids, valid, valid_ids, "coral_identity")
    np.testing.assert_allclose(evaluation, (valid - valid.mean(axis=0)) / valid.std(axis=0))


def test_coral_matrix_formula_and_rank_deficiency():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(8, 12))
    X[:, 0] = 0
    X -= X.mean(axis=0)
    actual = experiment.coral_training_alignment(X, np.repeat("a", 8), 0.5)
    covariance = X.T @ X / 7
    values, vectors = np.linalg.eigh(0.5 * covariance + 0.5 * np.eye(12))
    expected = X @ ((vectors * values ** -0.5) @ vectors.T)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert np.isfinite(actual).all()


def test_overlapping_subjects_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        experiment.prepare_features(np.ones((2, 3)), ["a", "a"], np.ones((2, 3)), ["a", "a"], "baseline_zscore")


def test_protocol_folds_match_experiment002_saved_oof():
    import pandas as pd

    cache = experiment.protocol.load_feature_cache(experiment.EXPERIMENT_DIR.parent / "001_spectral_logpower/cache/features.npz")
    folds = experiment.protocol.make_subject_folds(cache["subject_ids"], cache["health_groups"], 2026)
    keep = cache["subject_ids"] != "HC43"
    old = pd.read_csv(experiment.BASE_DIR / "predictions/subject_zscore_exclude_hc43_oof.csv")
    np.testing.assert_array_equal(folds[keep], old["fold"])
    np.testing.assert_array_equal(cache["subject_ids"][keep], old["subject_id"])
    assert len(np.unique(cache["subject_ids"][keep])) == 69
