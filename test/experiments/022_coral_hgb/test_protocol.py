import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("experiment022", HERE / "run_experiment.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


@pytest.fixture
def cache():
    return MODULE.load_cache((HERE / CONFIG["feature_cache"]).resolve())


def test_baseline_matches_016_row_by_row(cache):
    ours = pd.read_csv(HERE / "predictions" / "baseline_zscore_oof.csv")
    reference = pd.read_csv(HERE.parent / "016_hgb_fusion" / "predictions" / "hgb_oof.csv")
    pd.testing.assert_frame_equal(ours.drop(columns="prob_positive")[reference.columns[:-1]], reference.drop(columns="prob_positive"))
    np.testing.assert_allclose(ours.prob_positive, reference.prob_positive, rtol=0, atol=1e-12)


def test_subject_isolation(cache):
    oof = pd.read_csv(HERE / CONFIG["outer_reference"])
    for fold in range(5):
        assert not (set(oof.loc[oof.fold != fold, "subject_id"]) & set(oof.loc[oof.fold == fold, "subject_id"]))


def test_validation_perturbation_does_not_change_training_side(cache):
    keep = cache["subject_ids"] != "HC43"
    X = cache["X_train"][keep, :150]
    subjects = cache["subject_ids"][keep]
    folds = pd.read_csv(HERE / CONFIG["outer_reference"])["fold"].to_numpy()
    base = MODULE.coral_transform(X, subjects)
    changed = X.copy()
    changed[folds == 0] += 1000
    altered = MODULE.coral_transform(changed, subjects)
    np.testing.assert_allclose(base[folds != 0], altered[folds != 0], atol=1e-12)


def test_train_validation_coral_is_same_transform():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(12, 4))
    subjects = np.repeat(["A", "B"], 6)
    whole = MODULE.coral_transform(X, subjects)
    separate = np.vstack([MODULE.coral_transform(X[:6], subjects[:6]), MODULE.coral_transform(X[6:], subjects[6:])])
    np.testing.assert_allclose(whole, separate, atol=1e-12)


def test_transforms_do_not_accept_or_use_labels():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3))
    subjects = np.repeat(["A", "B"], 4)
    np.testing.assert_allclose(MODULE.coral_transform(X, subjects), MODULE.coral_transform(X, subjects))
    assert "y" not in MODULE.coral_transform.__code__.co_varnames


def test_near_zero_variance_and_finite_values():
    X = np.array([[1, 2, 3], [1, 4, 3], [1, 6, 3], [1, 8, 3]], dtype=float)
    subjects = np.array(["A"] * 4)
    for transform in (MODULE.subject_zscore, MODULE.coral_transform):
        result = transform(X, subjects)
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result[:, [0, 2]], 0, atol=1e-12)


def test_fixed_model_parameters():
    expected = CONFIG["hgb"]
    for fold in range(5):
        estimator = MODULE.model(CONFIG, 2026 + fold)
        assert isinstance(estimator[0], StandardScaler)
        params = estimator[-1].get_params()
        assert params["random_state"] == 2026 + fold
        for key, value in expected.items():
            assert params[key] == value


def test_output_completeness(cache):
    reference = pd.read_csv(HERE / CONFIG["outer_reference"])
    for name in CONFIG["conditions"]:
        oof = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv")
        assert len(oof) == 2760 and oof.subject_id.nunique() == 69 and "HC43" not in set(oof.subject_id)
        pd.testing.assert_frame_equal(oof.drop(columns="prob_positive")[reference.columns[:-1]], reference.drop(columns="prob_positive"))
        assert np.isfinite(oof.prob_positive).all() and oof.prob_positive.between(0, 1).all()
        public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(public) == 80 and not public.duplicated(["user_id", "trial_id"]).any()
        np.testing.assert_array_equal(public.user_id, cache["public_user_ids"])
        np.testing.assert_array_equal(public.trial_id, cache["public_trial_ids"])
        assert public["prob. of positive emotion"].between(0, 1).all()
        MODULE.validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
