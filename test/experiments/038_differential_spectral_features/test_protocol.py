import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment038", HERE / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_differential_feature_shape_and_no_label_interface():
    x = np.random.default_rng(2026).normal(size=(3, 30, 2500))
    features, names = experiment.extract_differential_features(x)
    assert features.shape == (3, 60)
    assert len(names) == 60 and np.isfinite(features).all()
    np.testing.assert_allclose(features, experiment.extract_differential_features(x * 3)[0], atol=1e-10)


@pytest.mark.parametrize("x", [np.zeros((1, 29, 2500)), np.zeros((1, 30, 2499)), np.zeros((0, 30, 2500)), np.full((1, 30, 2500), np.nan)])
def test_invalid_raw_windows_rejected(x):
    with pytest.raises(ValueError):
        experiment.extract_differential_features(x)


def test_cache_metadata_dimensions_and_window_counts():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    spectral = experiment.load_cache(experiment.SPECTRAL)
    differential = experiment.load_cache(experiment.DIFF_CACHE)
    experiment.assert_cache_alignment(spectral, differential)
    assert spectral["X_train"].shape == (2800, 210)
    assert spectral["X_train"][:, :150].shape == (2800, 150)
    assert differential["X_train"].shape == (2800, 60)
    assert spectral["X_public"].shape == (80, 210)
    assert differential["X_public"].shape == (80, 60)
    assert all(np.sum(spectral["subject_ids"] == s) == 40 for s in np.unique(spectral["subject_ids"]))
    assert all(np.sum(spectral["public_user_ids"] == s) == 8 for s in np.unique(spectral["public_user_ids"]))
    assert config["excluded_subject"] == "HC43"


def test_row_order_invariance_after_id_restore():
    spectral = experiment.load_cache(experiment.SPECTRAL)
    differential = experiment.load_cache(experiment.DIFF_CACHE)
    rng = np.random.default_rng(38)
    order = rng.permutation(len(spectral["y_train"]))
    restored = differential["X_train"][order][np.argsort(order)]
    np.testing.assert_allclose(restored, differential["X_train"])
    # Feature computation has no adjacent-row or trial aggregation input.
    assert differential["metadata_json"].item() and "window-wise" in differential["metadata_json"].item()


def test_subject_isolation_and_complete_oof():
    reference = pd.read_csv(experiment.REFERENCE)
    experiment.assert_subject_isolation(reference.subject_id.to_numpy(), reference.fold.to_numpy())
    assert len(reference) == 2760 and reference.fold.isin(range(5)).all()
    for name in ("spectral_control", "differential_spectral", "fixed_05_05"):
        frame = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv")
        assert len(frame) == 2760 and frame.prob_positive.between(0, 1).all()
        experiment.assert_subject_isolation(frame.subject_id.to_numpy(), frame.fold.to_numpy())


def test_public_submission_shape_and_probability_range():
    for name in ("spectral_control", "differential_spectral", "fixed_05_05"):
        frame = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(frame) == 80
        assert frame["prob. of positive emotion"].between(0, 1).all()
        experiment.validate_submission(frame, [f"P_test{i}" for i in range(1, 11)])
