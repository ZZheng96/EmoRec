import importlib.util
from pathlib import Path

import numpy as np
import pytest

from emo_rec.normalization import normalize_by_subject

PATH = Path(__file__).parent / "experiments" / "006_frequency_graph" / "run_experiment.py"
spec = importlib.util.spec_from_file_location("frequency_graph", PATH)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_graph_shape_constant_and_finite():
    features, names = experiment.graph_features(np.zeros((2, 30, 2500)))
    assert features.shape == (2, 150)
    assert len(set(names)) == 150
    np.testing.assert_array_equal(features, 0)


def test_graph_connectivity_and_amplitude_invariance():
    rng = np.random.default_rng(2026)
    common = rng.normal(size=(1, 1, 2500))
    segments = np.repeat(common, 30, axis=1)
    features, _ = experiment.graph_features(segments)
    np.testing.assert_allclose(features, 1, atol=1e-12)
    scales = np.linspace(-3, 4, 30)[None, :, None]
    scaled, _ = experiment.graph_features(segments * scales)
    np.testing.assert_allclose(scaled, features, atol=1e-12)
    independent, _ = experiment.graph_features(rng.normal(size=(1, 30, 2500)))
    assert independent.mean() < 0.2


def test_graph_rejects_invalid_input():
    with pytest.raises(ValueError):
        experiment.graph_features(np.zeros((1, 30, 100)))
    with pytest.raises(ValueError):
        experiment.graph_features(np.full((1, 30, 2500), np.nan))


def test_normalization_is_subject_local():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 300))
    subjects = np.repeat(["train", "valid"], 8)
    original = normalize_by_subject(X, subjects, "zscore")
    X[:8] += 100
    changed = normalize_by_subject(X, subjects, "zscore")
    np.testing.assert_allclose(original[8:], changed[8:])
    np.testing.assert_allclose(original[8:].mean(axis=0), 0, atol=1e-14)


def test_folds_match_baseline_after_exclusion():
    baseline = experiment.load_baseline()
    subjects = np.repeat([f"HC{i}" for i in range(1, 46)] + [f"DEP{i}" for i in range(1, 26)], 2)
    groups = np.where(np.char.startswith(subjects, "HC"), "HC", "DEP")
    folds = baseline.make_subject_folds(subjects, groups, 2026)
    keep = subjects != "HC43"
    assert len(np.unique(subjects[keep])) == 69
    for subject in np.unique(subjects[keep]):
        assert len(np.unique(folds[subjects == subject])) == 1
    for fold in range(5):
        assert not set(subjects[keep & (folds == fold)]) & set(subjects[keep & (folds != fold)])
