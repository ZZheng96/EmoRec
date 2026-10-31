import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment013", HERE / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_selection_handles_negative_direction_and_stable_ties():
    y = np.array([0, 0, 1, 1])
    X = np.column_stack([y, -y, np.ones(4)])
    selected, scores = experiment.select_features(X, y, 2)
    np.testing.assert_array_equal(selected, [0, 1])
    np.testing.assert_allclose(scores, [0.5, 0.5, 0.0])
    np.testing.assert_array_equal(experiment.select_features(X, y, 1)[0], [0])
    with pytest.raises(ValueError):
        experiment.select_features(X, y, 4)


def test_selection_receives_only_normalized_training_data(monkeypatch):
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(24, 4))
    y = np.tile([0, 1], 12)
    subjects = np.repeat(["A", "B"], 12)
    target = rng.normal(size=(8, 4))
    config = json.loads((experiment.BASE_DIR / "config.json").read_text(encoding="utf-8"))
    original = experiment.select_features
    seen = []

    def spy(train, labels, k):
        np.testing.assert_array_equal(labels, y)
        assert len(train) == 24
        for subject in np.unique(subjects):
            np.testing.assert_allclose(train[subjects == subject].mean(axis=0), 0, atol=1e-14)
        seen.append(train.copy())
        return original(train, labels, k)

    monkeypatch.setattr(experiment, "select_features", spy)
    _, first, scores = experiment.fit_predict(X, y, subjects, target, np.repeat("C", 8), 2, config)
    _, second, other_scores = experiment.fit_predict(X, y, subjects, target * 50, np.repeat("C", 8), 2, config)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(scores, other_scores)
    assert len(seen) == 2


def test_outputs_match_protocol_and_full_feature_baseline():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
    baseline = pd.read_csv(experiment.BASE_DIR / "predictions" / "subject_zscore_exclude_hc43_oof.csv")
    for k in experiment.K_VALUES:
        run = metrics["runs"][f"auc_top{k}"]
        oof = pd.read_csv(HERE / "predictions" / f"auc_top{k}_oof.csv")
        public = pd.read_csv(HERE / "predictions" / f"auc_top{k}_public.csv")
        assert oof.subject_id.nunique() == 69
        assert "HC43" not in set(oof.subject_id)
        assert len(public) == 80
        np.testing.assert_array_equal(oof.fold, baseline.fold)
        np.testing.assert_array_equal(oof.y_true, baseline.y_true)
        for fold in run["folds"]:
            assert not set(fold["train_subjects"]) & set(fold["validation_subjects"])
            assert len(set(fold["selected_indices"])) == k
        if k == 150:
            # Partition-wise normalization changes floating-point solver paths slightly.
            np.testing.assert_allclose(oof.prob_positive, baseline.prob_positive, atol=1e-6, rtol=0)
            assert run["overall"]["roc_auc"] == metrics["baseline_002"]["roc_auc"]
            assert run["overall"]["binary_f1"] == metrics["baseline_002"]["binary_f1"]
