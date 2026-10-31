import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment031", HERE / "run_experiment.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_protocol_config_and_cache_dimensions():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    assert config["hgb_candidates"] == [7, 15, 31]
    assert config["control_leaf"] == 15
    assert config["inner_folds"] == 3
    spectral = protocol.load_cache((HERE / config["spectral_cache"]).resolve())
    temporal = protocol.load_cache((HERE / config["temporal_cache"]).resolve())
    assert spectral["X_train"].shape[1] >= 150
    assert temporal["X_train"].shape[1] == 450
    assert spectral["X_public"].shape[0] == 80


def test_inner_subject_isolation_determinism_and_candidates():
    rng = np.random.default_rng(2026)
    subjects = np.repeat(np.array([f"s{i}" for i in range(12)]), 3)
    y = np.repeat(np.arange(12) % 2, 3)
    X = rng.normal(size=(36, 4))
    protocol.CONFIG = {"hgb": {"max_iter": 3, "learning_rate": .03, "min_samples_leaf": 2, "l2_regularization": 1, "early_stopping": False}, "hgb_candidates": [7, 15, 31], "control_leaf": 15, "inner_folds": 3}
    p1, a1 = protocol.inner_oof(X, y, subjects, 7, 2026)
    p2, a2 = protocol.inner_oof(X, y, subjects, 7, 2026)
    np.testing.assert_allclose(p1, p2)
    np.testing.assert_array_equal(a1, a2)
    for fold in range(3):
        assert set(subjects[a1 == fold]).isdisjoint(set(subjects[a1 != fold]))
    chosen, rows = protocol.choose_leaf(X, y, subjects, 2026)
    assert {row["leaf"] for row in rows} == {7, 15, 31}
    assert chosen["leaf"] in {7, 15, 31}


def test_platt_is_fitted_on_training_oof_and_probability_range():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([.1, .3, .6, .9, .2, .8])
    calibrator = protocol.fit_platt(p, y)
    calibrated = protocol.apply_platt(calibrator, p)
    assert calibrated.shape == p.shape
    assert np.all((calibrated > 0) & (calibrated < 1))


def test_complete_outputs_if_present():
    metrics_path = HERE / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("run experiment first")
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = {"spectral_control", "temporal_control", "spectral_selected", "temporal_selected", "spectral_lr", "temporal_lr", "raw_fusion", "platt_fusion"}
    assert set(result["runs"]) == expected
    for name in expected:
        oof = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv")
        assert len(oof) == 2760 and oof.subject_id.nunique() == 69 and "HC43" not in set(oof.subject_id)
        assert oof.prob_positive.between(0, 1).all()
        public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(public) == 80 and not public.duplicated(["user_id", "trial_id"]).any()
        assert public["prob. of positive emotion"].between(0, 1).all()
    selection = pd.read_csv(HERE / "predictions" / "leaf_selection.csv")
    assert len(selection) == 5
    assert selection.spectral_selected_leaf.isin([7, 15, 31]).all()
    assert selection.temporal_selected_leaf.isin([7, 15, 31]).all()
    inner = pd.read_csv(HERE / "predictions" / "inner_oof.csv")
    assert set(inner.outer_fold) == set(range(5))
    assert set(inner.leaf) == {7, 15, 31}
