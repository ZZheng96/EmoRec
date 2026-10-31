import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment032", HERE / "run_experiment.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cache_alignment_and_dimensions():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    spectral = module.load_cache((HERE / config["spectral_cache"]).resolve())
    temporal = module.load_cache((HERE / config["temporal_cache"]).resolve())
    module.assert_cache_alignment(spectral, temporal)
    assert spectral["X_train"].shape == (2800, 210)
    assert temporal["X_train"].shape == (2800, 450)


def test_subject_fold_isolation_and_determinism():
    subjects = np.repeat(np.array([f"s{i}" for i in range(20)]), 2)
    groups = np.repeat(np.arange(20) % 2, 2)
    folds1 = module.make_subject_folds(subjects, groups, 2027)
    folds2 = module.make_subject_folds(subjects, groups, 2027)
    np.testing.assert_array_equal(folds1, folds2)
    module.assert_subject_isolation(subjects, folds1)
    assert len(np.unique(folds1)) == 5


def test_hgb_parameters_and_probability_range():
    config = {"hgb": {"max_iter": 200, "learning_rate": .03, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1, "early_stopping": False}}
    estimator = module.make_model(config, 2026)[-1]
    assert estimator.get_params()["max_iter"] == 200
    assert estimator.get_params()["max_leaf_nodes"] == 15
    assert estimator.get_params()["early_stopping"] is False
    rng = np.random.default_rng(1)
    estimator.fit(rng.normal(size=(20, 3)), np.array([0, 1] * 10))
    probabilities = estimator.predict_proba(rng.normal(size=(4, 3)))[:, 1]
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_outputs_if_present():
    metrics_path = HERE / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("run experiment first")
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(result["runs"]) == {"spectral_control", "temporal_control", "fixed_05_05"}
    for name in result["runs"]:
        for repeat in range(1, 6):
            frame = pd.read_csv(HERE / "predictions" / f"{name}_repeat{repeat:02d}_oof.csv")
            assert len(frame) == 2760 and frame.subject_id.nunique() == 69 and "HC43" not in set(frame.subject_id)
            assert frame.prob_positive.between(0, 1).all()
        public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(public) == 80 and public["prob. of positive emotion"].between(0, 1).all()
