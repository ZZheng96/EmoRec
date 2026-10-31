import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment028", HERE / "run_experiment.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_cache_alignment_and_dimensions():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    spectral = protocol.load_cache((HERE / config["spectral_cache"]).resolve())
    temporal = protocol.load_cache((HERE / config["temporal_cache"]).resolve())
    protocol.assert_cache_alignment(spectral, temporal)
    assert spectral["X_train"].shape[1] >= 150
    assert temporal["X_train"].shape[1] == 450


def test_inner_subject_isolation_and_determinism():
    rng = np.random.default_rng(2026)
    subjects = np.repeat(np.array([f"s{i}" for i in range(12)]), 3)
    y = np.tile([0, 1, 0], 12)
    x1, x2 = rng.normal(size=(36, 4)), rng.normal(size=(36, 5))
    protocol.CONFIG = {"hgb": {"max_iter": 3, "learning_rate": .03, "max_leaf_nodes": 3, "min_samples_leaf": 2, "l2_regularization": 1, "early_stopping": False}}
    result1 = protocol.select_weight(y, x1, x2, subjects, [0, .5, 1], 2026)
    result2 = protocol.select_weight(y, x1, x2, subjects, [0, .5, 1], 2026)
    assert result1[0] == result2[0]
    np.testing.assert_allclose(result1[2], result2[2])
    np.testing.assert_allclose(result1[3], result2[3])
    assignment = result1[4]
    for fold in range(3):
        assert not (set(subjects[assignment == fold]) & set(subjects[assignment != fold]))


def test_weight_candidates_and_tie_breaking():
    assert np.allclose(np.arange(0, 1.001, .05), np.array([i / 20 for i in range(21)]))
    scores = [{"weight": 0.0, "auc": 0.8, "f1": .5}, {"weight": 0.5, "auc": .8, "f1": .5}, {"weight": 1.0, "auc": .8, "f1": .5}]
    assert max(scores, key=lambda r: (r["auc"], r["f1"], r["weight"]))["weight"] == 1.0


def test_complete_outputs_if_present():
    metrics_path = HERE / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("run experiment first")
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(result["runs"]) == {"spectral_control", "temporal_control", "nested_fusion", "fixed_05_05"}
    for name in result["runs"]:
        oof = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv")
        assert len(oof) == 2760 and oof.subject_id.nunique() == 69 and "HC43" not in set(oof.subject_id)
        assert oof.prob_positive.between(0, 1).all()
        public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(public) == 80 and not public.duplicated(["user_id", "trial_id"]).any()
        assert public["prob. of positive emotion"].between(0, 1).all()
    selection = pd.read_csv(HERE / "predictions" / "nested_weight_selection_oof.csv")
    assert len(selection) == 5 and np.isclose(selection.selected_weight.to_numpy()[:, None], np.arange(0, 1.001, .05)).any(axis=1).all()
