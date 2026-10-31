import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment026", HERE / "run_experiment.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_independent_metrics_and_fixed_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([.1, .6, .4, .9])
    got = protocol.metrics(y, p)
    assert got["auc"] == pytest.approx(roc_auc_score(y, p))
    assert got["f1"] == pytest.approx(f1_score(y, p >= .5))
    assert got["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, p >= .5))
    assert got["threshold"] == .5


def test_batches_normalization_and_subject_isolation():
    subjects = np.repeat(["a", "b"], 40)
    batches = protocol.segment_batches(subjects)
    X = np.random.default_rng(2026).normal(size=(80, 4))
    got = protocol.normalize(X, subjects, batches, 8)
    expected = X[:40:5]
    np.testing.assert_allclose(got[:40:5], (expected - expected.mean(0)) / expected.std(0))
    changed = X.copy()
    changed[(subjects == "b") | (batches != 0)] += 1000
    np.testing.assert_array_equal(got[:40:5], protocol.normalize(changed, subjects, batches, 8)[:40:5])
    np.testing.assert_array_equal(protocol.normalize(np.ones((80, 2)), subjects, batches, 8), 0)


def test_reference_fixed_folds_and_complete_keys():
    root = HERE.parents[2]
    with np.load(root / "test/experiments/001_spectral_logpower/cache/features.npz") as cache:
        keep = cache["subject_ids"] != "HC43"
        data = {k: cache[k][keep] for k in ("subject_ids", "health_groups", "original_trial_ids", "y_train")}
    ref = pd.read_csv(root / "test/experiments/002_subject_normalization/predictions/subject_zscore_exclude_hc43_oof.csv")
    folds = protocol.align_reference(data, ref)
    assert set(folds) == set(range(5)) and len(folds) == 2760
    assert list(ref.columns) == ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"]
    with pytest.raises(ValueError):
        protocol.align_reference(data, ref.iloc[::-1])


def test_control_reproduction_and_saved_outputs_if_present():
    metrics_path = HERE / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("Run experiment first")
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert result["runs"]["control_40_40"]["reference_016_max_abs_probability_difference"] < 1e-8
    for name in ("control_40_40", "matched_8_8", "eval_only_40_8"):
        oof = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv")
        assert len(oof) == 2760 and oof.subject_id.nunique() == 69 and "HC43" not in set(oof.subject_id)
        assert oof.prob_positive.between(0, 1).all()
        public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(public) == 80 and not public.duplicated(["user_id", "trial_id"]).any()
        assert public["prob. of positive emotion"].between(0, 1).all()
