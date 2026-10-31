import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("window_protocol", Path(__file__).with_name("run_experiment.py"))
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_batches_cover_all_rows_without_trial_ids_or_labels():
    subjects = np.repeat(["a", "b"], 40)
    batches = protocol.segment_batches(subjects)
    for subject in np.unique(subjects):
        for batch in range(5):
            local = np.flatnonzero(batches[subjects == subject] == batch)
            np.testing.assert_array_equal(local, np.arange(batch, 40, 5))
    with pytest.raises(ValueError):
        protocol.segment_batches(subjects[:-1])


def test_batch_statistics_and_other_batch_isolation():
    subjects = np.repeat(["a", "b"], 40)
    batches = protocol.segment_batches(subjects)
    X = np.random.default_rng(2026).normal(size=(80, 3))
    expected = X[:40:5]
    normalized = protocol.normalize(X, subjects, batches, 8)
    np.testing.assert_allclose(normalized[:40:5], (expected - expected.mean(0)) / expected.std(0))
    changed = X.copy()
    changed[(subjects == "b") | (batches != 0)] += 1000
    actual = protocol.normalize(changed, subjects, batches, 8)
    np.testing.assert_array_equal(normalized[:40:5], actual[:40:5])
    constant = protocol.normalize(np.ones((80, 2)), subjects, batches, 8)
    np.testing.assert_array_equal(constant, 0)


def test_actual_cache_reference_alignment():
    root = Path(__file__).resolve().parent.parent
    with np.load(root / "001_spectral_logpower/cache/features.npz") as cache:
        keep = cache["subject_ids"] != "HC43"
        data = {k: cache[k][keep] for k in ("subject_ids", "health_groups", "original_trial_ids", "y_train")}
    ref = pd.read_csv(root / "002_subject_normalization/predictions/subject_zscore_exclude_hc43_oof.csv")
    folds = protocol.align_reference(data, ref)
    assert len(folds) == 2760
    with pytest.raises(ValueError):
        protocol.align_reference(data, ref.iloc[::-1])


def test_saved_outputs_complete_if_present():
    here = Path(__file__).resolve().parent
    if not (here / "metrics.json").exists():
        pytest.skip("Run experiment first to validate saved outputs")
    for name in ("control_40_40", "matched_8_8", "eval_only_40_8"):
        oof = pd.read_csv(here / "predictions" / f"{name}_oof.csv")
        assert len(oof) == 2760 and oof.cache_row.nunique() == 2760
        assert oof.subject_id.nunique() == 69 and "HC43" not in set(oof.subject_id)
        assert oof.prob_positive.between(0, 1).all()
        public = pd.read_csv(here / "predictions" / f"{name}_public.csv")
        assert len(public) == 80 and not public.duplicated(["user_id", "trial_id"]).any()
        assert public["prob. of positive emotion"].between(0, 1).all()
