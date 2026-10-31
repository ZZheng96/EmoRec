import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment039", HERE / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_150_to_165_and_math_identities():
    rng = np.random.default_rng(39)
    x = rng.normal(size=(4, 150))
    ids = np.array(["A", "A", "B", "B"])
    out = experiment.robust_power_features(x, ids)
    assert out.shape == (4, 165)
    z = experiment.normalize_by_subject(x, ids, "zscore").reshape(4, 30, 5)
    assert np.allclose(out[:, 150:165][:, :5], np.median(z, axis=1))
    assert np.allclose(out[:, 150:165][:, 5:10], np.median(np.abs(z - np.median(z, axis=1)[:, None, :]), axis=1))
    assert np.allclose(out[:, 150:165][:, 10:15], np.percentile(z, 75, axis=1) - np.percentile(z, 25, axis=1))


def test_no_label_interface_and_subject_isolation():
    x = np.ones((4, 150))
    ids = np.array(["A", "A", "B", "B"])
    assert experiment.robust_power_features(x, ids).shape[1] == 165
    folds = np.array([0, 0, 1, 1])
    experiment.assert_subject_isolation(ids, folds)
    with pytest.raises(ValueError):
        experiment.assert_subject_isolation(np.array(["A", "A", "A", "B"]), folds)


def test_cache_metadata_alignment_and_dimensions():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    data = experiment.load_cache(experiment.SPECTRAL)
    experiment.assert_cache_alignment(data)
    assert data["X_train"].shape == (2800, 210)
    assert data["X_public"].shape == (80, 210)
    assert config["base_dimensions"] == 150 and config["output_dimensions"] == 165
    assert np.sum(data["subject_ids"] != "HC43") == 2760


def test_row_order_invariance_after_id_restore():
    data = experiment.load_cache(experiment.SPECTRAL)
    keep = data["subject_ids"] != "HC43"
    x = data["X_train"][keep, :150]
    ids = data["subject_ids"][keep]
    rng = np.random.default_rng(39)
    order = rng.permutation(len(x))
    restored = experiment.robust_power_features(x[order], ids[order])[np.argsort(order)]
    expected = experiment.robust_power_features(x, ids)
    np.testing.assert_allclose(restored, expected)


def test_complete_oof_probability_range_public_rows():
    for name in ("spectral_control", "robust_power", "fixed_05_05"):
        frame = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv")
        assert len(frame) == 2760 and frame.prob_positive.between(0, 1).all()
        public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(public) == 80 and public.iloc[:, -1].between(0, 1).all()
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
    assert all(f"{name}_seed{seed}" in metrics["runs"] for name in ("spectral_control", "robust_power") for seed in [2027, 2028, 2029, 2030, 2031])
