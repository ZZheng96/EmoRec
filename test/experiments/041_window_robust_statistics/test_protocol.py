import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment041", HERE / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_300_dimensions_statistics_and_filter_definition():
    rng = np.random.default_rng(41); x = rng.normal(size=(2, 30, 2500))
    out = experiment.robust_window_features(x)
    assert out.shape == (2, 300) and np.isfinite(out).all()
    assert experiment.BANDS == ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
    band = __import__("scipy").signal.sosfiltfilt(__import__("scipy").signal.butter(4, experiment.BANDS[0], btype="bandpass", fs=250, output="sos"), x, axis=-1)
    np.testing.assert_allclose(out[:, :30], np.median(np.abs(band), axis=-1))
    np.testing.assert_allclose(out[:, 30:60], np.percentile(band, 75, axis=-1) - np.percentile(band, 25, axis=-1))


def test_label_free_row_order_and_subject_isolation():
    x = np.random.default_rng(42).normal(size=(4, 30, 2500)); order = np.array([2, 0, 3, 1])
    np.testing.assert_allclose(experiment.robust_window_features(x[order])[np.argsort(order)], experiment.robust_window_features(x))
    ids = np.array(["A", "A", "B", "B"]); experiment.assert_subject_isolation(ids, np.array([0, 0, 1, 1]))
    with pytest.raises(AssertionError): experiment.assert_subject_isolation(np.array(["A", "A", "A", "B"]), np.array([0, 0, 1, 1]))


def test_cache_outputs_and_public_format():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8")); assert config["robust_dimensions"] == 300
    data = experiment.load_cache(); assert data["X_robust"].shape == (2800, 300); assert data["X_public_robust"].shape == (80, 300)
    for name in ("spectral_control", "robust_window", "fixed_05_05"):
        oof = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv"); public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(oof) == 2760 and oof.prob_positive.between(0, 1).all(); assert len(public) == 80 and public.iloc[:, -1].between(0, 1).all()
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8")); assert all(f"{n}_seed{s}" in metrics["runs"] for n in ("spectral_control", "robust_window") for s in range(2027, 2032))
