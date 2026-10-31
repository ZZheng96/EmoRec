import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment040", HERE / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_dimensions_bands_and_label_free_interface():
    rng = np.random.default_rng(40); x = rng.normal(size=(2, 30, 2500))
    control = experiment.extract_spectral_control(x); modulation = experiment.extract_modulation_features(x)
    assert control.shape == (2, 150); assert modulation.shape == (2, 450)
    assert experiment.EEG_BANDS == ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
    assert experiment.MOD_BANDS == ((0.2, 0.5), (0.5, 1.0), (1.0, 2.0))
    assert np.isfinite(modulation).all()


def test_row_order_restore_subject_isolation():
    rng = np.random.default_rng(41); x = rng.normal(size=(6, 30, 2500)); ids = np.array(["A", "A", "B", "B", "C", "C"])
    order = rng.permutation(6); restored = experiment.extract_modulation_features(x[order])[np.argsort(order)]
    np.testing.assert_allclose(restored, experiment.extract_modulation_features(x))
    folds = np.array([0, 0, 1, 1, 2, 2]); experiment.assert_subject_isolation(ids, folds)


def test_cache_and_outputs():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8")); assert config["modulation_dimensions"] == 450
    data = experiment.load_cache(); assert data["X_modulation"].shape == (2800, 450); assert data["X_public_modulation"].shape == (80, 450)
    for name in ("spectral_control", "modulation_spectral", "fixed_05_05"):
        oof = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv"); public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(oof) == 2760 and oof.prob_positive.between(0, 1).all(); assert len(public) == 80 and public.iloc[:, -1].between(0, 1).all()
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
    assert all(f"{name}_seed{seed}" in metrics["runs"] for name in ("spectral_control", "modulation_spectral") for seed in [2032, 2033, 2034])
