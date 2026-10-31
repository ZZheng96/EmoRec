import importlib.util
from pathlib import Path

import numpy as np

from emo_rec.features import ASYMMETRY_PAIRS, CHANNEL_NAMES, FREQUENCY_BANDS

_spec = importlib.util.spec_from_file_location("experiment014", Path(__file__).with_name("run_experiment.py"))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
asymmetry = _module.asymmetry


def test_asymmetry_shape_and_left_minus_right():
    bands = len(FREQUENCY_BANDS)
    names = np.array([f"log_bandpower_{channel}_{band}" for channel in CHANNEL_NAMES for band in FREQUENCY_BANDS])
    X = np.zeros((2, len(names)))
    X[:, 0] = [3.0, 5.0]
    X[:, 1] = [1.0, 2.0]
    combined, combined_names = asymmetry(X, names)
    assert combined.shape == (2, 150 + len(ASYMMETRY_PAIRS) * bands)
    assert combined_names[-1] == "asymmetry_O1_O2_gamma"
    assert np.array_equal(combined[:, 150:155], X[:, :5] - X[:, bands:bands + 5])


def test_asymmetry_is_finite():
    names = np.array([f"log_bandpower_{channel}_{band}" for channel in CHANNEL_NAMES for band in FREQUENCY_BANDS])
    combined, _ = asymmetry(np.ones((3, len(names))), names)
    assert np.isfinite(combined).all()
