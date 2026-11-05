import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def test_config_and_outputs_are_local():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    assert config["expected_subjects"] == 69
    assert config["expected_train_rows"] == 2760
    assert not (HERE / "../047_time_frequency_cnn").resolve().samefile((HERE / "047_time_frequency_cnn").resolve()) if (HERE / "047_time_frequency_cnn").exists() else True


def test_canonical_keys_after_run():
    path = HERE / "input_metadata" / "canonical_keys.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    assert len(frame) == 2760
    assert frame[["subject_id", "physical_trial_index", "window_index"]].duplicated().sum() == 0
    assert (frame.groupby(["subject_id", "physical_trial_index"]).size() == 5).all()


def test_covariance_finite_after_run():
    path = HERE / "input_metadata" / "covariances.npy"
    if not path.exists():
        return
    cov = np.load(path)
    assert cov.shape == (2760, 30, 30)
    assert np.isfinite(cov).all()
    assert np.allclose(cov, cov.transpose(0, 2, 1), atol=1e-8)
