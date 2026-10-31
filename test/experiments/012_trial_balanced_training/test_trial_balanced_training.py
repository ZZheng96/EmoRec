import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location("experiment012", Path(__file__).with_name("run_experiment.py"))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
aggregate_trials = _module.aggregate_trials


def test_aggregate_trials_requires_five_segments_and_preserves_fold() -> None:
    frame = pd.DataFrame({
        "subject_id": ["S1"] * 5,
        "health_group": ["HC"] * 5,
        "original_trial_id": [1] * 5,
        "fold": [2] * 5,
        "y_true": [0] * 5,
        "prob_positive": np.arange(5) / 10,
    })
    result = aggregate_trials(frame)
    assert result.loc[0, "n_segments"] == 5
    assert result.loc[0, "fold"] == 2
    assert result.loc[0, "prob_positive"] == pytest.approx(0.2)


def test_aggregate_trials_rejects_incomplete_trial() -> None:
    frame = pd.DataFrame({
        "subject_id": ["S1"] * 4,
        "health_group": ["HC"] * 4,
        "original_trial_id": [1] * 4,
        "fold": [0] * 4,
        "y_true": [0] * 4,
        "prob_positive": [0.1] * 4,
    })
    with pytest.raises(ValueError, match="exactly 5"):
        aggregate_trials(frame)
