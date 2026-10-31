import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location("experiment011", Path(__file__).with_name("run_experiment.py"))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
aggregate_oof = _module.aggregate_oof
validate_oof = _module.validate_oof


def sample_oof() -> pd.DataFrame:
    rows = []
    for trial_id, label, values in ((1, 0, [0.1, 0.2, 0.3, 0.4, 0.5]), (2, 1, [0.6, 0.7, 0.8, 0.9, 1.0])):
        for value in values:
            rows.append({
                "subject_id": "S1",
                "health_group": "HC",
                "original_trial_id": trial_id,
                "fold": 0,
                "y_true": label,
                "prob_positive": value,
            })
    return pd.DataFrame(rows)


def test_aggregate_uses_subject_and_original_trial_and_preserves_five_segments() -> None:
    frame = sample_oof()
    mean = aggregate_oof(frame, "mean")
    median = aggregate_oof(frame, "median")
    assert mean["n_segments"].tolist() == [5, 5]
    np.testing.assert_allclose(mean["prob_positive"], [0.3, 0.8])
    np.testing.assert_allclose(median["prob_positive"], [0.3, 0.8])
    assert mean["y_true"].tolist() == [0, 1]


def test_validation_rejects_incomplete_original_trial() -> None:
    frame = sample_oof().iloc[:-1]
    with pytest.raises(ValueError, match="exactly 5"):
        validate_oof(frame)


def test_validation_rejects_mixed_labels_within_trial() -> None:
    frame = sample_oof()
    frame.loc[1, "y_true"] = 1
    with pytest.raises(ValueError, match="inconsistent labels"):
        validate_oof(frame)
