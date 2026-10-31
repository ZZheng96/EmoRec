import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("experiment015", Path(__file__).with_name("run_experiment.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_choose_weight_uses_auc_then_f1():
    y = np.array([0, 0, 1, 1])
    baseline = np.array([0.1, 0.2, 0.8, 0.9])
    auxiliary = np.array([[0.9, 0.8, 0.2, 0.1], [0.9, 0.8, 0.2, 0.1]])
    weight, _ = MODULE.choose_weight(y, baseline, auxiliary, np.ones(4, dtype=bool), [0.0, 1.0])
    assert weight == 1.0


def test_load_oof_rejects_key_mismatch(tmp_path):
    frame = pd.DataFrame({"subject_id": ["A"], "health_group": ["HC"], "original_trial_id": [1], "fold": [0], "y_true": [0], "prob_positive": [0.2]})
    path = tmp_path / "x.csv"
    frame.to_csv(path, index=False)
    reference = frame.copy()
    reference.loc[0, "subject_id"] = "B"
    with pytest.raises(RuntimeError, match="alignment"):
        MODULE.load_oof(path, reference)
