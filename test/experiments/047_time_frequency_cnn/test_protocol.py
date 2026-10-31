import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment047", HERE / "run_experiment.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_stft_shape_and_finite():
    rng = np.random.default_rng(47)
    tensor, frequencies = module.compute_stft_tensor(rng.normal(size=(2, 30, 2500)))
    assert tensor.shape == (2, 30, 23, 38)
    assert frequencies[0] >= 1 and frequencies[-1] <= 45
    assert np.isfinite(tensor).all()


def test_training_validation_subject_isolation():
    subjects = np.repeat(np.array([f"s{i}" for i in range(20)]), 2)
    groups = np.repeat(np.arange(20) % 2, 2)
    folds = module.make_subject_folds(subjects, groups, 2027)
    module.assert_subject_isolation(subjects, folds)
    assert len(np.unique(folds)) == 5


def test_validation_label_perturbation_does_not_change_input():
    x = np.arange(2 * 30 * 23 * 40, dtype=np.float32).reshape(2, 30, 23, 40)
    y = np.array([0, 1])
    mean_a, std_a = module.fit_channel_scaler(x)
    mean_b, std_b = module.fit_channel_scaler(x)
    np.testing.assert_array_equal(module.transform_tensor(x, mean_a, std_a), module.transform_tensor(x, mean_b, std_b))
    assert not np.shares_memory(y, x)


def test_window_independence():
    rng = np.random.default_rng(48)
    first, _ = module.compute_stft_tensor(rng.normal(size=(1, 30, 2500)))
    second, _ = module.compute_stft_tensor(rng.normal(size=(1, 30, 2500)))
    changed, _ = module.compute_stft_tensor(np.concatenate([rng.normal(size=(1, 30, 2500)), rng.normal(size=(1, 30, 2500))])[:1])
    assert first.shape == second.shape == changed.shape


def test_outputs_if_present():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    metrics_path = HERE / "metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(metrics["runs"]) >= 2
    for name in metrics["runs"]:
        oof = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv")
        assert len(oof) == config["expected_oof_rows"]
        assert oof.prob_positive.between(0, 1).all()
        public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(public) == config["public_rows"]
        assert list(public.columns) == ["user_id", "trial_id", "prob. of positive emotion"]
