import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment030", HERE / "run_experiment.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def config():
    return json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def test_roi_coverage_nonoverlap_and_dimensions():
    rois = protocol.roi_indices(config())
    assert sorted(i for values in rois.values() for i in values) == list(range(30))
    x = np.arange(2 * 150, dtype=float).reshape(2, 150)
    features, names = protocol.build_roi_features(x, config())
    assert features.shape == (2, 60)
    assert len(names) == 60


def test_trial_grouping_and_no_label_dependency():
    subjects = np.repeat(["s1", "s2"], 10)
    trials = np.tile(np.repeat([1, 2], 5), 2)
    x = np.random.default_rng(2026).normal(size=(20, 150))
    a, names = protocol.build_robust_features(x, subjects, trials, config())
    b, _ = protocol.build_robust_features(x, subjects, trials, config())
    assert a.shape == (20, 300) and len(names) == 300
    np.testing.assert_array_equal(a, b)
    assert np.allclose(a[0], a[4]) and not np.allclose(a[0], a[5])


def test_subject_isolation_and_deterministic_robust_features():
    subjects = np.repeat(["s1", "s2"], 5)
    trials = np.ones(10, dtype=int)
    x = np.zeros((10, 150)); x[:5, 0] = np.arange(5); x[5:, 0] = np.arange(5) + 100
    out, _ = protocol.build_robust_features(x, subjects, trials, config())
    assert np.all(out[:5, 0] == out[0, 0]) and np.all(out[5:, 0] == out[5, 0])
    assert out[0, 0] == out[5, 0]


def test_complete_outputs_if_present():
    metrics_path = HERE / "metrics.json"
    if not metrics_path.exists():
        return
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert {"spectral_control", "roi_static", "robust_trial_deviation", "fusion_roi_temporal"}.issubset(result["runs"])
    for name in result["runs"]:
        oof = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv")
        assert len(oof) == 2760 and oof.subject_id.nunique() == 69 and "HC43" not in set(oof.subject_id)
        assert oof.prob_positive.between(0, 1).all()
        public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(public) == 80 and public["prob. of positive emotion"].between(0, 1).all()
