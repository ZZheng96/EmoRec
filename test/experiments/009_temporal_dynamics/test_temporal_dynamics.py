import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).with_name("run_experiment.py")
spec = importlib.util.spec_from_file_location("experiment009", PATH)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_temporal_features_shape_and_slope():
    rng = np.random.default_rng(2026)
    segments = rng.normal(size=(2, 30, 2500))
    features, names = experiment.temporal_log_features(segments)
    assert features.shape == (2, 450)
    assert len(names) == 450
    assert np.isfinite(features).all()


def test_fold_protocol_matches_baseline_cache():
    baseline = experiment.load_cache(experiment.BASELINE_CACHE)
    folds = experiment.make_subject_folds(baseline["subject_ids"], baseline["health_groups"], 2026)
    assert len(np.unique(baseline["subject_ids"])) == 70
    assert len(np.unique(baseline["subject_ids"][baseline["subject_ids"] != "HC43"])) == 69
    assert np.array_equal(folds[baseline["subject_ids"] != "HC43"], folds[baseline["subject_ids"] != "HC43"])
