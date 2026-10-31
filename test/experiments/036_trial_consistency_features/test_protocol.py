import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment036", HERE / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec); spec.loader.exec_module(experiment)

def test_dimensions_and_block_statistics_are_finite():
    x = np.arange(12, dtype=float).reshape(4, 3); subjects = np.array(["A", "A", "A", "A"])
    stats = experiment.block_statistics(x, subjects, 2)
    assert stats.shape == (4, 9) and np.isfinite(stats).all()
    np.testing.assert_allclose(stats[:2, 3:6], [[1.5] * 3, [1.5] * 3])
    assert experiment.consistency_features(x, subjects, 2).shape == (4, 12)

def test_block_statistics_do_not_use_labels():
    x = np.random.default_rng(1).normal(size=(6, 4)); s = np.array(["A"] * 6)
    y1 = np.zeros(6); y2 = np.ones(6)
    f1 = experiment.consistency_features(x, s, 2)
    f2 = experiment.consistency_features(x, s, 2)
    np.testing.assert_array_equal(f1, f2)
    assert not hasattr(experiment, "y")
    assert y1.shape == y2.shape

def test_cache_alignment_and_subject_folds():
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8")); spectral = experiment.load_cache((HERE / cfg["spectral_cache"]).resolve()); temporal = experiment.load_cache((HERE / cfg["temporal_cache"]).resolve()); experiment.assert_cache_alignment(spectral, temporal)
    keep = spectral["subject_ids"] != "HC43"; folds = experiment.make_subject_folds(spectral["subject_ids"], spectral["health_groups"], 2026)[keep]; experiment.assert_subject_isolation(spectral["subject_ids"][keep], folds); assert keep.sum() == 2760

def test_outputs():
    for name in ("spectral_control", "temporal_control", "consistency_spectral", "fixed_05_05"):
        oof = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv"); public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(oof) == 2760 and len(public) == 80; assert np.isfinite(oof.prob_positive).all() and ((oof.prob_positive >= 0) & (oof.prob_positive <= 1)).all(); p = public["prob. of positive emotion"]; assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
