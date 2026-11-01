import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp059', HERE / 'run_experiment.py')
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))


def test_zero_signal_and_finite_shape():
    x = np.zeros((2, 30, 2500))
    out, details = exp.burst_features(x, CONFIG, return_details=True)
    assert out.shape == (2, 240)
    np.testing.assert_array_equal(out, 0)
    assert all(not events for band in details for row in band for events in row)


def test_window_independence_and_row_permutation():
    x = np.random.default_rng(59).normal(size=(3, 30, 2500))
    a = exp.burst_features(x, CONFIG)
    np.testing.assert_allclose(a[[2, 0, 1]], exp.burst_features(x[[2, 0, 1]], CONFIG))
    np.testing.assert_allclose(a[:1], exp.burst_features(x[:1], CONFIG)[:1])
    assert np.isfinite(a).all()


def test_dimension_and_invalid_shape():
    assert len(exp.column_sets()['control_pe']) == 720
    assert len(exp.column_sets()['plus_burst']) == 960
    assert len(exp.feature_names(CONFIG)) == 240
    with pytest.raises(ValueError): exp.burst_features(np.zeros((1, 30, 2499)), CONFIG)


def test_feature_names_and_band_specific_minimum_duration():
    names = exp.feature_names(CONFIG)
    assert names[:4] == [
        'burst_alpha_FP1_event_rate_s', 'burst_alpha_FP1_duty_cycle',
        'burst_alpha_FP1_mean_duration_s', 'burst_alpha_FP1_mean_peak_over_median',
    ]
    assert names[120].startswith('burst_beta_')
    assert exp._minimum_samples([8.0, 13.0], 250) != exp._minimum_samples([13.0, 30.0], 250)


def test_runs_are_strict_and_minimum_duration():
    assert exp._runs(np.array([False, True, True, False]), 2) == [(1, 3)]
    assert exp._runs(np.array([False, True, False]), 2) == []
    assert exp._runs(np.array([True, True, False, True, True]), 2) == [(0, 2), (3, 5)]


def test_fold_isolation_and_oof_recomputed_metrics_if_present():
    if not (HERE / 'metrics.json').exists(): pytest.skip('full experiment has not run yet')
    metrics = json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))
    for method in exp.METHODS:
        for split in ['fixed002'] + [f'repeat{s}' for s in CONFIG['repeat_seeds']]:
            frame = pd.read_csv(HERE / 'predictions' / f'{method}_{split}_oof.csv')
            exp.base.assert_subject_isolation(frame.subject_id.to_numpy(), frame.fold.to_numpy())
            assert frame.prob_positive.between(0, 1).all()
            assert abs(roc_auc_score(frame.y_true, frame.prob_positive) - metrics['runs'][method][split]['overall']['auc']) < 1e-12
            threshold = CONFIG['threshold']; pred = (frame.prob_positive >= threshold).astype(int)
            f1 = 2 * ((pred & frame.y_true.astype(bool)).sum()) / max(pred.sum() + frame.y_true.sum(), 1)
            assert abs(f1 - metrics['runs'][method][split]['overall']['f1']) < 1e-12
            dep = frame[frame.health_group == 'DEP']
            assert abs(roc_auc_score(dep.y_true, dep.prob_positive) - metrics['runs'][method][split]['health_groups']['DEP']['pooled']['auc']) < 1e-12
        values = [exp.scalars(metrics['runs'][method][f'repeat{s}']) for s in CONFIG['repeat_seeds']]
        for key in exp.KEYS:
            assert abs(np.mean([v[key] for v in values]) - metrics['aggregate'][method][key]['mean']) < 1e-12
    assert max(metrics['control_reproduction_max_abs_diff_vs_057'].values()) < 1e-12
