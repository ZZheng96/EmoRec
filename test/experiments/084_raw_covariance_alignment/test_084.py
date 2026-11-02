import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp084', HERE / 'run_experiment.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_config_and_frozen_protocol():
    c = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    assert c['shrinkage'] == 0.1 and c['expected_model_fits'] == 30
    assert c['feature_blocks'] == [150, 450, 120]
    assert c['repeat_seeds'] == [2042, 2043, 2044, 2045, 2046]


def test_covariance_positive_definite_and_whitening():
    rng = np.random.default_rng(84)
    x = rng.normal(size=(8, 30, 100))
    cov = m.regularized_covariances(x, 0.1).mean(axis=0)
    w = m.whiten_matrix(cov)
    np.testing.assert_allclose(w @ cov @ w, np.eye(30), atol=1e-7)


def test_window_order_permutation_invariance():
    rng = np.random.default_rng(840)
    x = rng.normal(size=(8, 30, 100))
    a = m.regularized_covariances(x, 0.1).mean(axis=0)
    b = m.regularized_covariances(x[[4, 0, 7, 2, 1, 6, 3, 5]], 0.1).mean(axis=0)
    np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(m.whiten_matrix(a), m.whiten_matrix(b), atol=1e-10)


def test_training_data_loader_does_not_read_public_arrays(monkeypatch):
    c = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    real = m.load_train_segments
    def guarded(path):
        assert '公开测试集' not in str(path)
        return real(path)
    monkeypatch.setattr(m, 'load_train_segments', guarded)
    raw, ref, folds, info = m.load_data(c)
    assert raw.shape == (2760, 720)
    assert info['public_arrays_loaded'] is False
    assert set(ref.position) == set(range(5))
    assert (ref.groupby('position').size() == 552).all()


def test_control_sources_are_070():
    c = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    folder = HERE.parent / '070_train_context_match_new_splits' / 'predictions'
    for name in ['fixed002'] + [f'repeat{s}' for s in c['repeat_seeds']]:
        frame = pd.read_csv(folder / f'{name}_oof.csv')
        assert len(frame) == 2760 and frame.p_train8.between(0, 1).all()


def test_artifacts_if_present():
    path = HERE / 'metrics.json'
    if not path.exists(): pytest.skip('OOF not run yet')
    metrics = json.loads(path.read_text(encoding='utf8'))
    assert len(metrics['fit_seconds']) == 30
    assert metrics['provenance']['model_fits'] == 30
    assert metrics['provenance']['public_arrays_loaded'] is False
