import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('experiment033', HERE / 'run_experiment.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_feature_shape_finite_and_amplitude_invariance():
    x = np.random.default_rng(2026).normal(size=(2, 30, 2500))
    features = experiment.extract_detrended_features(x)
    assert features.shape == (2, 150)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features, experiment.extract_detrended_features(x * 10), atol=1e-12)
    assert np.isfinite(experiment.extract_detrended_features(np.zeros_like(x))).all()


@pytest.mark.parametrize('x', [np.zeros((1, 29, 2500)), np.zeros((1, 30, 2499)), np.zeros((0, 30, 2500)), np.full((1, 30, 2500), np.nan), np.full((1, 30, 2500), np.inf)])
def test_invalid_feature_input(x):
    with pytest.raises(ValueError):
        experiment.extract_detrended_features(x)


def test_cache_alignment_and_rejection():
    config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    spectral = experiment.load_cache(HERE / config['spectral_cache'])
    detrended = experiment.load_cache(experiment.CACHE)
    experiment.assert_cache_alignment(spectral, detrended)
    assert detrended['X_train'].shape == (2800, 150)
    assert detrended['X_public'].shape == (80, 150)
    metadata = json.loads(str(detrended['metadata_json']))
    assert len(metadata['sources']) == 80
    assert metadata['spectral_sha256'] == experiment.sha256(HERE / config['spectral_cache'])
    bad = dict(detrended); bad['original_trial_ids'] = detrended['original_trial_ids'][::-1]
    with pytest.raises(ValueError):
        experiment.assert_cache_alignment(spectral, bad)


def test_fixed_folds_subject_isolation():
    config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    spectral = experiment.load_cache(HERE / config['spectral_cache'])
    keep, frame = experiment.load_reference(spectral, config)
    assert keep.sum() == 2760
    assert 'HC43' not in set(frame.subject_id)
    for fold in range(5):
        assert not set(frame.loc[frame.fold == fold, 'subject_id']) & set(frame.loc[frame.fold != fold, 'subject_id'])
    bad = frame.copy(); bad.loc[0, 'fold'] = (bad.loc[0, 'fold'] + 1) % 5
    with pytest.raises(ValueError):
        experiment.assert_subject_isolation(bad)


@pytest.mark.parametrize('name', ['spectral_control', 'detrended150'])
def test_prediction_outputs(name):
    oof = pd.read_csv(HERE / 'predictions' / f'{name}_oof.csv')
    public = pd.read_csv(HERE / 'predictions' / f'{name}_public.csv')
    assert len(oof) == 2760 and len(public) == 80
    experiment.assert_subject_isolation(oof)
    for p in (oof.prob_positive.to_numpy(), public['prob. of positive emotion'].to_numpy()):
        assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    experiment.validate_submission(public, [f'P_test{i}' for i in range(1, 11)])
    config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    reference = pd.read_csv(HERE / config['reference_oof'])
    pd.testing.assert_frame_equal(oof.drop(columns='prob_positive'), reference.drop(columns='prob_positive'))
    result = json.loads((HERE / 'metrics.json').read_text(encoding='utf-8'))
    assert experiment.metrics(oof.y_true, oof.prob_positive)['auc'] == pytest.approx(result['runs'][name]['overall']['auc'])
