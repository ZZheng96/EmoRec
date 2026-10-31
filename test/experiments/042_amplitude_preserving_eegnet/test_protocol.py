import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('experiment042', HERE / 'run_experiment.py')
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def artifacts():
    return json.loads((HERE / 'metrics.json').read_text(encoding='utf-8'))


def test_input_shape_finite_and_metadata():
    frame = exp.metadata()
    assert len(frame) == 2760 and frame.subject_id.nunique() == 69
    assert 'HC43' not in set(frame.subject_id)
    assert 'original_trial_id' not in frame
    for name, n in [('train', 2760), ('public', 80)]:
        x = np.load(HERE / 'cache' / f'{name}.npy', mmap_mode='r')
        assert x.shape == (n, 30, 2000)
        for chunk in np.array_split(x, 40):
            assert np.isfinite(chunk).all()
    meta = json.loads((HERE / 'input_metadata.json').read_text(encoding='utf-8'))
    assert meta['labels_used_in_preprocessing'] is False
    assert len(meta['raw_sha256']) == 79


def test_window_preprocessing_is_unlabeled_independent_and_permutation_equivariant():
    raw = np.random.default_rng(8).normal(size=(3, 30, 2500))
    a = exp.preprocess_windows(raw)
    permutation = np.array([2, 0, 1])
    np.testing.assert_array_equal(exp.preprocess_windows(raw[permutation]), a[permutation])
    raw[1:] *= 100
    np.testing.assert_array_equal(exp.preprocess_windows(raw[:1])[0], a[0])
    with pytest.raises(ValueError):
        exp.preprocess_windows(np.full((1, 30, 2500), np.nan))


def test_normalization_preserves_amplitude_and_uses_training_only_scale():
    x = np.random.default_rng(9).normal(size=(6, 30, 2000)).astype(np.float32)
    scale = exp.fit_scale(x[:4])
    before = scale.copy()
    a = exp.transform(x[4:], 'amplitude_preserving', scale)
    b = exp.transform(3*x[4:], 'amplitude_preserving', scale)
    np.testing.assert_allclose(b, 3*a, atol=3e-6)
    np.testing.assert_array_equal(scale, before)
    z = exp.transform(x, 'per_window_zscore')
    np.testing.assert_allclose(z.std(axis=2), 1, atol=1e-6)
    np.testing.assert_allclose(a.mean(axis=2), 0, atol=1e-6)
    np.testing.assert_allclose(exp.transform(3*x, 'per_window_zscore'), z, atol=2e-6)
    with pytest.raises(ValueError):
        exp.transform(x, 'amplitude_preserving')


def test_reordering_restores_by_metadata_not_row_position():
    frame = pd.read_csv(HERE / 'cache' / 'rows.csv')
    x = np.load(HERE / 'cache' / 'train.npy', mmap_mode='r')
    take = np.array([0, 41, 82, 123, 164, 205])
    metadata = frame.iloc[take].reset_index(drop=True)
    values = x[take]
    order = np.array([5, 0, 3, 1, 4, 2])
    restored = exp.align_windows(values[order], metadata.iloc[order], metadata)
    np.testing.assert_array_equal(restored, values)
    fake = metadata.copy(); fake['y_true'] = 1 - fake.y_true
    np.testing.assert_array_equal(exp.align_windows(values, fake, metadata), values)


def test_saved_amplitude_scale_is_fitted_on_training_fold_only(artifacts):
    frame = exp.metadata()
    x = np.load(HERE / 'cache' / 'train.npy', mmap_mode='r')
    expected = exp.fit_scale(x[frame.fold.to_numpy() != 0]).ravel()
    saved = artifacts['runs']['amplitude_preserving_fixed']['training']['0']['training_channel_scale']
    np.testing.assert_array_equal(expected, np.asarray(saved, dtype=np.float32))


def test_model_and_training_match_024():
    old = json.loads((HERE.parent / '024_eegnet_cnn' / 'config.json').read_text(encoding='utf-8'))
    for key in ['epochs', 'batch_size', 'lr', 'weight_decay']:
        assert CONFIG['training'][key] == old['training'][key]
    exp.base.seed_all(2026)
    first = exp.base.EEGNet().eval()
    exp.base.seed_all(2026)
    second = exp.base.EEGNet().eval()
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)
    with torch.no_grad():
        assert first(torch.zeros(2, 1, 30, 2000)).shape == (2, 2)


@pytest.mark.parametrize('seed', [2027, 2028, 2029, 2030, 2031])
def test_repeat_seeds_match_032(seed):
    frame = exp.metadata()
    folds = exp.repeat.make_subject_folds(frame.subject_id.to_numpy(), frame.health_group.to_numpy(), seed)
    exp.repeat.assert_subject_isolation(frame.subject_id.to_numpy(), folds)
    old = pd.read_csv(HERE.parent / '032_repeated_subject_folds' / 'predictions' / f'folds_repeat{seed}.csv')
    assignment = old.groupby('subject_id').fold.first()
    np.testing.assert_array_equal(folds, frame.subject_id.map(assignment).to_numpy())


@pytest.mark.parametrize('condition', CONFIG['conditions'])
def test_complete_oof_probabilities_isolation_and_seed_results(condition, artifacts):
    frame = exp.metadata()
    for tag, split_seed in [('fixed', None)] + [(f'repeat{s}', s) for s in CONFIG['repeat_seeds']]:
        name = f'{condition}_{tag}'
        assert name in artifacts['runs'], f'Incomplete run: {name}'
        r = artifacts['runs'][name]
        out = pd.read_csv(HERE / 'predictions' / f'{name}_oof.csv')
        assert len(out) == 2760 and not out.duplicated(['subject_id', 'window_id']).any()
        assert np.isfinite(out.prob_positive).all() and out.prob_positive.between(0, 1).all()
        pd.testing.assert_frame_equal(out[frame.columns].drop(columns='fold'), frame.drop(columns='fold'))
        expected = frame.fold.to_numpy() if split_seed is None else exp.repeat.make_subject_folds(frame.subject_id.to_numpy(), frame.health_group.to_numpy(), split_seed)
        np.testing.assert_array_equal(out.fold.to_numpy(), expected)
        assert r['split_seed'] == split_seed and r['oof_counts'] == [1]
        exp.repeat.assert_subject_isolation(out.subject_id.to_numpy(), out.fold.to_numpy())
        scores = exp.repeat.classification_metrics(out.y_true.to_numpy(), out.prob_positive.to_numpy())
        for key in ['auc', 'f1', 'balanced_accuracy']:
            assert scores[key] == pytest.approx(r['overall'][key])
        for f in range(5):
            audit = r['training'][str(f)]
            assert audit['seed'] == 2026+f and len(audit['epoch_train_losses']) == 30
            assert set(audit['train_subjects']).isdisjoint(audit['valid_subjects'])
            assert set(audit['train_subjects']) == set(frame.loc[expected != f, 'subject_id'])
            if condition == 'amplitude_preserving':
                assert len(audit['training_channel_scale']) == 30
                assert np.all(np.asarray(audit['training_channel_scale']) > 0)
    assert not artifacts['pending']


@pytest.mark.parametrize('condition', CONFIG['conditions'])
def test_public_80_rows_no_labels(condition, artifacts):
    public = pd.read_csv(HERE / 'predictions' / f'{condition}_public.csv')
    assert list(public.columns) == ['user_id', 'trial_id', 'prob. of positive emotion']
    assert len(public) == 80
    exp.validate_submission(public, [f'P_test{i}' for i in range(1, 11)])
    assert artifacts['public_training'][condition]['seed'] == 2031
    assert artifacts['experiment_config']['protocol']['public_labels_used'] is False
