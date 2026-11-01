import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp062', HERE / 'run_experiment.py')
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))


def test_constants_and_shape():
    for value in (0., 7.):
        out = exp.oAEC_features(np.full((1, 30, 2500), value), CONFIG)
        assert out.shape == (1, 120) and np.isfinite(out).all()
    np.testing.assert_array_equal(exp.oAEC_features(np.zeros((1, 30, 2500)), CONFIG), 0)
    np.testing.assert_array_equal(exp.pearson_last(np.ones(20), np.arange(20.)), 0)
    np.testing.assert_array_equal(exp.orthogonal_envelope_matrix(np.ones((30, 2000), dtype=complex)), 0)
    assert len(exp.column_sets()['control_pe']) == 720
    assert len(exp.column_sets()['plus_oAEC']) == 840
    with pytest.raises(ValueError): exp.oAEC_features(np.zeros((1, 30, 2499)), CONFIG)
    with pytest.raises(ValueError): exp.oAEC_features(np.full((1, 30, 2500), np.nan), CONFIG)


def test_scaling_sign_and_window_order():
    rng = np.random.default_rng(62)
    x = rng.normal(size=(3, 30, 2500))
    f = exp.oAEC_features(x, CONFIG)
    factors = rng.uniform(0.3, 4, (1, 30, 1)) * rng.choice([-1, 1], (1, 30, 1))
    np.testing.assert_allclose(f, exp.oAEC_features(x * factors, CONFIG), atol=1e-12, rtol=0)
    np.testing.assert_allclose(f[[2, 0, 1]], exp.oAEC_features(x[[2, 0, 1]], CONFIG), atol=1e-12, rtol=0)


def test_direct_pair_reference_and_symmetry():
    rng = np.random.default_rng(6201)
    z = rng.normal(size=(30, 2000)) + 1j * rng.normal(size=(30, 2000))
    z[0] = 0
    expected = np.zeros((30, 30))
    for i in range(30):
        for j in range(i + 1, 30):
            correlations = []
            for a, b in ((z[i], z[j]), (z[j], z[i])):
                amplitude = np.abs(a)
                orthogonal = np.abs(np.imag(b * np.conj(a) / np.maximum(amplitude, np.finfo(np.float64).eps)))
                correlations.append(0. if amplitude.std() == 0 or orthogonal.std() == 0 else abs(np.corrcoef(amplitude, orthogonal)[0, 1]))
            expected[i, j] = expected[j, i] = np.mean(correlations)
    matrix = exp.orthogonal_envelope_matrix(z)
    np.testing.assert_array_equal(matrix, matrix.T)
    np.testing.assert_allclose(matrix, expected, atol=1e-12, rtol=0)


def test_real_data_reference_and_cache_order():
    from emo_rec.data import load_train_segments
    paths = sorted((exp.ROOT / '数据/训练集').glob('*/*.mat'))
    x, y, trial = load_train_segments(paths[0]); x = x[:3]
    a = exp.oAEC_features(x, CONFIG)
    assert a.shape == (3, 120) and np.isfinite(a).all()
    np.testing.assert_allclose(a[[2, 0, 1]], exp.oAEC_features(x[[2, 0, 1]], CONFIG), atol=1e-12, rtol=0)
    np.testing.assert_allclose(a[:1], exp.oAEC_features(x[:1], CONFIG), atol=1e-12, rtol=0)
    np.testing.assert_allclose(a, exp.oAEC_features(-3 * x, CONFIG), atol=1e-12, rtol=0)
    blocks = []
    for band in CONFIG['oAEC']['bands']:
        sos = exp.butter(4, band, btype='bandpass', fs=250, output='sos')
        z = exp.hilbert(exp.sosfiltfilt(sos, x, axis=-1), axis=-1)[..., 250:-250]
        blocks.append(np.array([exp.orthogonal_envelope_matrix(w).sum(axis=1) / 29 for w in z]))
    np.testing.assert_allclose(a, np.concatenate(blocks, axis=1), atol=1e-12, rtol=0)
    cache = HERE / CONFIG['oAEC_cache']
    if cache.exists():
        c = exp.base.load_cache(cache)
        np.testing.assert_allclose(c['X_train'][:3], a, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(c['subject_ids'][:3], [paths[0].stem] * 3)
        np.testing.assert_array_equal(c['y_train'][:3], y[:3])
        np.testing.assert_array_equal(c['original_trial_ids'][:3], trial[:3])
        assert 'X_public' not in c


def assert_nested_close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_nested_close(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_nested_close(x, y)
    elif isinstance(a, (float, np.floating)):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0, equal_nan=True)
    else:
        assert a == b


def test_all_oof_metrics_isolation_control_and_hashes():
    if not (HERE / 'metrics.json').exists(): pytest.skip('full experiment has not run yet')
    metrics = json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))
    ref = pd.read_csv((HERE / CONFIG['fixed_reference']).resolve())
    recomputed_runs = {}
    for method in exp.METHODS:
        recomputed_runs[method] = {}
        for split in ['fixed002'] + [f'repeat{s}' for s in CONFIG['repeat_seeds']]:
            frame = pd.read_csv(HERE / 'predictions' / f'{method}_{split}_oof.csv')
            folds = pd.read_csv(HERE / 'input_metadata' / f'folds_{split}.csv')
            pd.testing.assert_frame_equal(frame[folds.columns], folds)
            keys = ['subject_id', 'health_group', 'original_trial_id', 'y_true']
            pd.testing.assert_frame_equal(frame[keys], ref[keys])
            expected_folds = ref.fold.to_numpy() if split == 'fixed002' else exp.base.make_subject_folds(ref.subject_id.to_numpy(), ref.health_group.to_numpy(), int(split[6:]))
            np.testing.assert_array_equal(frame.fold, expected_folds)
            assert len(frame) == 2760 and frame.subject_id.nunique() == 69
            assert (frame.groupby('subject_id').size() == 40).all()
            exp.base.assert_subject_isolation(frame.subject_id.to_numpy(), frame.fold.to_numpy())
            assert frame.prob_positive.between(0, 1).all()
            result = exp.base.summarize(frame.y_true.to_numpy(), frame.prob_positive.to_numpy(), frame, frame.fold.to_numpy())
            assert_nested_close(result, metrics['runs'][method][split])
            recomputed_runs[method][split] = exp.scalars(result)
            if method == 'control_pe':
                old = pd.read_csv(exp.SOURCE / 'predictions' / f'plus_pe_{split}_oof.csv')
                pd.testing.assert_frame_equal(old[folds.columns], frame[folds.columns])
                np.testing.assert_allclose(old.prob_positive, frame.prob_positive, atol=1e-12, rtol=0)
        for key in exp.KEYS:
            values = [recomputed_runs[method][f'repeat{s}'][key] for s in CONFIG['repeat_seeds']]
            assert_nested_close({'values': values, 'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1))}, metrics['aggregate'][method][key])
    delta = metrics['deltas_vs_control']['plus_oAEC']
    for key in exp.KEYS:
        assert abs(delta['fixed002'][key] - (recomputed_runs['plus_oAEC']['fixed002'][key] - recomputed_runs['control_pe']['fixed002'][key])) < 1e-12
        assert abs(delta['repeat_mean'][key] - (metrics['aggregate']['plus_oAEC'][key]['mean'] - metrics['aggregate']['control_pe'][key]['mean'])) < 1e-12
    assert metrics['decision_accept']['plus_oAEC'] == exp.accepts(delta['repeat_mean'], delta['fixed002']['auc'], CONFIG)
    assert len(list((HERE / 'predictions').glob('*.csv'))) == 12
    assert len(list((HERE / 'input_metadata').glob('folds_*.csv'))) == 6
    source_config = json.loads((exp.SOURCE / 'config.json').read_text(encoding='utf8'))
    inputs = [exp.SOURCE / source_config[k] for k in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')]
    inputs += [exp.SOURCE / 'run_experiment.py', Path(exp.base.__file__)]
    artifacts = [HERE / n for n in ('config.json', 'test_protocol.txt', 'run_experiment.py', 'test_062.py', 'metrics.json', 'summary.csv', 'summary.md', CONFIG['oAEC_cache'])]
    artifacts += list((HERE / 'predictions').glob('*.csv')) + list((HERE / 'input_metadata').glob('folds_*.csv'))
    raw = json.loads((HERE / 'input_metadata/raw_inputs.json').read_text(encoding='utf8'))
    assert len(raw) == len(list((exp.ROOT / '数据/训练集').glob('*/*.mat')))
    for record in raw: assert exp.base.sha256(exp.ROOT / record['path']) == record['sha256']
    assert metrics['provenance']['inputs']['config'] == exp.base.sha256(HERE / 'config.json')
    manifest = {'verified_tests': 'real data, scaling/sign invariance, pair reference, subject isolation, full OOF metrics and aggregate, row alignment, control reproduction <1e-12', 'inputs': {str(p.resolve()): exp.base.sha256(p) for p in inputs}, 'artifacts': {str(p.relative_to(HERE)): exp.base.sha256(p) for p in artifacts}}
    (HERE / 'input_metadata/verified_hashes.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
