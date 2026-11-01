import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import stft
from scipy.spatial.distance import jensenshannon

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp064', HERE / 'run_experiment.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
C = json.loads((HERE / 'config.json').read_text(encoding='utf8'))


def test_direct_reference():
    x = np.random.default_rng(64).normal(size=(2, 30, 2500))
    actual = e.spectral_js_features(x, C)
    f, _, z = stft(x, fs=250, window='hann', nperseg=250, noverlap=125, nfft=250, boundary=None, padded=False, detrend=False)
    power = abs(z[..., (f >= 1) & (f <= 45), :]) ** 2
    p = power / power.sum(axis=-2, keepdims=True)
    direct = np.empty((2, 30, 18))
    for n in range(2):
        for ch in range(30):
            for t in range(18):
                a, b = p[n, ch, :, t], p[n, ch, :, t + 1]
                m = (a + b) / 2
                direct[n, ch, t] = .5 * np.sum(a * np.log2(a / m) + b * np.log2(b / m))
                np.testing.assert_allclose(direct[n, ch, t], jensenshannon(a, b, base=2) ** 2, atol=1e-14)
    np.testing.assert_allclose(actual, np.column_stack([direct.mean(-1), direct.std(-1, ddof=0)]), atol=1e-14)
    assert len(e.feature_names()) == 60


def test_scale_sign_zero_batch():
    x = np.random.default_rng(65).normal(size=(3, 30, 2500))
    a = e.spectral_js_features(x, C)
    np.testing.assert_allclose(a, e.spectral_js_features(x * -7.3, C), atol=1e-14)
    np.testing.assert_allclose(a, np.concatenate([e.spectral_js_features(v[None], C) for v in x]), atol=1e-14)
    zero = e.spectral_js_features(np.zeros_like(x), C)
    assert np.isfinite(zero).all(); np.testing.assert_array_equal(zero, 0)
    mixed = np.concatenate([x[:1], np.zeros_like(x[:1]), x[1:]])
    np.testing.assert_allclose(e.spectral_js_features(mixed, C)[[0, 2, 3]], a, atol=1e-14)


def test_real_rows_folds():
    old = json.loads((e.SOURCE / 'config.json').read_text(encoding='utf8'))
    A, AP, ref, s, p = e.source.load_data(old)
    assert A.shape == (2760, 720) and ref.subject_id.nunique() == 69 and 'HC43' not in set(ref.subject_id)
    for seed in [None] + C['repeat_seeds']:
        folds = ref.fold.to_numpy() if seed is None else e.base.make_subject_folds(ref.subject_id.to_numpy(), ref.health_group.to_numpy(), seed)
        e.base.assert_subject_isolation(ref.subject_id.to_numpy(), folds)
        assert set(folds) == set(range(5))
        name = 'fixed002' if seed is None else f'repeat{seed}'
        saved = pd.read_csv(e.SOURCE / 'predictions' / f'plus_pe_{name}_oof.csv')
        pd.testing.assert_frame_equal(saved[ref.columns], ref.assign(fold=folds))
    if (HERE / C['spectral_js_cache']).exists():
        B, rr, _, o = e.load_data(C)
        np.testing.assert_array_equal(B[:, :720], A)
        pd.testing.assert_frame_equal(rr, ref)
        from emo_rec.data import load_train_segments
        from emo_rec.normalization import normalize_by_subject
        files = sorted((e.ROOT / '数据/训练集').glob('*/*.mat'))
        for f in (files[0], files[-1]):
            seg, y, trial = load_train_segments(f)
            mask = o['subject_ids'] == f.stem
            np.testing.assert_array_equal(o['y_train'][mask], y)
            np.testing.assert_array_equal(o['original_trial_ids'][mask], trial)
            np.testing.assert_allclose(o['X_train'][mask], e.spectral_js_features(seg, C), atol=1e-14)
        keep = s['subject_ids'] != 'HC43'
        np.testing.assert_array_equal(B[:, 720:], normalize_by_subject(o['X_train'][keep], s['subject_ids'][keep], 'zscore'))
        bad = dict(o); bad['original_trial_ids'] = o['original_trial_ids'][::-1]
        with pytest.raises(RuntimeError): e.source.assert_alignment(s, bad)


def compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for k in expected: compare(actual[k], expected[k])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected): compare(a, b)
    elif isinstance(expected, (float, int)):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    else: assert actual == expected


@pytest.mark.skipif(not (HERE / 'metrics.json').exists(), reason='requires real run artifacts')
def test_all_metrics_pairing_control():
    metrics = json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))
    assert len(list((HERE / 'predictions').glob('*.csv'))) == 12
    summary = pd.read_csv(HERE / 'summary.csv')
    recomputed = {m: {} for m in e.METHODS}
    for name in ['fixed002'] + [f'repeat{s}' for s in C['repeat_seeds']]:
        ref = pd.read_csv(HERE / 'input_metadata' / f'folds_{name}.csv')
        assert len(ref) == 2760 and ref.subject_id.nunique() == 69
        e.base.assert_subject_isolation(ref.subject_id.to_numpy(), ref.fold.to_numpy())
        for m in e.METHODS:
            f = pd.read_csv(HERE / 'predictions' / f'{m}_{name}_oof.csv')
            pd.testing.assert_frame_equal(f[ref.columns], ref)
            assert np.isfinite(f.prob_positive).all()
            result = e.base.summarize(f.y_true.to_numpy(), f.prob_positive.to_numpy(), ref, ref.fold.to_numpy())
            compare(result, metrics['runs'][m][name]); recomputed[m][name] = e.scalars(result)
            row = summary[(summary.method == m) & (summary.split == name)].iloc[0]
            for k in e.KEYS: compare(row[k], recomputed[m][name][k])
            if m == 'control_pe':
                old = pd.read_csv(e.SOURCE / 'predictions' / f'plus_pe_{name}_oof.csv')
                pd.testing.assert_frame_equal(old[ref.columns], ref)
                np.testing.assert_allclose(f.prob_positive, old.prob_positive, atol=1e-12, rtol=0)
    for m in e.METHODS:
        for k in e.KEYS:
            v = [recomputed[m][f'repeat{s}'][k] for s in C['repeat_seeds']]
            compare(metrics['aggregate'][m][k], {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)), 'values': v})
            row = summary[(summary.method == m) & (summary.split == 'repeat_mean')].iloc[0]
            compare(row[k], np.mean(v))
    d = metrics['deltas_vs_control']['plus_spectral_js']
    for k in e.KEYS:
        compare(d['repeat_mean'][k], metrics['aggregate']['plus_spectral_js'][k]['mean'] - metrics['aggregate']['control_pe'][k]['mean'])
        compare(d['fixed002'][k], recomputed['plus_spectral_js']['fixed002'][k] - recomputed['control_pe']['fixed002'][k])
    compare(d['repeat_paired_auc'], [recomputed['plus_spectral_js'][f'repeat{s}']['auc'] - recomputed['control_pe'][f'repeat{s}']['auc'] for s in C['repeat_seeds']])
    assert metrics['decision_accept']['plus_spectral_js'] == e.accepts(d['repeat_mean'], d['fixed002']['auc'], C)
    for item in metrics['provenance']['inputs']: assert e.base.sha256(Path(item['path'])) == item['sha256']
    raw = json.loads((HERE / 'input_metadata/raw_inputs.json').read_text(encoding='utf8'))
    assert len(raw) == 70
    for item in raw: assert e.base.sha256(e.ROOT / item['path']) == item['sha256']
