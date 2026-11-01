import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp061', HERE / 'run_experiment.py')
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))


def test_tie_zero_and_dimension():
    x = np.zeros((2, 30, 2500))
    out = exp.ordinal_features(x, CONFIG)
    assert out.shape == (2, 240)
    np.testing.assert_array_equal(out, 0)
    assert len(exp.column_sets()['control_pe']) == 720
    assert len(exp.column_sets()['plus_ordinal']) == 960
    np.testing.assert_array_equal(exp.ordinal_codes(np.full(20, 7.0)), -1)
    np.testing.assert_array_equal(exp.transition_pair_features(np.full(20, 7.0)), 0)
    np.testing.assert_array_equal(exp.transition_pair_features(np.array([0., 1., 1., 2.])), 0)


def test_invariances_and_reverse_js():
    x = np.random.default_rng(61).normal(size=250)
    f = exp.transition_pair_features(x)
    np.testing.assert_allclose(f[1], exp.transition_pair_features(x[::-1])[1], atol=1e-14)
    np.testing.assert_allclose(f, exp.transition_pair_features(7.0 + 3.0 * x), atol=1e-14)
    for sequence in (np.arange(20.), np.array([0., 4., 1., 3., 2., 7., 5., 6.]), np.array([0., 0., 2., 1., 3., 5., 4.])):
        a = exp.transition_pair_features(sequence)
        np.testing.assert_allclose(a[1], exp.transition_pair_features(sequence[::-1])[1], atol=1e-14)
        np.testing.assert_allclose(a, exp.transition_pair_features(3 * sequence + 7), atol=1e-14)


def test_manual_p_q_direct_reverse():
    x = np.array([0., 2., 1., 3., 0., 2., 1.])
    got = exp.transition_pair_features(x)
    c, r = np.array([1, 2, 4, 3, 1]), np.array([4, 2, 1, 3, 4])
    np.testing.assert_array_equal(exp.ordinal_codes(x), c)
    np.testing.assert_array_equal(exp.ordinal_codes(x[::-1]), r)
    p = np.zeros((6, 6)); np.add.at(p, (c[:-1], c[1:]), 1); p /= p.sum()
    q = np.zeros((6, 6)); valid = (r[:-1] >= 0) & (r[1:] >= 0); np.add.at(q, (r[:-1][valid], r[1:][valid]), 1); q /= q.sum()
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got[1], exp.js_divergence(p.ravel(), q.ravel()))


def test_window_reorder_and_shape():
    x = np.random.default_rng(6101).normal(size=(3, 30, 2500))
    np.testing.assert_allclose(exp.ordinal_features(x[[2, 0, 1]], CONFIG), exp.ordinal_features(x, CONFIG)[[2, 0, 1]])
    with pytest.raises(ValueError): exp.ordinal_features(np.zeros((1, 30, 2499)), CONFIG)


def test_real_data_slice():
    from emo_rec.data import load_train_segments
    path = sorted((exp.ROOT / '数据/训练集').glob('*/*.mat'))[0]
    x, _, _ = load_train_segments(path); x = x[:3]
    a = exp.ordinal_features(x, CONFIG)
    assert a.shape == (3, 240) and np.isfinite(a).all()
    np.testing.assert_allclose(a[[2, 0, 1]], exp.ordinal_features(x[[2, 0, 1]], CONFIG))
    np.testing.assert_allclose(a[:1], exp.ordinal_features(x[:1], CONFIG))
    np.testing.assert_allclose(a, exp.ordinal_features(x * 3 + 7, CONFIG), atol=1e-12)


def test_oof_recomputed_metrics_and_subject_isolation_if_present():
    if not (HERE / 'metrics.json').exists(): pytest.skip('full experiment has not run yet')
    metrics = json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))
    for method in exp.METHODS:
        for split in ['fixed002'] + [f'repeat{s}' for s in CONFIG['repeat_seeds']]:
            frame = pd.read_csv(HERE / 'predictions' / f'{method}_{split}_oof.csv')
            folds = pd.read_csv(HERE / 'input_metadata' / f'folds_{split}.csv')
            pd.testing.assert_frame_equal(frame[folds.columns], folds)
            assert len(frame) == 2760 and frame.subject_id.nunique() == 69
            assert (frame.groupby('subject_id').size() == 40).all()
            exp.base.assert_subject_isolation(frame.subject_id.to_numpy(), frame.fold.to_numpy())
            assert frame.prob_positive.between(0, 1).all()
            recomputed = exp.base.summarize(frame.y_true.to_numpy(), frame.prob_positive.to_numpy(), frame, frame.fold.to_numpy())
            for key, value in exp.scalars(recomputed).items():
                assert abs(value - exp.scalars(metrics['runs'][method][split])[key]) < 1e-12
            if method == 'control_pe':
                old = pd.read_csv(exp.SOURCE / 'predictions' / f'plus_pe_{split}_oof.csv')
                np.testing.assert_allclose(old.prob_positive, frame.prob_positive, atol=1e-12, rtol=0)
        for key in exp.KEYS:
            values = []
            for seed in CONFIG['repeat_seeds']:
                frame = pd.read_csv(HERE / 'predictions' / f'{method}_repeat{seed}_oof.csv')
                values.append(exp.scalars(exp.base.summarize(frame.y_true.to_numpy(), frame.prob_positive.to_numpy(), frame, frame.fold.to_numpy()))[key])
            np.testing.assert_allclose(values, metrics['aggregate'][method][key]['values'], atol=1e-12)
            assert abs(np.mean(values) - metrics['aggregate'][method][key]['mean']) < 1e-12
            assert abs(np.std(values, ddof=1) - metrics['aggregate'][method][key]['std']) < 1e-12
    assert len(list((HERE / 'predictions').glob('*.csv'))) == 12
    source_config = json.loads((exp.SOURCE / 'config.json').read_text(encoding='utf8'))
    inputs = [exp.SOURCE / source_config[k] for k in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')]
    inputs += [exp.SOURCE / 'run_experiment.py', Path(exp.base.__file__)]
    artifacts = [HERE / n for n in ('config.json', 'test_protocol.txt', 'run_experiment.py', 'test_061.py', 'metrics.json', 'summary.csv', 'summary.md', CONFIG['ordinal_cache'])]
    artifacts += list((HERE / 'predictions').glob('*.csv')) + list((HERE / 'input_metadata').glob('folds_*.csv'))
    manifest = {'verified_tests': 'true data, deterministic ordinal invariants, subject isolation, OOF recomputation and aggregate, control reproduction <1e-12', 'inputs': {str(p.resolve()): exp.base.sha256(p) for p in inputs}, 'artifacts': {str(p.relative_to(HERE)): exp.base.sha256(p) for p in artifacts}}
    (HERE / 'input_metadata/verified_hashes.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
