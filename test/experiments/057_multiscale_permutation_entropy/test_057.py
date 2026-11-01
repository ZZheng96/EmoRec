import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp057', HERE / 'run_experiment.py')
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))


def test_entropy_known_signals_and_bounds():
    rng = np.random.default_rng(57)
    noise = exp.permutation_entropy(rng.normal(size=(30, 1250)), 3, 1)
    assert (noise > .98).all() and (noise <= 1).all()
    for x in (np.zeros((2, 1250)), np.tile(np.arange(1250), (2, 1))):
        np.testing.assert_allclose(exp.permutation_entropy(x, 3, 1), 0)
    np.testing.assert_allclose(exp.permutation_entropy(np.array([[0, 1, 0, 1, 0, 1]]), 2, 1),
                               -(0.6*np.log(.6)+.4*np.log(.4))/np.log(2))


def test_shape_window_independence_and_row_equivariance():
    x = np.random.default_rng(1).normal(size=(3, 30, 2500))
    a = exp.pe_features(x, CONFIG)
    assert a.shape == (3, 120) and np.isfinite(a).all()
    assert ((a >= 0) & (a <= 1)).all()
    np.testing.assert_array_equal(a[[2, 0, 1]], exp.pe_features(x[[2, 0, 1]], CONFIG))
    np.testing.assert_array_equal(a[:1], exp.pe_features(x[:1], CONFIG))
    x[1:] = 1e5
    np.testing.assert_array_equal(a[:1], exp.pe_features(x, CONFIG)[:1])


@pytest.mark.parametrize('shape', [(1, 29, 2500), (1, 30, 2499), (30, 2500)])
def test_bad_shape_rejected(shape):
    with pytest.raises(ValueError):
        exp.pe_features(np.zeros(shape), CONFIG)


def test_alignment_rejects_each_metadata_mismatch():
    a = {k: np.arange(4) for k in exp.ALIGN_KEYS}
    exp.assert_alignment(a, a)
    for key in a:
        b = {k: v.copy() for k, v in a.items()}
        b[key] = b[key][::-1]
        with pytest.raises(RuntimeError):
            exp.assert_alignment(a, b)


def test_protocol_dimensions_parameters_and_acceptance():
    old = json.loads((HERE.parent / '056_theta_beta_pac/config.json').read_text(encoding='utf8'))
    for key in ('hgb', 'repeat_seeds', 'outer_model_seed_base', 'threshold', 'fixed_reference'):
        assert CONFIG[key] == old[key]
    assert CONFIG['pe']['scales'] == [1, 2, 4, 8]
    assert CONFIG['pe']['embedding_order'] == 3
    assert {k: len(v) for k, v in exp.column_sets().items()} == {'control_600': 600, 'pe_only': 120, 'plus_pe': 720}
    assert exp.accepts({'auc': .005, 'f1': 0, 'dep_auc': 0}, .001, CONFIG)
    for d, fixed in [({'auc': .0049, 'f1': 0, 'dep_auc': 0}, .001), ({'auc': .006, 'f1': -.001, 'dep_auc': 0}, .001), ({'auc': .006, 'f1': 0, 'dep_auc': -.001}, .001), ({'auc': .006, 'f1': 0, 'dep_auc': 0}, 0)]:
        assert not exp.accepts(d, fixed, CONFIG)


def test_subject_isolation_and_normalization():
    from emo_rec.normalization import normalize_by_subject
    ids = np.repeat([f's{i}' for i in range(20)], 4)
    groups = np.repeat(np.arange(20) % 2, 4)
    x = np.random.default_rng(3).normal(size=(80, 3))
    z = normalize_by_subject(x, ids, 'zscore')
    x[ids != 's0'] += 1000
    np.testing.assert_allclose(z[ids == 's0'], normalize_by_subject(x, ids, 'zscore')[ids == 's0'])
    for seed in CONFIG['repeat_seeds']:
        folds = exp.base.make_subject_folds(ids, groups, seed)
        exp.base.assert_subject_isolation(ids, folds)
        np.testing.assert_array_equal(folds, exp.base.make_subject_folds(ids, groups, seed))
    folds[0] = (folds[1]+1) % 5
    with pytest.raises(RuntimeError):
        exp.base.assert_subject_isolation(ids, folds)


def test_saved_outputs_if_present():
    if not (HERE / 'metrics.json').exists():
        pytest.skip('full experiment has not run yet')
    m = json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))
    assert m['provenance']['public_metrics'] is False
    for method in exp.METHODS:
        for split in ['fixed002'] + [f'repeat{s}' for s in CONFIG['repeat_seeds']]:
            f = pd.read_csv(HERE / 'predictions' / f'{method}_{split}_oof.csv')
            assert len(f) == 2760 and f.subject_id.nunique() == 69
            assert 'HC43' not in set(f.subject_id) and f.prob_positive.between(0, 1).all()
            exp.base.assert_subject_isolation(f.subject_id.to_numpy(), f.fold.to_numpy())
        p = pd.read_csv(HERE / 'predictions' / f'{method}_public.csv')
        assert len(p) == 80 and list(p.columns) == ['user_id', 'trial_id', 'prob. of positive emotion']
        assert p.iloc[:, 2].between(0, 1).all()
    assert max(m['control_reproduction_max_abs_diff_vs_056'].values()) < 1e-12
