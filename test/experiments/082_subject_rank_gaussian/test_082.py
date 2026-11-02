import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp082', HERE / 'run_experiment.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture(scope='module')
def data():
    c = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    return c, m.load_data(c)


def test_frozen_protocol_and_shape(data):
    c, (raw, ref, fixed) = data
    m.validate_config(c)
    assert m.FEATURE_BLOCKS == (150, 450, 120)
    assert raw.shape == (2760, 720)
    assert 'HC43' not in set(ref.subject) and ref.subject.nunique() == 69
    assert set(ref.position) == set(range(5))
    assert (ref.groupby(['subject', 'position']).size() == 8).all()
    baseline, baseline_ref, _ = m.e.load_data(c)
    np.testing.assert_array_equal(raw, baseline)
    pd.testing.assert_frame_equal(ref, baseline_ref)


def test_formula_and_average_ties():
    from scipy.special import ndtri
    assert m.N_CONTEXT == 8
    values = np.tile(np.arange(8, dtype=float)[:, None], (1, 720))
    values[:, 1] = 7
    values[:, 2] *= 1e-14
    values[:, 3] = [0, 0, 0, 0, 0, 0, 0, 100]
    raw = np.tile(values, (5, 1))
    subjects, positions = np.repeat('subject', 40), np.repeat(np.arange(5), 8)
    got = m.rank_context8(raw, subjects, positions)
    ranks = np.tile(np.arange(1, 9, dtype=float)[:, None], (1, 720))
    ranks[:, 1] = 4.5
    ranks[:, 3] = [4, 4, 4, 4, 4, 4, 4, 8]
    expected = ndtri((ranks - 0.5) / 8)
    assert np.all(expected[:, 1] == 0)
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(got, np.tile(expected, (5, 1)), rtol=0, atol=1e-15)
    with pytest.raises(AssertionError):
        m.rank_context8(raw[:-1], subjects[:-1], positions[:-1])
    with pytest.raises(AssertionError):
        m.rank_context8(raw[:, :-1], subjects, positions)


def test_permutation_label_free_and_context_isolation(data):
    import inspect
    _, (raw, ref, _) = data
    subjects, positions = ref.subject.to_numpy(), ref.position.to_numpy()
    assert list(inspect.signature(m.rank_context8).parameters) == ['raw', 'subjects', 'positions']
    got = m.rank_context8(raw, subjects, positions)
    assert got.shape == (2760, 720) and np.isfinite(got).all()
    perm = np.random.default_rng(81).permutation(len(raw))
    np.testing.assert_array_equal(m.rank_context8(raw[perm], subjects[perm], positions[perm]), got[perm])
    relabeled = ref.copy()
    relabeled['y'] = 1 - relabeled.y
    relabeled['trial'] = -999
    np.testing.assert_array_equal(m.rank_context8(raw, relabeled.subject.to_numpy(), relabeled.position.to_numpy()), got)
    for subject in np.unique(subjects):
        for position in range(5):
            ix = (subjects == subject) & (positions == position)
            ranks = pd.DataFrame(raw[ix]).rank(axis=0, method='average').to_numpy()
            np.testing.assert_array_equal(got[ix], m.ndtri((ranks - 0.5) / 8))
    keep = (subjects == subjects[0]) & (positions == 0)
    changed = raw.copy()
    changed[~keep] += 1000
    np.testing.assert_array_equal(m.rank_context8(changed, subjects, positions)[keep], got[keep])


def test_job_uses_a8_for_both_fit_and_predict(monkeypatch):
    x = np.arange(60).reshape(12, 5)
    y = np.arange(12) % 2
    folds = np.arange(12) % 3
    class Spy:
        def fit(self, xx, yy):
            np.testing.assert_array_equal(xx, x[folds != 1])
            np.testing.assert_array_equal(yy, y[folds != 1])
        def predict_proba(self, xx):
            np.testing.assert_array_equal(xx, x[folds == 1])
            return np.tile([0.4, 0.6], (len(xx), 1))
    monkeypatch.setattr(m.base, 'make_model', lambda config, seed: Spy())
    result = m.job(x, y, 'test', folds, 1, 'rank8', {'outer_model_seed_base': 2052})
    np.testing.assert_array_equal(result[3], np.full(4, 0.6))


def test_splits_match_070(data):
    c, (_, ref, fixed) = data
    splits = m.e.splits_for(ref, fixed, c)
    assert [n for n, _ in splits] == ['fixed002'] + [f'repeat{s}' for s in range(2042, 2047)]
    for name, folds in splits:
        m.base.assert_subject_isolation(ref.subject.to_numpy(), folds)
        assert set(folds) == set(range(5))
        old = pd.read_csv(m.DEP / 'predictions' / f'{name}_oof.csv')
        for col in m.IDENTITY:
            np.testing.assert_array_equal(ref.assign(fold=folds)[col], old[col])


def test_no_public_array_access(monkeypatch, data):
    real_load = np.load
    allowed = {'X_train', 'subject_ids', 'health_groups', 'original_trial_ids', 'y_train', 'feature_names'}
    class Guard:
        def __init__(self, path, **kwargs):
            self.cache = real_load(path, **kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.cache.close()
        def __getitem__(self, key):
            assert key in allowed
            return self.cache[key]
    monkeypatch.setattr(m.np, 'load', Guard)
    m.load_data(data[0])


def test_acceptance_requires_all_conditions(data):
    c = data[0]
    d = {k: {'mean': 0.006, 'values': [0.006] * 5} for k in m.KEYS}
    a = {'all_rows': {'delta_rank8_minus_control8': d}}
    assert m.acceptance(a, c)['accepted']
    for key in ('auc', 'f1', 'hc_auc', 'dep_auc'):
        previous = d[key]['mean']
        d[key]['mean'] = -0.006
        assert not m.acceptance(a, c)['accepted']
        d[key]['mean'] = previous
    for key in ('auc', 'f1'):
        d[key]['values'] = [-0.001, -0.001, 0.01, 0.01, 0.012]
        assert not m.acceptance(a, c)['accepted']
        d[key]['values'] = [0.006] * 5
    d['auc']['values'] = [-0.001, 0.01, 0.01, 0.01, 0.01]
    d['f1']['values'] = [0.01, -0.001, 0.01, 0.01, 0.01]
    assert not m.acceptance(a, c)['accepted']


def test_artifacts_when_present(data):
    if not (HERE / 'metrics.json').exists():
        pytest.skip('OOF not run yet')
    metrics = json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))
    c, (_, ref, fixed) = data
    assert len(metrics['fit_seconds']) == 30
    assert metrics['provenance']['fit_roles'] == {'rank8': 30}
    assert metrics['provenance']['model_fits'] == 30
    assert max(metrics['control_070_max_abs_diff'].values()) <= 1e-12
    assert len({(r['split'], r['fold']) for r in metrics['fit_seconds']}) == 30
    assert metrics['provenance']['public_arrays_loaded'] is False
    for fit in metrics['fit_seconds']:
        assert fit['train_context'] == fit['validation_context'] == 8
    recomputed = {}
    for name, _ in m.e.splits_for(ref, fixed, c):
        frame = pd.read_csv(HERE / 'predictions' / f'{name}_oof.csv')
        old = pd.read_csv(m.DEP / 'predictions' / f'{name}_oof.csv')
        for col in m.IDENTITY:
            np.testing.assert_array_equal(frame[col], old[col])
        np.testing.assert_allclose(frame.p_train8, old.p_train8, rtol=0, atol=1e-12)
        assert set(frame.columns) == set(m.IDENTITY) | {'p_train8', 'p_rank8'}
        assert ((frame[['p_train8', 'p_rank8']] >= 0) & (frame[['p_train8', 'p_rank8']] <= 1)).all().all()
        recomputed[name] = {}
        for subset in ['all_rows'] + [str(i) for i in c['positions']]:
            q = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            recomputed[name][subset] = {arm: m.summarize(q, col) for arm, col in (('control8', 'p_train8'), ('rank8', 'p_rank8'))}
            for arm in ('control8', 'rank8'):
                for key, value in m.scalars(recomputed[name][subset][arm]).items():
                    assert value == pytest.approx(m.scalars(metrics['runs'][name][subset][arm])[key], abs=1e-12)
    aggregate = m.aggregate_runs(recomputed, c)
    assert m.acceptance(aggregate, c) == metrics['acceptance']
    for key in m.KEYS:
        np.testing.assert_allclose(aggregate['all_rows']['delta_rank8_minus_control8'][key]['values'], metrics['aggregate']['all_rows']['delta_rank8_minus_control8'][key]['values'], atol=1e-12, rtol=0)
    for section in ('inputs', 'sources'):
        for path, digest in metrics['provenance'][section].items():
            assert m.base.sha256(Path(path)) == digest
