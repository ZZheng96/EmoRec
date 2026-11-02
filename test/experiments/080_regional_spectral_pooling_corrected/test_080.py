import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp080', HERE / 'run_experiment.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture(scope='module')
def data():
    c = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    return c, m.load_data(c)


def test_frozen_protocol_and_mapping(data):
    c, (raw, treatment, ref, fixed) = data
    m.validate_config(c)
    assert m.REGIONS == m.old.REGIONS
    assert [len(v) for v in m.REGIONS.values()] == [7, 5, 5, 10, 3]
    assert len(set(sum(m.REGIONS.values(), []))) == 30
    assert len(m.FEATURE_NAMES) == 25
    assert raw.shape == (2760, 720) and treatment.shape == (2760, 595)
    assert 'HC43' not in set(ref.subject) and ref.subject.nunique() == 69
    np.testing.assert_array_equal(raw[:, 150:], treatment[:, 25:])
    for ri, chans in enumerate(m.REGIONS.values()):
        for bi in range(5):
            expected = raw[:, [m.CHANNELS.index(ch) * 5 + bi for ch in chans]].mean(axis=1)
            np.testing.assert_array_equal(treatment[:, ri * 5 + bi], expected)


def test_a8_context_and_unchanged_other_blocks(data):
    _, (raw, treatment, ref, _) = data
    subjects, positions = ref.subject.to_numpy(), ref.position.to_numpy()
    region8 = m.region_context8(treatment, subjects, positions)
    control8 = np.empty_like(raw)
    for position in range(5):
        ix = positions == position
        control8[ix] = np.column_stack([m.old.normalize_by_subject(raw[ix, a:b], subjects[ix], 'zscore') for a, b in ((0, 150), (150, 600), (600, 720))])
    np.testing.assert_array_equal(region8[:, 25:], control8[:, 150:])
    for position in range(5):
        ix = positions == position
        assert (ref[ix].groupby('subject').size() == 8).all()
        expected = np.column_stack([m.old.normalize_by_subject(treatment[ix, a:b], subjects[ix], 'zscore') for a, b in ((0, 25), (25, 475), (475, 595))])
        np.testing.assert_array_equal(region8[ix], expected)
    changed = treatment.copy()
    changed[positions != 0] += 1000
    np.testing.assert_array_equal(m.region_context8(changed, subjects, positions)[positions == 0], region8[positions == 0])


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
    result = m.job(x, y, 'test', folds, 1, 'region8', {'outer_model_seed_base': 2052})
    np.testing.assert_array_equal(result[3], np.full(4, 0.6))


def test_splits_match_070(data):
    c, (_, _, ref, fixed) = data
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
    a = {'all_rows': {'delta_region8_minus_control8': d}}
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
    c, (_, _, ref, fixed) = data
    assert len(metrics['fit_seconds']) == 30
    assert metrics['provenance']['fit_roles'] == {'region8': 30}
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
        assert set(frame.columns) == set(m.IDENTITY) | {'p_train8', 'p_region8'}
        assert ((frame[['p_train8', 'p_region8']] >= 0) & (frame[['p_train8', 'p_region8']] <= 1)).all().all()
        recomputed[name] = {}
        for subset in ['all_rows'] + [str(i) for i in c['positions']]:
            q = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            recomputed[name][subset] = {arm: m.summarize(q, col) for arm, col in (('control8', 'p_train8'), ('region8', 'p_region8'))}
            for arm in ('control8', 'region8'):
                for key, value in m.scalars(recomputed[name][subset][arm]).items():
                    assert value == pytest.approx(m.scalars(metrics['runs'][name][subset][arm])[key], abs=1e-12)
    aggregate = m.aggregate_runs(recomputed, c)
    assert m.acceptance(aggregate, c) == metrics['acceptance']
    for key in m.KEYS:
        np.testing.assert_allclose(aggregate['all_rows']['delta_region8_minus_control8'][key]['values'], metrics['aggregate']['all_rows']['delta_region8_minus_control8'][key]['values'], atol=1e-12, rtol=0)
    for section in ('inputs', 'sources'):
        for path, digest in metrics['provenance'][section].items():
            assert m.base.sha256(Path(path)) == digest
