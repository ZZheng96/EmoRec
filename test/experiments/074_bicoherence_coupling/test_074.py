import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp074', HERE / 'run_experiment.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CONFIG = json.loads((HERE / 'config.json').read_text())


def test_fixed_frequency_formula_and_dimensions():
    assert m.bicoherence(np.ones((2, 30, 2500)), 'cpu').shape == (2, 150)
    assert np.allclose(m.bicoherence(np.ones((2, 30, 2500)), 'cpu'), 1)
    rng = np.random.default_rng(74)
    x = rng.normal(size=(2, 30, 2500))
    actual = m.bicoherence(x, 'cpu')
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[:1], m.bicoherence(x[:1], 'cpu'), atol=1e-14)
    x[1] *= 100
    np.testing.assert_allclose(actual[:1], m.bicoherence(x, 'cpu')[:1], atol=1e-14)


def test_artificial_second_phase_coupling():
    t = np.arange(2500) / 250
    x = np.sin(2 * np.pi * 6 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
    signal = np.tile(x, (2, 30, 1))
    result = m.bicoherence(signal, 'cpu')
    assert result[:, 0:30].mean() > 0.9


@pytest.mark.parametrize('shape', [(1, 30, 2499), (1, 29, 2500)])
def test_window_shape(shape):
    with pytest.raises(ValueError):
        m.bicoherence(np.zeros(shape), 'cpu')


def test_alignment_and_unlabelled_normalization():
    raw, ref, fixed = m.e.load_data(CONFIG)
    m.assert_alignment(ref, ref.copy())
    with pytest.raises(AssertionError):
        m.assert_alignment(ref, ref.iloc[::-1])
    rng = np.random.default_rng(74)
    features = rng.normal(size=(len(ref), 150))
    normalized = m.normalize_extra(features, ref.subject.to_numpy(), ref.position.to_numpy())
    shuffled = ref.copy()
    shuffled['y'] = 1 - shuffled.y
    np.testing.assert_array_equal(normalized, m.normalize_extra(features, shuffled.subject.to_numpy(), shuffled.position.to_numpy()))
    for _, indices in ref.groupby(['subject', 'position']).groups.items():
        np.testing.assert_allclose(normalized[indices].mean(0), 0, atol=1e-14)
        np.testing.assert_allclose(normalized[indices].std(0), 1, atol=1e-12)
    assert m.e.contexts(raw, ref.subject.to_numpy(), ref.position.to_numpy())[1].shape == (2760, 720)


def test_protocol_and_acceptance():
    assert CONFIG['triples_hz'] == [list(t) for t in m.TRIPLES]
    assert CONFIG['nperseg'] == 500 and CONFIG['noverlap'] == 250 and CONFIG['frequency_grid_hz'] == 1.0
    assert len(m.e.splits_for(m.e.load_data(CONFIG)[1], m.e.load_data(CONFIG)[2], CONFIG)) == 6
    rows = [{'split': f'repeat{s}', 'position': 'all_rows', **{k: .006 for k in m.e.KEYS}} for s in CONFIG['repeat_seeds']]
    assert m.acceptance(m.aggregate(rows, CONFIG), CONFIG)['accepted']


def test_real_oof_and_control070():
    if not (HERE / 'metrics.json').exists():
        pytest.skip('Real OOF not yet run')
    metrics = json.loads((HERE / 'metrics.json').read_text())
    cache = np.load(HERE / 'cache/bicoherence.npz', allow_pickle=False)
    _, ref, fixed = m.e.load_data(CONFIG)
    assert cache['X_train'].shape == (2760, 150)
    m.assert_alignment(ref, cache)
    deltas = []
    for name, folds in m.e.splits_for(ref, fixed, CONFIG):
        frame = pd.read_csv(HERE / 'predictions' / f'{name}_oof.csv')
        old = pd.read_csv(m.OLD / 'predictions' / f'{name}_oof.csv')
        m.assert_alignment(ref, frame)
        np.testing.assert_array_equal(frame.fold, folds)
        m.base.assert_subject_isolation(frame.subject.to_numpy(), folds)
        np.testing.assert_allclose(frame.p_control, old.p_train8, atol=1e-12, rtol=0)
        assert np.isfinite(frame[['p_control', 'p_plus']]).all().all()
        for p in ['all_rows'] + [str(i) for i in CONFIG['positions']]:
            q = frame if p == 'all_rows' else frame[frame.position == int(p)]
            vals = {arm: m.e.scalars(m.summarize(q, col)) for arm, col in (('control', 'p_control'), ('plus', 'p_plus'))}
            for arm in vals:
                assert vals[arm] == {k: pytest.approx(v, abs=1e-12) for k, v in m.e.scalars(metrics['runs'][name][p][arm]).items()}
            deltas.append({'split': name, 'position': p, **{k: vals['plus'][k] - vals['control'][k] for k in m.e.KEYS}})
    assert m.acceptance(m.aggregate(deltas, CONFIG), CONFIG) == metrics['acceptance']
    assert metrics['provenance']['model_fits'] == 60
    assert metrics['provenance']['public_arrays_loaded'] is False
