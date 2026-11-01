import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp075', HERE / 'run_experiment.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CONFIG = m.CONFIG
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_numpy_formula_and_independence():
    x = np.random.default_rng(75).normal(size=(2, 30, 2500))
    actual = m.psi_edges(x, DEVICE).cpu().numpy()
    frames = np.lib.stride_tricks.sliding_window_view(x, 500, axis=-1)[..., ::250, :]
    frames = frames - frames.mean(-1, keepdims=True)
    z = np.fft.rfft(frames * np.hanning(500), axis=-1)
    for b, (low, high) in enumerate(CONFIG['bands']):
        zz = z[..., 2 * low:2 * high + 1]
        s = np.einsum('ncsf,ndsf->ncdf', zz, zz.conj()) / 9
        power = np.mean(abs(zz) ** 2, axis=-2)
        c = s / np.sqrt(power[:, :, None, :] * power[:, None, :, :])
        expected = np.imag(c[..., :-1].conj() * c[..., 1:]).sum(-1)
        np.testing.assert_allclose(actual[:, b], expected, atol=1e-12)
    np.testing.assert_allclose(actual[:1], m.psi_edges(x[:1], DEVICE).cpu(), atol=1e-12)
    x[1] *= 100
    np.testing.assert_allclose(actual, m.psi_edges(x, DEVICE).cpu(), atol=1e-12)
    np.testing.assert_allclose(actual, m.psi_edges(x, 'cpu').numpy(), atol=1e-12)


def test_constant_shape_and_summaries():
    for value in (0., 1.):
        out = m.phase_slope_index(np.full((2, 30, 2500), value), DEVICE)
        assert out.shape == (2, 24)
        np.testing.assert_array_equal(out, 0)
    x = np.random.default_rng(750).normal(size=(1, 30, 2500))
    edges = m.psi_edges(x, DEVICE).cpu().numpy()
    out = m.phase_slope_index(x, DEVICE).reshape(1, 4, 6)
    np.testing.assert_array_equal(out[..., 2], 0)
    np.testing.assert_allclose(out[..., 0], out[..., 1], atol=1e-12)
    np.testing.assert_allclose(out[..., 3], 2 * out[..., 0], atol=1e-12)
    np.testing.assert_allclose(out[..., 4], edges[:, :, m.SOURCE, :][:, :, :, m.TARGET].sum((-2, -1)), atol=1e-12)
    np.testing.assert_array_equal(out[..., 4], -out[..., 5])


@pytest.mark.parametrize('shape', [(1, 29, 2500), (1, 30, 2499), (30, 2500)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError):
        m.phase_slope_index(np.zeros(shape), DEVICE)


def test_delay_reversal_permutation_diagnostics():
    rng = np.random.default_rng(751)
    x = rng.normal(size=(1, 30, 2500))
    x[:, 1] = np.roll(x[:, 0], 4, axis=-1)
    edges = m.psi_edges(x, DEVICE).cpu().numpy()
    assert (edges[0, :, 0, 1] > 0).all()
    np.testing.assert_array_equal(edges, -edges.swapaxes(-1, -2))
    reverse = m.psi_edges(x[..., ::-1].copy(), DEVICE).cpu().numpy()
    np.testing.assert_allclose(reverse, -edges, atol=1e-12)
    permutation = rng.permutation(30)
    permuted = m.psi_edges(x[:, permutation], DEVICE).cpu().numpy()
    np.testing.assert_allclose(permuted, edges[:, :, permutation, :][:, :, :, permutation], atol=1e-12)
    np.testing.assert_allclose(m.phase_slope_index(x[:, permutation], DEVICE).reshape(1, 4, 6)[..., :4], m.phase_slope_index(x, DEVICE).reshape(1, 4, 6)[..., :4], atol=1e-12)


def test_frozen_protocol_and_normalization():
    old_config = json.loads((m.OLD / 'config.json').read_text())
    for key in ('hgb', 'repeat_seeds', 'positions', 'outer_model_seed_base', 'threshold', 'expected_model_fits'):
        assert CONFIG[key] == old_config[key]
    assert CONFIG['bands'] == [[4, 8], [8, 13], [13, 30], [30, 45]]
    assert CONFIG['nperseg'] == CONFIG['nfft'] == 500
    assert CONFIG['noverlap'] == 250 and CONFIG['frequency_bin_hz'] == .5
    assert len(set(m.SOURCE) & set(m.TARGET)) == 0
    raw, ref, fixed = m.e.load_data(CONFIG)
    m.assert_alignment(ref, ref.copy())
    with pytest.raises(AssertionError):
        m.assert_alignment(ref, ref.iloc[::-1])
    features = np.random.default_rng(75).normal(size=(len(ref), 24))
    features[:, 2::6] = 0
    norm = m.normalize_extra(features, ref.subject.to_numpy(), ref.position.to_numpy())
    assert np.isfinite(norm).all()
    for _, idx in ref.groupby(['subject', 'position']).groups.items():
        np.testing.assert_allclose(norm[idx].mean(0), 0, atol=1e-14)
    other = ref.assign(y=1-ref.y, trial=999)
    np.testing.assert_array_equal(norm, m.normalize_extra(features, other.subject.to_numpy(), other.position.to_numpy()))
    assert m.e.contexts(raw, ref.subject.to_numpy(), ref.position.to_numpy())[1].shape == (2760, 720)
    assert len(m.e.splits_for(ref, fixed, CONFIG)) == 6
    rows = [{'split': f'repeat{s}', 'position': p, **{k: .006 for k in m.e.KEYS}} for p in ['all_rows'] + [str(i) for i in CONFIG['positions']] for s in CONFIG['repeat_seeds']]
    assert m.acceptance(m.aggregate(rows, CONFIG), CONFIG)['accepted']
    rows[0]['auc'] = rows[1]['auc'] = -.001
    assert not m.acceptance(m.aggregate(rows, CONFIG), CONFIG)['accepted']


def test_real_oof_and_control070():
    if not (HERE / 'metrics.json').exists():
        pytest.skip('Real OOF not yet run')
    metrics = json.loads((HERE / 'metrics.json').read_text())
    cache = np.load(HERE / 'cache/psi.npz', allow_pickle=False)
    _, ref, fixed = m.e.load_data(CONFIG)
    assert cache['X_train'].shape == (2760, 24)
    assert str(cache['config_sha256']) == m.base.sha256(HERE / 'config.json')
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
    actual = pd.DataFrame(deltas)
    saved = pd.read_csv(HERE / 'paired_delta.csv', dtype={'position': str})
    np.testing.assert_allclose(actual[list(m.e.KEYS)].to_numpy(), saved[list(m.e.KEYS)].to_numpy(), atol=1e-12)
    assert m.acceptance(m.aggregate(deltas, CONFIG), CONFIG) == metrics['acceptance']
    assert metrics['provenance']['model_fits'] == 60
    assert metrics['provenance']['public_arrays_loaded'] is False
    for path, digest in metrics['provenance']['sources'].items():
        assert m.base.sha256(Path(path)) == digest
