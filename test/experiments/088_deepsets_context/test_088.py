import copy
import importlib.util
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('experiment088', HERE / 'run_experiment.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))


def test_protocol():
    assert CONFIG['repeat_seeds'] == list(range(2042, 2047))
    assert CONFIG['expected_stage_fits'] == {'A': 0, 'B': 30, 'C': 30}
    assert CONFIG['training']['epochs'] == 60
    assert CONFIG['acceptance_comparisons'] == ['C_minus_A', 'C_minus_B']
    assert not CONFIG['public_access']
    assert list(inspect.signature(m.fit_predict).parameters) == ['x_train', 'y_train', 'x_valid', 'config', 'context', 'seed']


def test_cuda_required():
    with patch.object(torch.cuda, 'is_available', return_value=False):
        with pytest.raises(RuntimeError, match='CPU fallback'):
            m.configure(1)


def test_public_guard():
    with m.e.public_guard():
        with pytest.raises(AssertionError, match='Forbidden public path'):
            np.load('public_test.npz')
        with pytest.raises(AssertionError, match='Forbidden public loader'):
            m.e.data.load_public_test_segments(None)
        with np.load(m.OLD / '../001_spectral_logpower/cache/features.npz', allow_pickle=False) as cache:
            with pytest.raises(AssertionError, match='Forbidden array'):
                cache['X_test']


def test_metadata_controls():
    cfg = json.loads((m.OLD / 'config.json').read_text(encoding='utf8'))
    with m.e.public_guard():
        raw, ref, fixed = m.e.load_data(cfg)
    historical = json.loads((m.OLD / 'metrics.json').read_text(encoding='utf8'))
    for name, folds in m.e.splits_for(ref, fixed, CONFIG):
        ix, sf = m.group_indices(ref, folds)
        assert ix.shape == (345, 8)
        for fold in range(5):
            assert not set(ref.subject.iloc[ix[sf == fold].ravel()]) & set(ref.subject.iloc[ix[sf != fold].ravel()])
        m.check_control(name, ref, folds, historical)
    altered = ref.copy()
    altered.loc[0, 'subject'] = 'wrong_subject'
    with pytest.raises(AssertionError):
        m.e.validate_metadata(altered.assign(fold=fixed), ref.assign(fold=fixed))
    broken = fixed.copy()
    broken[0] = (broken[0] + 1) % 5
    with pytest.raises(RuntimeError, match='subject leakage'):
        m.group_indices(ref, broken)


def test_gpu_equivariance_and_no_context():
    m.configure(2088)
    x = torch.randn(4, 8, 720, device='cuda')
    models = [m.WindowNet(720, CONFIG, flag).cuda() for flag in (False, True)]
    assert sum(p.numel() for p in models[0].parameters()) == sum(p.numel() for p in models[1].parameters())
    for model in models:
        p = m.predict(model, x)
        for _ in range(8):
            perm = torch.randperm(8, device='cuda')
            torch.testing.assert_close(p, m.predict(model, x[:, perm])[:, torch.argsort(perm)], atol=CONFIG['permutation_atol'], rtol=0)
    changed = x.clone()
    changed[:, 1:] *= 7
    torch.testing.assert_close(m.predict(models[0], x)[:, 0], m.predict(models[0], changed)[:, 0], rtol=0, atol=0)


def test_fit_independent_of_validation_labels():
    cfg = copy.deepcopy(CONFIG)
    cfg['training']['epochs'] = 2
    m.configure(15)
    x = torch.randn(6, 8, 720, device='cuda')
    y = torch.randint(0, 2, (6, 8), device='cuda').float()
    for context in (False, True):
        before, _ = m.fit_predict(x[:4], y[:4], x[4:], cfg, context, 99)
        y[4:] = 1 - y[4:]
        after, _ = m.fit_predict(x[:4], y[:4], x[4:], cfg, context, 99)
        np.testing.assert_array_equal(before, after)


def test_completed_artifacts():
    if not (HERE / 'metrics.json').exists():
        pytest.skip('Full experiment not run yet')
    metrics = json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))
    prov = json.loads((HERE / 'input_metadata/provenance.json').read_text(encoding='utf8'))
    assert metrics['stage_fits'] == prov['stage_fits'] == CONFIG['expected_stage_fits']
    assert len(metrics['diagnostics']) == 60
    assert prov['reused_control_fits'] == 30 and not prov['public_arrays_loaded']
    assert max(d['permutation_max_abs_diff'] for d in metrics['diagnostics']) <= CONFIG['permutation_atol']
    for section in ('inputs', 'sources', 'outputs'):
        for path, digest in prov[section].items():
            assert len(digest) == 64 and m.e.sha256(path) == digest
    for name in metrics['runs']:
        frame = pd.read_csv(HERE / 'predictions' / f'{name}_oof.csv')
        assert len(frame) == 2760
        assert np.isfinite(frame[['p_A', 'p_B', 'p_C']]).all().all()
        assert frame[['p_A', 'p_B', 'p_C']].ge(0).all().all() and frame[['p_A', 'p_B', 'p_C']].le(1).all().all()
        old = pd.read_csv(m.OLD / 'predictions' / f'{name}_oof.csv')
        m.e.validate_metadata(frame, old)
        np.testing.assert_array_equal(frame.p_A, old.p_train8)
