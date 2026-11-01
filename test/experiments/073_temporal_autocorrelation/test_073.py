import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp073', HERE/'run_experiment.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CONFIG = json.loads((HERE/'config.json').read_text())


def test_formula_constant_artificial_and_independence():
    np.testing.assert_array_equal(m.autocorrelation(np.ones((2,30,2500))),np.zeros((2,60)))
    rng = np.random.default_rng(73)
    x = rng.normal(size=(2,30,2500))
    z = x-x.mean(-1,keepdims=True)
    expected = np.concatenate([(z[...,:-lag]*z[...,lag:]).sum(-1)/np.sqrt((z[...,:-lag]**2).sum(-1)*(z[...,lag:]**2).sum(-1)) for lag in (250,500)],axis=1)
    actual = m.autocorrelation(x)
    np.testing.assert_allclose(actual,expected,atol=1e-14)
    np.testing.assert_allclose(actual[:1],m.autocorrelation(x[:1]),atol=1e-14)
    x[1] *= 100
    np.testing.assert_allclose(actual[:1],m.autocorrelation(x)[:1],atol=1e-14)
    t = np.arange(2500)
    periodic = np.tile(np.sin(2*np.pi*t/250),(1,30,1))
    np.testing.assert_allclose(m.autocorrelation(periodic),1,atol=1e-14)


@pytest.mark.parametrize('shape',[(1,30,2499),(1,30,5000),(1,29,2500)])
def test_length(shape):
    with pytest.raises(ValueError):
        m.autocorrelation(np.zeros(shape))


def test_alignment_and_unlabelled_normalization():
    raw, ref, fixed = m.e.load_data(CONFIG)
    m.assert_alignment(ref,ref.copy())
    with pytest.raises(AssertionError):
        m.assert_alignment(ref,ref.iloc[::-1])
    rng = np.random.default_rng(73)
    features = rng.normal(size=(len(ref),60))
    normalized = m.normalize_extra(features,ref.subject.to_numpy(),ref.position.to_numpy())
    shuffled_labels = ref.copy()
    shuffled_labels['y'] = 1-ref.y
    np.testing.assert_array_equal(normalized,m.normalize_extra(features,shuffled_labels.subject.to_numpy(),shuffled_labels.position.to_numpy()))
    for _,indices in ref.groupby(['subject','position']).groups.items():
        np.testing.assert_allclose(normalized[indices].mean(0),0,atol=1e-14)
        np.testing.assert_allclose(normalized[indices].std(0),1,atol=1e-12)
    _, control = m.e.contexts(raw,ref.subject.to_numpy(),ref.position.to_numpy())
    joined = np.column_stack([control,normalized])
    np.testing.assert_array_equal(joined[:,:720],control)
    assert len(m.e.splits_for(ref,fixed,CONFIG)) == 6


def test_acceptance_joint_and_fixed_exclusion():
    rows = []
    for s in CONFIG['repeat_seeds']:
        for p in ['all_rows']+[str(i) for i in CONFIG['positions']]:
            rows.append({'split':f'repeat{s}','position':p,**{k:.006 for k in m.e.KEYS}})
    rows.append({'split':'fixed002','position':'all_rows',**{k:-100 for k in m.e.KEYS}})
    agg = m.aggregate(rows,CONFIG)
    assert m.acceptance(agg,CONFIG)['accepted']
    agg['all_rows']['auc']['values'] = [-.001,.01,.01,.01,.01]
    agg['all_rows']['f1']['values'] = [.01,-.001,.01,.01,.01]
    assert not m.acceptance(agg,CONFIG)['accepted']


def test_real_oof_and_control070():
    if not (HERE/'metrics.json').exists():
        pytest.skip('Real OOF not yet run')
    metrics = json.loads((HERE/'metrics.json').read_text())
    cache = np.load(HERE/'cache/autocorrelation.npz',allow_pickle=False)
    raw, ref, fixed = m.e.load_data(CONFIG)
    m.assert_alignment(ref,cache)
    assert cache['X_train'].shape == (2760,60)
    assert str(cache['config_sha256']) == m.base.sha256(HERE/'config.json')
    deltas = []
    for name,folds in m.e.splits_for(ref,fixed,CONFIG):
        frame = pd.read_csv(HERE/'predictions'/f'{name}_oof.csv')
        old = pd.read_csv(m.OLD/'predictions'/f'{name}_oof.csv')
        m.assert_alignment(ref,frame)
        assert frame.row_index.nunique() == len(frame) == 2760
        np.testing.assert_array_equal(frame.fold,folds)
        m.base.assert_subject_isolation(frame.subject.to_numpy(),folds)
        np.testing.assert_allclose(frame.p_control,old.p_train8,atol=1e-12,rtol=0)
        assert np.isfinite(frame[['p_control','p_plus']]).all().all()
        for p in ['all_rows']+[str(i) for i in CONFIG['positions']]:
            q = frame if p=='all_rows' else frame[frame.position==int(p)]
            values = {arm:m.e.scalars(m.summarize(q,col)) for arm,col in (('control','p_control'),('plus','p_plus'))}
            for arm in values:
                for k,v in values[arm].items():
                    assert v == pytest.approx(m.e.scalars(metrics['runs'][name][p][arm])[k],abs=1e-12)
            deltas.append({'split':name,'position':p,**{k:values['plus'][k]-values['control'][k] for k in m.e.KEYS}})
    agg = m.aggregate(deltas,CONFIG)
    for p in agg:
        for k in m.e.KEYS:
            np.testing.assert_allclose(agg[p][k]['values'],metrics['aggregate_delta'][p][k]['values'],atol=1e-12)
    assert m.acceptance(agg,CONFIG) == metrics['acceptance']
    saved = pd.read_csv(HERE/'paired_delta.csv')
    for k in m.e.KEYS:
        np.testing.assert_allclose(saved[k],[r[k] for r in deltas],atol=1e-12)
    assert metrics['provenance']['model_fits'] == 60
    assert metrics['provenance']['public_arrays_loaded'] is False
