import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp078',HERE/'run_experiment.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(1)


def test_kernels_shape_and_constants():
    for f in (1,2,4,8,16,32):
        k = m.wavelet(f)
        assert k.dtype==torch.complex128 and len(k)%2==1
        assert abs(k.sum())<1e-14
        assert abs(k.abs().sum()-1)<1e-14
        assert len(k)==2*int(np.ceil(4*250/(np.pi*f)))+1
    assert len(m.FEATURE_NAMES)==len(set(m.FEATURE_NAMES))==720
    np.testing.assert_allclose(m.scattering(np.ones((1,30,2500)),DEVICE),0,atol=1e-14)
    with pytest.raises(ValueError): m.scattering(np.zeros((30,2500)),'cpu')
    with pytest.raises(ValueError): m.scattering(np.zeros((1,29,2500)),'cpu')


def test_reflect_padding_linear_convolution():
    x = torch.zeros((2,30,2500),dtype=torch.float64)
    x[0,0,0]=1; x[1,1,-1]=1
    for k in (m.wavelet(1),m.wavelet(32),torch.ones(125,dtype=torch.float64)/125):
        r=(len(k)-1)//2
        padded=F.pad(x.reshape(-1,1,2500),(r,r),mode='reflect')
        kr=k.to(torch.complex128).flip(0)
        direct=F.conv1d(padded,kr.real.reshape(1,1,-1))+1j*F.conv1d(padded,kr.imag.reshape(1,1,-1))
        np.testing.assert_allclose(m.convolve_reflect(x,k),direct.reshape(2,30,2500),atol=1e-14,rtol=1e-12)


def test_modulation_and_window_channel_independence():
    t=np.arange(2500)/250
    x=np.zeros((2,30,2500))
    x[0,0]=np.sin(2*np.pi*16*t)
    x[1,0]=(1+0.8*np.sin(2*np.pi*2*t))*np.sin(2*np.pi*16*t)
    a=m.scattering(x,DEVICE).reshape(2,30,12,2)
    index=m.PATHS.index((16,2))
    assert a[1,0,index,0]>a[0,0,index,0]*2
    np.testing.assert_allclose(a[:,1:],0,atol=1e-14)
    np.testing.assert_allclose(a[1].reshape(-1),m.scattering(x[1:],DEVICE)[0],atol=1e-12,rtol=1e-12)
    x[0]=np.random.default_rng(78).normal(size=(30,2500))
    np.testing.assert_allclose(a[1].reshape(-1),m.scattering(x,DEVICE)[1],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(3*a[1].reshape(-1),m.scattering(3*x[1:],DEVICE)[0],atol=1e-12,rtol=1e-12)


def test_normalization_without_labels():
    rng=np.random.default_rng(78)
    frame=pd.DataFrame({'subject':np.repeat(['A','B'],16),'position':np.tile(np.repeat([0,1],8),2),'y':rng.integers(0,2,32),'trial':np.arange(32)})
    raw=rng.normal(size=(32,6)); raw[:,-1]=7
    z=m.normalize_extra(raw,frame.subject.to_numpy(),frame.position.to_numpy())
    for _,q in frame.groupby(['subject','position']):
        expected=(raw[q.index]-raw[q.index].mean(0))/np.where(raw[q.index].std(0)>1e-12,raw[q.index].std(0),1)
        np.testing.assert_allclose(z[q.index],expected,atol=1e-14)
    frame.y=1-frame.y; frame.trial=999
    np.testing.assert_array_equal(z,m.normalize_extra(raw,frame.subject.to_numpy(),frame.position.to_numpy()))


def test_frozen_protocol_and_splits():
    for folder in ('070_train_context_match_new_splits','076_cross_channel_rank_mi','077_alpha_traveling_wave_balance'):
        old=json.loads((HERE.parent/folder/'config.json').read_text())
        for k in ('hgb','repeat_seeds','positions','outer_model_seed_base','threshold','excluded_subject','expected_model_fits'): assert m.CONFIG[k]==old[k]
    raw,ref,fixed=m.e.load_data(m.CONFIG)
    _,control=m.e.contexts(raw,ref.subject.to_numpy(),ref.position.to_numpy())
    assert control.shape==(2760,720) and 'HC43' not in set(ref.subject)
    for name,folds in m.e.splits_for(ref,fixed,m.CONFIG):
        m.base.assert_subject_isolation(ref.subject.to_numpy(),folds)
        for folder in ('070_train_context_match_new_splits','076_cross_channel_rank_mi','077_alpha_traveling_wave_balance'):
            old=pd.read_csv(HERE.parent/folder/'predictions'/f'{name}_oof.csv')
            m.assert_alignment(ref,old)
            np.testing.assert_array_equal(folds,old.fold)


def test_production_rejects_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA'): m.build_cache()


def test_real_artifacts_control_and_metrics():
    if not (HERE/'metrics.json').exists(): pytest.skip('Real OOF not yet run')
    metrics=json.loads((HERE/'metrics.json').read_text())
    _,ref,fixed=m.e.load_data(m.CONFIG)
    with np.load(HERE/'cache/scattering.npz',allow_pickle=False) as cache:
        assert cache['X_train'].shape==(2760,720)
        assert np.isfinite(cache['X_train']).all()
        m.assert_alignment(ref,cache)
        np.testing.assert_array_equal(cache['feature_names'],m.FEATURE_NAMES)
        assert str(cache['config_sha256'])==m.base.sha256(HERE/'config.json')
    deltas=[]
    for name,folds in m.e.splits_for(ref,fixed,m.CONFIG):
        frame=pd.read_csv(HERE/'predictions'/f'{name}_oof.csv')
        old=pd.read_csv(m.OLD/'predictions'/f'{name}_oof.csv')
        m.assert_alignment(ref,frame)
        np.testing.assert_array_equal(frame.fold,folds)
        np.testing.assert_allclose(frame.p_control,old.p_train8,atol=1e-12,rtol=0)
        assert frame[['p_control','p_plus']].ge(0).all().all() and frame[['p_control','p_plus']].le(1).all().all()
        sc={a:m.e.scalars(m.helper.summarize(frame,c)) for a,c in (('control','p_control'),('plus','p_plus'))}
        for a in sc:
            for k,v in sc[a].items(): assert abs(v-m.e.scalars(metrics['runs'][name]['all_rows'][a])[k])<1e-12
        deltas.append({'split':name,'position':'all_rows',**{k:sc['plus'][k]-sc['control'][k] for k in m.e.KEYS}})
    for k in m.e.KEYS:
        expected=np.mean([r[k] for r in deltas if r['split'].startswith('repeat')])
        assert abs(expected-metrics['aggregate_delta']['all_rows'][k]['mean'])<1e-12
    assert metrics['acceptance']==m.helper.acceptance(metrics['aggregate_delta'],m.CONFIG)
    prov=metrics['provenance']
    assert prov['model_fits']==60 and prov['feature_device']=='cuda' and prov['public_arrays_loaded'] is False
    assert prov['treatment_shape']==[2760,1440]
    for path,digest in prov['sources'].items(): assert m.base.sha256(Path(path))==digest
    assert prov['cache_sha256']==m.base.sha256(HERE/'cache/scattering.npz')
    assert len(pd.read_csv(HERE/'paired_delta.csv'))==36
    assert len(pd.read_csv(HERE/'summary.csv'))==72
