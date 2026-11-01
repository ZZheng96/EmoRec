import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
try:
    import torch
except ImportError:
    torch = None

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exp076',HERE/'run_experiment.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

@pytest.mark.skipif(torch is None, reason='torch is required for rank-MI implementation')
def test_rank_mi_numpy_formula_and_independence():
    rng=np.random.default_rng(76); x=rng.normal(size=(2,30,2500)); actual=m.rank_mi(x,'cuda' if torch.cuda.is_available() else 'cpu')
    expected=[]
    for i,j in m.PAIRS:
        a,b=x[:,i],x[:,j]; qa=np.quantile(a,[.25,.5,.75],axis=1).T; qb=np.quantile(b,[.25,.5,.75],axis=1).T
        ca=(a[:,:,None]>qa[:,None,:]).sum(-1); cb=(b[:,:,None]>qb[:,None,:]).sum(-1); vals=[]
        for n in range(2):
            joint=np.zeros((4,4))
            np.add.at(joint,(ca[n],cb[n]),1); p=joint/joint.sum(); px=p.sum(1,keepdims=True); py=p.sum(0,keepdims=True); vals.append(np.where(p>0,p*np.log(np.divide(p,px*py,out=np.ones_like(p),where=px*py>0)),0).sum())
        expected.append(vals)
    np.testing.assert_allclose(actual,np.asarray(expected).T,atol=1e-12)
    x[1]*=100; np.testing.assert_allclose(actual,m.rank_mi(x,'cpu'),atol=1e-12)

@pytest.mark.skipif(torch is None, reason='torch is required for rank-MI implementation')
def test_constant_shape_and_window_independence():
    np.testing.assert_array_equal(m.rank_mi(np.ones((2,30,2500)), 'cpu'),0)
    with pytest.raises(ValueError): m.rank_mi(np.zeros((30,2500)),'cpu')
    with pytest.raises(ValueError): m.rank_mi(np.zeros((1,29,2500)),'cpu')
    x=np.zeros((2,30,2500)); x[1,3]=np.arange(2500); x[1,5]=np.arange(2500); out=m.rank_mi(x,'cpu'); x[0,0]=1; np.testing.assert_array_equal(out[1],m.rank_mi(x,'cpu')[1])

def test_frozen_protocol_and_metadata_independence():
    old=json.loads((m.OLD/'config.json').read_text())
    for key in ('hgb','repeat_seeds','positions','outer_model_seed_base','threshold','expected_model_fits'): assert m.CONFIG[key]==old[key]
    assert m.CONFIG['n_bins']==4 and len(m.PAIRS)==3
    raw,ref,fixed=m.e.load_data(m.CONFIG); m.assert_alignment(ref,ref.copy())
    f=np.random.default_rng(1).normal(size=(len(ref),3)); norm=m.normalize_extra(f,ref.subject.to_numpy(),ref.position.to_numpy()); other=ref.assign(y=1-ref.y,trial=999); np.testing.assert_array_equal(norm,m.normalize_extra(f,other.subject.to_numpy(),other.position.to_numpy()))
    assert len(m.e.splits_for(ref,fixed,m.CONFIG))==6

def test_real_oof_and_artifacts():
    if not (HERE/'metrics.json').exists(): pytest.skip('Real OOF not yet run')
    metrics=json.loads((HERE/'metrics.json').read_text()); cache=np.load(HERE/'cache/rank_mi.npz',allow_pickle=False); _,ref,fixed=m.e.load_data(m.CONFIG)
    assert cache['X_train'].shape==(2760,3); m.assert_alignment(ref,cache)
    for name,folds in m.e.splits_for(ref,fixed,m.CONFIG):
        frame=pd.read_csv(HERE/'predictions'/f'{name}_oof.csv'); old=pd.read_csv(m.OLD/'predictions'/f'{name}_oof.csv'); m.assert_alignment(ref,frame); np.testing.assert_array_equal(frame.fold,folds); np.testing.assert_allclose(frame.p_control,old.p_train8,atol=1e-12)
    assert metrics['provenance']['model_fits']==60 and metrics['provenance']['public_arrays_loaded'] is False
