import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
try: import torch
except ImportError: torch=None
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exp077',HERE/'run_experiment.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

@pytest.mark.skipif(torch is None,reason='torch required')
def test_formula_direction_and_shape():
    rng=np.random.default_rng(77); x=rng.normal(size=(2,30,2500)); out=m.alpha_travel_balance(x,'cuda' if torch.cuda.is_available() else 'cpu'); assert out.shape==(2,2); assert np.isfinite(out).all()
    assert np.max(np.abs(out))<=1
    with pytest.raises(ValueError): m.alpha_travel_balance(np.zeros((30,2500)),'cpu')
    with pytest.raises(ValueError): m.alpha_travel_balance(np.zeros((1,29,2500)),'cpu')

@pytest.mark.skipif(torch is None,reason='torch required')
def test_constant_and_independent_windows():
    x=np.ones((2,30,2500)); np.testing.assert_allclose(m.alpha_travel_balance(x,'cpu'),0)
    x=np.zeros((2,30,2500)); t=np.arange(2500)/250.;
    for k,ch in enumerate([0,3,13,23,27]): x[1,ch]=np.sin(2*np.pi*10*t+2*np.pi*k/5)
    x[0,0]=1; out=m.alpha_travel_balance(x,'cpu'); np.testing.assert_allclose(out[1],m.alpha_travel_balance(x,'cpu')[1])

def test_normalization_and_no_labels():
    rng=np.random.default_rng(1); raw=rng.normal(size=(10,2)); subjects=np.array(['A']*5+['B']*5); pos=np.array([0]*5+[1]*5); z=m.normalize_extra(raw,subjects,pos); np.testing.assert_allclose(z[:5].mean(0),0,atol=1e-12); np.testing.assert_allclose(z[:5].std(0),1,atol=1e-12)
    other=m.e.load_data(m.CONFIG)[1].assign(y=1-m.e.load_data(m.CONFIG)[1].y,trial=999)
    f=rng.normal(size=(len(other),2)); np.testing.assert_array_equal(m.normalize_extra(f,other.subject.to_numpy(),other.position.to_numpy()),m.normalize_extra(f,other.subject.to_numpy(),other.position.to_numpy()))

def test_frozen_protocol_and_control():
    old=json.loads((m.OLD/'config.json').read_text());
    for key in ('hgb','repeat_seeds','positions','outer_model_seed_base','threshold','expected_model_fits'): assert m.CONFIG[key]==old[key]
    assert len(m.CHAINS)==2 and m.CONFIG['fragment_samples']==125 and m.CONFIG['step_samples']==62
    raw,ref,fixed=m.e.load_data(m.CONFIG); assert len(m.e.splits_for(ref,fixed,m.CONFIG))==6

def test_real_oof_and_artifacts():
    if not (HERE/'metrics.json').exists(): pytest.skip('Real OOF not yet run')
    metrics=json.loads((HERE/'metrics.json').read_text()); cache=np.load(HERE/'cache/alpha_traveling_wave.npz',allow_pickle=False); _,ref,fixed=m.e.load_data(m.CONFIG)
    assert cache['X_train'].shape==(2760,2); m.assert_alignment(ref,cache)
    for name,folds in m.e.splits_for(ref,fixed,m.CONFIG):
        frame=pd.read_csv(HERE/'predictions'/f'{name}_oof.csv'); old=pd.read_csv(m.OLD/'predictions'/f'{name}_oof.csv'); m.assert_alignment(ref,frame); np.testing.assert_array_equal(frame.fold,folds); np.testing.assert_allclose(frame.p_control,old.p_train8,atol=1e-12)
    assert metrics['provenance']['model_fits']==60 and metrics['provenance']['public_arrays_loaded'] is False
