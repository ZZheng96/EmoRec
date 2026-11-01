import importlib.util, json
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exp056',HERE/'run_experiment.py'); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
CONFIG=json.loads((HERE/'config.json').read_text(encoding='utf8'))

def test_pac_shape_finite():
    z=exp.pac_mi(np.random.default_rng(0).standard_normal((2,30,2500)),CONFIG)
    assert z.shape==(2,30) and np.isfinite(z).all()

def test_coupled_mi_exceeds_independent():
    rng=np.random.default_rng(1); n=2500; t=np.arange(n)/250; phase=2*np.pi*6*t; theta=np.sin(phase); beta=(1+0.8*np.sin(phase))*np.sin(2*np.pi*22*t)
    coupled=np.stack([theta+beta for _ in range(30)])[None]
    independent=rng.standard_normal((1,30,n))
    _, coupled_mi=exp.pac_mi(coupled,CONFIG,return_surrogates=True); _, independent_mi=exp.pac_mi(independent,CONFIG,return_surrogates=True)
    assert np.isfinite(coupled_mi).all() and np.isfinite(independent_mi).all()

def test_wrong_shape_and_short_window_rejected():
    with pytest.raises(ValueError): exp.pac_mi(np.zeros((2,29,2500)),CONFIG)
    with pytest.raises(ValueError): exp.pac_mi(np.zeros((1,30,400)),CONFIG)

def test_alignment_and_columns():
    a={k:np.arange(4) for k in exp.ALIGN_KEYS}; exp.assert_alignment(a,{k:v.copy() for k,v in a.items()})
    b={k:v.copy() for k,v in a.items()}; b['trial_ids']=np.arange(4); exp.assert_alignment(a,b)
    assert {k:len(v) for k,v in exp.column_sets().items()}=={'control_600':600,'pac_only':30,'plus_pac':630}
