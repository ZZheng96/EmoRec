import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd, pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exp066',HERE/'run_experiment.py'); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
CONFIG=json.loads((HERE/'config.json').read_text(encoding='utf8'))

def brute_matches(x,m,r):
    t=np.stack([x[i:i+len(x)-(m-1)] for i in range(m)],axis=1); return sum(np.max(np.abs(t[i]-t[j]))<=r for i in range(len(t)) for j in range(i+1,len(t)))

def test_sampen_direct_small_reference_and_definition():
    x=np.array([0.,0.,0.,1.,1.,1.]); r=.2*np.std(x)
    B=brute_matches(x,2,r); A=brute_matches(x,3,r); expected=0. if not B or not A else -np.log(A/B)
    assert exp.sample_entropy(x,2,.2,1,2)==pytest.approx(expected)
    assert np.isfinite(exp.sample_entropy(np.arange(20.),2,.2,1,2))

def test_lz76_direct_string_references():
    assert exp.lz76_count('')==0
    for s,c in [('0',1),('01',2),('010101',4),('000000',3),('001011011',4)]: assert exp.lz76_count(s)==c

def test_scale_invariant_zero_finite_and_lzc_finite():
    x=np.random.default_rng(66).normal(size=2500)
    assert exp.sample_entropy(x,2,.2,1,64)==pytest.approx(exp.sample_entropy(17*x,2,.2,1,64))
    for z in (np.zeros(2500),np.ones(2500),x): assert np.isfinite(exp.sample_entropy(z)) and np.isfinite(exp.lzc(z))

def test_batch_independent_shape_and_order():
    x=np.random.default_rng(1).normal(size=(3,30,2500)); a=exp.complexity_features(x,CONFIG)
    assert a.shape==(3,60) and np.isfinite(a).all(); np.testing.assert_array_equal(a[[2,0,1]],exp.complexity_features(x[[2,0,1]],CONFIG)); np.testing.assert_array_equal(a[:1],exp.complexity_features(x[:1],CONFIG)[:1])

def test_bad_shape_rejected():
    with pytest.raises(ValueError): exp.complexity_features(np.zeros((1,30,2499)),CONFIG)

def test_alignment_and_columns():
    a={k:np.arange(4) for k in exp.ALIGN_KEYS}; exp.assert_alignment(a,a)
    assert {k:len(v) for k,v in exp.column_sets().items()}=={'control_pe':720,'plus_complexity':780}
    for key in a:
        b={k:v.copy() for k,v in a.items()}; b[key]=b[key][::-1]
        with pytest.raises(RuntimeError): exp.assert_alignment(a,b)

def test_real_cache_rows_subject_fold_if_present():
    p=HERE/'cache/complexity_features.npz'
    if not p.exists(): pytest.skip('real cache not built')
    with np.load(p,allow_pickle=False) as z:
        assert z['X_train'].shape[0]==2800 and z['X_train'].shape[1]==60 and np.isfinite(z['X_train']).all(); assert len(np.unique(z['subject_ids']))==70
    m=HERE/'metrics.json'
    if not m.exists(): pytest.skip('full experiment not run')
    data=json.loads(m.read_text()); assert data['provenance']['public_metrics'] is False
    for method in exp.METHODS:
        for split in ['fixed002']+[f'repeat{s}' for s in CONFIG['repeat_seeds']]:
            f=pd.read_csv(HERE/'predictions'/f'{method}_{split}_oof.csv'); assert len(f)==2760 and f.subject_id.nunique()==69 and f.prob_positive.between(0,1).all(); exp.base.assert_subject_isolation(f.subject_id.to_numpy(),f.fold.to_numpy())
