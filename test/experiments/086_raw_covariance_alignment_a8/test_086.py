import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp086', HERE / 'run_experiment.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
C = json.loads((HERE / 'config.json').read_text(encoding='utf8'))


def test_protocol():
    e.validate_config(C)
    assert C['expected_model_fits'] == 6*5
    tree = ast.parse((HERE/'run_experiment.py').read_text(encoding='utf8'))
    fit = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_one')
    assert not any(isinstance(n,ast.With) for n in ast.walk(fit))
    for name in ('align_contexts','whiten_gpu'):
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert not {'y','group'} & {n.id for n in ast.walk(fn) if isinstance(n,ast.Name)}


def test_gpu_identity_cpu_equivalence_and_permutation():
    rng=np.random.default_rng(42); x=rng.normal(size=(8,5,70))*np.arange(1,6)[None,:,None]+3
    a,c,w=e.whiten_gpu(x); b,cc,ww=e.whiten_cpu(x)
    np.testing.assert_allclose(a,b,atol=1e-10)
    np.testing.assert_allclose(c,cc,atol=1e-10)
    np.testing.assert_allclose(w,ww,atol=1e-10)
    np.testing.assert_allclose(w@c@w,np.eye(5),atol=1e-10)
    p=rng.permutation(8); q=rng.permutation(5)
    aa,_,_=e.whiten_gpu(x[p][:,q])
    np.testing.assert_allclose(aa,a[p][:,q],atol=1e-10)
    subject=np.repeat(['a','b'],8); positions=np.zeros(16,int); xx=np.concatenate([x,x*5])
    aa,_=e.align_contexts(xx,subject,positions)
    xx[8:]*=4; bb,_=e.align_contexts(xx,subject,positions)
    np.testing.assert_array_equal(aa[:8],bb[:8])


def test_a8_actually_passes_to_fit(monkeypatch):
    rng=np.random.default_rng(8)
    raw=rng.normal(size=(80,720))*2+4
    subjects=np.repeat(['a','b'],40); pos=np.tile(np.repeat(np.arange(5),8),2)
    a8=e.contexts(raw,subjects,pos)[1]
    assert not np.allclose(raw,a8)
    for s in ('a','b'):
        for p in range(5):
            block=a8[(subjects==s)&(pos==p)]
            np.testing.assert_allclose(block.mean(0),0,atol=1e-12)
            np.testing.assert_allclose(block.std(0),1,atol=1e-12)
    class Model:
        def fit(self,x,y): np.testing.assert_array_equal(x,a8[40:])
        def predict_proba(self,x):
            np.testing.assert_array_equal(x,a8[:40]); return np.tile([.5,.5],(len(x),1))
    monkeypatch.setattr(e.base,'make_model',lambda *args:Model())
    e.fit_one(a8,np.zeros(80),'synthetic',np.repeat([0,1],40),0,C)
    source=(HERE/'run_experiment.py').read_text(encoding='utf8')
    assert 'a8 = contexts(raw, ref.subject.to_numpy(), ref.position.to_numpy())[1]' in source
    assert 'delayed(fit_one)(a8,' in source


def test_control_and_raw_alignment_negative():
    ctl=pd.read_csv(e.OLD/'predictions/repeat2042_oof.csv'); ref=ctl.drop(columns=['fold'])
    e.validate_control(ref,ctl.fold.to_numpy(),ctl)
    for col in e.META:
        wrong=ctl.copy(); wrong.loc[0,col]=wrong.loc[1,col] if wrong.loc[1,col]!=wrong.loc[0,col] else ('bad' if col in ('subject','group') else -99)
        with pytest.raises(AssertionError): e.validate_control(ref,ctl.fold.to_numpy(),wrong)
    r=ref[ref.subject==ref.subject.iloc[0]]
    e.check_raw_metadata(r.y.to_numpy(),r.trial.to_numpy(),r)
    with pytest.raises(AssertionError): e.check_raw_metadata(1-r.y.to_numpy(),r.trial.to_numpy(),r)
    with pytest.raises(AssertionError): e.check_raw_metadata(r.y.to_numpy(),r.trial.to_numpy()[::-1],r)


def test_public_guard():
    with e.public_guard():
        with np.load((HERE/C['spectral_cache']).resolve(),allow_pickle=False) as cache:
            assert cache['X_train'].ndim==2
            for key in ('X_public','public_user_ids','public_trial_ids'):
                with pytest.raises(AssertionError): cache[key]
        with pytest.raises(AssertionError): np.load('public_forbidden.npz')
        with pytest.raises(AssertionError): e.data.load_public_test_segments(Path('x'))
        with pytest.raises(AssertionError): e.h5py.File('测试集/x.mat','r')


def test_joint_acceptance_and_fixed_exclusion():
    agg={'all_rows':{'delta':{k:{'mean':.01,'values':[.01]*5} for k in e.KEYS}}}
    agg['all_rows']['delta']['auc']['values']=[.1,.1,.1,.1,-.1]
    agg['all_rows']['delta']['f1']['values']=[-.1,-.1,-.1,-.1,.1]
    assert not e.acceptance(agg,C)['checks']['joint_nonnegative_splits']


def test_real_cache_metadata_and_splits():
    with e.public_guard(): raw,ref,fixed=e.e070.load_data(C)
    assert raw.shape==(2760,720) and 'HC43' not in set(ref.subject)
    assert (ref.groupby(['subject','position']).size()==8).all()
    for n,folds in e.e070.splits_for(ref,fixed,C):
        e.base.assert_subject_isolation(ref.subject.to_numpy(),folds)
        e.validate_control(ref,folds,pd.read_csv(e.OLD/'predictions'/f'{n}_oof.csv'))


def test_outputs_recompute_and_all_hashes():
    if not (HERE/'metrics.json').exists(): pytest.skip('完整运行后复算')
    m=json.loads((HERE/'metrics.json').read_text(encoding='utf8')); p=m['provenance']
    assert p['model_fits']==len(m['fit_seconds'])==30
    assert p['feature_checks']['a8_applied'] and p['feature_checks']['unchanged_temporal_pe_raw_and_a8']
    assert p['feature_checks']['contexts_compared']==345
    assert p['public_arrays_loaded'] is False
    assert all(t['num_threads']==1 for t in p['active_threadpools'])
    for section in ('inputs','sources'):
        for path,digest in p[section].items(): assert e.sha256(path)==digest
    assert sum(Path(path).suffix=='.mat' for path in p['inputs'])==69
    for k in ('spectral_cache','temporal_cache','pe_cache','fixed_reference'):
        assert str((HERE/C[k]).resolve()) in p['inputs']
    runs={}
    for name in ['fixed002']+[f'repeat{s}' for s in C['repeat_seeds']]:
        f=pd.read_csv(HERE/'predictions'/f'{name}_oof.csv'); ctl=pd.read_csv(e.OLD/'predictions'/f'{name}_oof.csv')
        e.validate_control(f,f.fold.to_numpy(),ctl)
        np.testing.assert_allclose(f.p_control8,ctl.p_train8,atol=1e-15)
        e.base.assert_subject_isolation(f.subject.to_numpy(),f.fold.to_numpy())
        assert np.isfinite(f[['p_control8','p_alignment8']]).all().all()
        runs[name]={}
        for subset in ['all_rows']+[str(i) for i in C['positions']]:
            q=f if subset=='all_rows' else f[f.position==int(subset)]
            r={}
            for arm,col in (('control8','p_control8'),('alignment8','p_alignment8')):
                r[arm]=e.summarize(q,col)
                np.testing.assert_allclose(list(e.scalars(r[arm]).values()),list(e.scalars(m['runs'][name][subset][arm]).values()),atol=1e-14)
            r['delta']={k:e.scalars(r['alignment8'])[k]-e.scalars(r['control8'])[k] for k in e.KEYS}
            runs[name][subset]=r
    agg=e.aggregate_runs(runs,C)
    for subset in agg:
        for arm in agg[subset]:
            for k in e.KEYS:
                np.testing.assert_allclose(agg[subset][arm][k]['values'],m['aggregate'][subset][arm][k]['values'],atol=1e-14)
    assert e.acceptance(agg,C)==m['acceptance']
    runs['fixed002']={}
    assert e.aggregate_runs(runs,C)==agg
