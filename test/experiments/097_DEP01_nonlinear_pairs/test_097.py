import ast, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e097', HERE/'run_experiment.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)

@pytest.fixture(scope='module')
def data():
    x,r = e.load_subject(json.loads((HERE/'config.json').read_text()))
    return x,r,r.y.to_numpy(),e.base.trial_keys(r)

def test_fixed_protocol(data):
    x,r,y,k = data
    assert x.shape == (40,720) and set(r.subject) == {'DEP01'}
    assert len(e.CANDIDATES) == len(set(e.CANDIDATES)) == 32
    assert e.CANDIDATES == tuple((b,c,g) for b in ('all','spectral','temporal','PE') for c in (.1,1,10,100) for g in ('scale',.01))
    assert all((k==v).sum()==5 for v in set(k))

def check_split(y,k,tr,va,n):
    assert not set(k[tr]) & set(k[va])
    assert np.all(tr ^ va)
    assert np.array_equal(np.bincount(y[tr].astype(int)),[n//2,n//2])
    assert np.array_equal(np.bincount(y[va].astype(int)),[5,5])

def test_inner_outer_isolation_balance(data):
    x,r,y,k = data
    outer=list(e.splits(y,k)); assert len(outer)==16
    for tr,va in outer:
        check_split(y,k,tr,va,30)
        inner=list(e.splits(y[tr],k[tr])); assert len(inner)==9
        for it,iv in inner: check_split(y[tr],k[tr],it,iv,20)
        assert np.all(np.sum([iv for it,iv in inner],axis=0)==3)
    assert np.all(np.sum([va for tr,va in outer],axis=0)==4)

def test_differing_predictions_aggregate(data):
    x,r,y,k=data
    masks=[va for tr,va in e.splits(y,k)]
    values=[np.full(va.sum(),i/16) for i,va in enumerate(masks)]
    p,c=e.aggregate(values,masks,4)
    expected=np.array([np.mean([i/16 for i,m in enumerate(masks) if m[j]]) for j in range(40)])
    assert np.allclose(p,expected) and np.all(c==4)
    assert any(len(set(v[jj] for v,m in zip(values,masks) for jj,j in enumerate(np.flatnonzero(m)) if j==row))>1 for row in range(40))

def test_selector_only_training_labels_and_tie_order(monkeypatch):
    tree=ast.parse((HERE/'run_experiment.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='select_candidate']
    assert len(calls)==1
    assert [ast.unparse(a) for a in calls[0].args]==['x[tr]','y[tr]','keys[tr]']
    monkeypatch.setattr(e,'fit_predict',lambda xt,yt,xv,c:np.full(len(xv),.5))
    y=np.repeat([0,1],15); keys=np.repeat(np.arange(6),5)
    best,scores,p=e.select_candidate(np.zeros((30,720)),y,keys)
    assert best==e.CANDIDATES[0] and len(scores)==32

def test_scaler_training_only(monkeypatch):
    original=e.StandardScaler
    seen=[]
    class Spy(original):
        def fit(self,x,y=None,sample_weight=None):
            seen.append(x.copy())
            return super().fit(x,y,sample_weight=sample_weight)
    monkeypatch.setattr(e,'StandardScaler',Spy)
    x=np.arange(12).reshape(4,3).astype(float)
    p=e.fit_predict(x,np.array([0,0,1,1]),np.full((2,3),999.),e.CANDIDATES[0])
    assert len(seen)==1 and np.array_equal(seen[0],x) and np.isfinite(p).all()

def test_saved_artifacts_and_metric_recompute(data):
    if not (HERE/'metrics.json').exists(): pytest.skip('Run artifacts not yet generated')
    x,r,y,k=data
    d=pd.read_csv(HERE/'predictions/DEP01_nonlinear_pairs_oof.csv')
    q=pd.read_csv(HERE/'predictions/pair_predictions.csv')
    inner=pd.read_csv(HERE/'predictions/inner_mean_predictions.csv')
    m=json.loads((HERE/'metrics.json').read_text())
    selections=json.loads((HERE/'input_metadata/selections.json').read_text())
    assert len(q)==160 and len(d)==40 and d.row_index.is_unique
    assert 'fold' not in d and d.prediction_count.eq(4).all()
    avg=q.groupby('row_index').prob_positive.mean().reindex(d.row_index)
    assert np.allclose(d.prob_positive,avg,rtol=0,atol=1e-15)
    assert q.groupby('row_index').size().eq(4).all()
    assert (q.groupby('row_index').prob_positive.nunique()>1).any()
    for key,val in e.metric(d.y,d.prob_positive).items(): assert np.isclose(val,m['pooled_oof'][key])
    pair=[e.metric(g.y,g.prob_positive) for _,g in q.groupby('pair_id')]
    for key in ('auc','f1'):
        assert np.isclose(np.mean([v[key] for v in pair]),m['pair_mean'][key])
        assert np.isclose(e.metric(y,np.full(40,.5))[key],m['prior_0.5'][key])
        assert np.allclose([v[key] for v in pair],[v[key] for v in m['pair_metrics']])
    assert len(inner)==16*32*30 and inner.prediction_count.eq(3).all()
    for rec,(tr,va) in zip(selections,e.splits(y,k)):
        assert rec['train_indices']==np.flatnonzero(tr).tolist() and rec['valid_indices']==np.flatnonzero(va).tolist()
        idx=np.flatnonzero(tr)
        for split,(it,iv) in zip(rec['inner_splits'],e.splits(y[tr],k[tr])):
            assert split['train_indices']==idx[it].tolist() and split['valid_indices']==idx[iv].tolist()
            assert not set(split['train_indices']+split['valid_indices']) & set(np.flatnonzero(va))
        assert len(rec['inner_scores'])==32
        for score in rec['inner_scores']:
            z=inner[(inner.pair_id==rec['pair_id']) & (inner.candidate_index==score['candidate_index'])]
            assert set(z.local_index)==set(idx)
            for key,val in e.metric(z.y,z.prob_positive).items(): assert np.isclose(val,score[key])
        best=max(rec['inner_scores'],key=lambda s:(s['auc'],s['f1'],-s['candidate_index']))
        assert rec['selected_candidate']==best['candidate']
    prov=json.loads((HERE/'input_metadata/provenance.json').read_text())
    for group in ('sources','inputs'):
        for path,digest in prov[group].items(): assert e.sha(Path(path))==digest
