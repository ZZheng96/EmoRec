import hashlib, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd, pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exp066',HERE/'run_experiment.py'); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
CONFIG=json.loads((HERE/'config.json').read_text(encoding='utf8'))

def brute_matches(x,m,r,tau,n):
    t=np.stack([x[i*tau:i*tau+n] for i in range(m)],axis=1)
    return sum(np.max(np.abs(t[i]-t[j]))<=r for i in range(n) for j in range(i+1,n))

def independent_lz_reference(s):
    count=0; i=0; n=len(s)
    while i < n:
        longest=0
        for j in range(i):
            length=0
            while i+length < n and s[j+length] == s[i+length]:
                length += 1
            longest=max(longest,length)
        i += min(longest+1,n-i)
        count += 1
    return count

def test_sampen_direct_small_reference_and_definition():
    x=np.array([0.,0.,1.,2.,1.,1.,3.,1.,1.,3.,0.,1.,3.,1.,0.,3.,0.,3.,2.,1.])
    for tau in (1,2):
        max_order=3; n=len(x)-(max_order-1)*tau; r=.2*np.std(x)
        B=brute_matches(x,2,r,tau,n); A=brute_matches(x,3,r,tau,n)
        assert B and A
        expected=-np.log(A/B)
        assert exp.sample_entropy(x,2,.2,tau,2)==pytest.approx(expected)
    assert np.isfinite(exp.sample_entropy(np.arange(20.),2,.2,1,2))

def test_lz76_direct_string_references():
    assert exp.lz76_count('')==0
    for n in range(1,11):
        for value in range(1 << n):
            s=f'{value:0{n}b}'
            assert exp.lz76_count(s)==independent_lz_reference(s)
    assert exp.lz76_count('010101')==3
    assert exp.lz76_count('000000')==2

def test_scale_invariant_zero_finite_and_lzc_finite():
    x=np.random.default_rng(66).normal(size=2500)
    assert exp.sample_entropy(x,2,.2,1,64)==pytest.approx(exp.sample_entropy(17*x,2,.2,1,64))
    for z in (np.zeros(2500),np.ones(2500),x):
        assert np.isfinite(exp.sample_entropy(z)) and np.isfinite(exp.lzc(z))
        if np.all(z == 0): assert exp.sample_entropy(z) == 0.0

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

def _metric_scalars(result):
    return {**{k: result['overall'][k] for k in exp.KEYS[:3]},
            'subject_auc_mean': result['subject_auc_mean'],
            'hc_auc': result['health_groups']['HC']['pooled']['auc'],
            'dep_auc': result['health_groups']['DEP']['pooled']['auc'],
            'worst_subject_auc': min(v['auc'] for v in result['subjects'])}


def _assert_json_close(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], dict): _assert_json_close(actual[key], expected[key])
        elif isinstance(actual[key], list): np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
        elif isinstance(actual[key], (float, int)): assert actual[key] == pytest.approx(expected[key], abs=1e-12)
        else: assert actual[key] == expected[key]


def test_real_cache_rows_subject_fold_if_present():
    p=HERE/'cache/complexity_features_v2.npz'
    if not p.exists(): pytest.skip('real cache not built')
    with np.load(p,allow_pickle=False) as z:
        assert z['X_train'].shape[0]==2800 and z['X_train'].shape[1]==60 and np.isfinite(z['X_train']).all(); assert len(np.unique(z['subject_ids']))==70
        assert str(z['config_sha256']) == exp.base.sha256(HERE/'config.json')
        assert str(z['source_sha256']) == exp.base.sha256(HERE/'run_experiment.py')
        first=sorted((HERE.parents[2]/'数据/训练集').glob('*/*.mat'))[0]
        seg,_,_=__import__('emo_rec.data',fromlist=['load_train_segments']).load_train_segments(first)
        expected=np.array([independent_lz_reference(''.join((ch>=np.median(ch)).astype(int).astype(str)))*np.log2(len(ch))/len(ch) for ch in seg[0]])
        np.testing.assert_allclose(z['X_train'][0,30:], expected, rtol=0, atol=1e-12)
        assert np.count_nonzero(z['X_train'][:,:30] == 0) == 0
    m=HERE/'metrics.json'
    if not m.exists(): pytest.skip('full experiment not run')
    data=json.loads(m.read_text()); assert data['provenance']['public_metrics'] is False
    assert data['provenance']['config_sha256'] == exp.base.sha256(HERE/'config.json')
    assert data['provenance']['script_sha256'] == exp.base.sha256(HERE/'run_experiment.py')
    recomputed={method:{} for method in exp.METHODS}
    for method in exp.METHODS:
        for split in ['fixed002']+[f'repeat{s}' for s in CONFIG['repeat_seeds']]:
            f=pd.read_csv(HERE/'predictions'/f'{method}_{split}_oof.csv'); assert len(f)==2760 and f.subject_id.nunique()==69 and f.prob_positive.between(0,1).all(); exp.base.assert_subject_isolation(f.subject_id.to_numpy(),f.fold.to_numpy())
            assert set(f.y_true)=={0,1}
            assert f.groupby('subject_id').size().eq(40).all()
            assert f.groupby('subject_id').y_true.apply(lambda values: values.value_counts().to_dict()=={0:20,1:20}).all()
            folds=pd.read_csv(HERE/'input_metadata'/f'folds_{split}.csv')
            pd.testing.assert_frame_equal(f.drop(columns='prob_positive'),folds)
            y=f.y_true.to_numpy(); p=f.prob_positive.to_numpy()
            subjects=[roc_auc_score(g.y_true,g.prob_positive) for _,g in f.groupby('subject_id')]
            independent={'auc':roc_auc_score(y,p),'f1':f1_score(y,p>=CONFIG['threshold']),'balanced_accuracy':balanced_accuracy_score(y,p>=CONFIG['threshold']),'subject_auc_mean':np.mean(subjects),'worst_subject_auc':min(subjects),'hc_auc':roc_auc_score(f[f.health_group=='HC'].y_true,f[f.health_group=='HC'].prob_positive),'dep_auc':roc_auc_score(f[f.health_group=='DEP'].y_true,f[f.health_group=='DEP'].prob_positive)}
            _assert_json_close(independent,_metric_scalars(data['runs'][method][split]))
            recomputed[method][split]=data['runs'][method][split]
    for method in exp.METHODS:
        values=[_metric_scalars(recomputed[method][f'repeat{s}']) for s in CONFIG['repeat_seeds']]
        agg={k:{'mean':float(np.mean([v[k] for v in values])),'std':float(np.std([v[k] for v in values],ddof=1)),'values':[v[k] for v in values]} for k in exp.KEYS}
        _assert_json_close(agg,data['aggregate'][method])
    recomputed_delta={}
    for method in ('plus_complexity',):
        repeat_values=[_metric_scalars(recomputed[method][f'repeat{s}']) for s in CONFIG['repeat_seeds']]
        control_values=[_metric_scalars(recomputed['control_pe'][f'repeat{s}']) for s in CONFIG['repeat_seeds']]
        repeat_mean={k:float(np.mean([v[k] for v in repeat_values])-np.mean([v[k] for v in control_values])) for k in exp.KEYS}
        fixed_plus=_metric_scalars(recomputed[method]['fixed002']); fixed_control=_metric_scalars(recomputed['control_pe']['fixed002'])
        fixed={k:fixed_plus[k]-fixed_control[k] for k in exp.KEYS}
        recomputed_delta[method]={'repeat_mean':repeat_mean,'fixed002':fixed}
    _assert_json_close(recomputed_delta,data['deltas_vs_control'])
    for method, delta in recomputed_delta.items():
        assert exp.accepts(delta['repeat_mean'],delta['fixed002']['auc'],CONFIG) == data['decision_accept'][method]
    control_057=HERE.parent/'057_multiscale_permutation_entropy/predictions/plus_pe_'
    for split in ['fixed002']+[f'repeat{s}' for s in CONFIG['repeat_seeds']]:
        a=pd.read_csv(HERE/'predictions'/f'control_pe_{split}_oof.csv')
        b=pd.read_csv(Path(str(control_057)+split+'_oof.csv'))
        pd.testing.assert_frame_equal(a.drop(columns='prob_positive'),b.drop(columns='prob_positive'))
        np.testing.assert_allclose(a.prob_positive,b.prob_positive,rtol=0,atol=1e-12)
