import importlib.util, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits, threadpool_info
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DEP = HERE.parent / '070_train_context_match_new_splits'
spec = importlib.util.spec_from_file_location('exp070_dependency', DEP / 'run_experiment.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
base, load_data, contexts = e.base, e.load_data, e.contexts
KEYS = e.KEYS

def validate_config(c):
    old=json.loads((DEP/'config.json').read_text(encoding='utf8'))
    for k in ('hgb','outer_model_seed_base','repeat_seeds','threshold','excluded_subject','positions','n_jobs','threads_per_job'):
        assert c[k]==old[k]
    assert c['expected_model_fits']==30 and c['train_contexts']==[8]

def splits_for(ref,fixed,c):
    return [('fixed002',fixed)]+[(f'repeat{s}',base.make_subject_folds(ref.subject.to_numpy(),ref.group.to_numpy(),s)) for s in c['repeat_seeds']]

def job(a8,y,name,folds,fold,c):
    t=time.perf_counter(); train,valid=folds!=fold,folds==fold
    model=base.make_model(c,c['outer_model_seed_base']+fold); model.fit(a8[train],y[train])
    return name,fold,model.predict_proba(a8[valid])[:,1],time.perf_counter()-t,int(train.sum())

def center(frame,col='p_control'):
    p=np.clip(frame[col].to_numpy(float),1e-12,1-1e-12)
    z=np.log(p/(1-p)); keys=pd.MultiIndex.from_arrays([frame.subject.to_numpy(),frame.position.to_numpy()]); means=pd.Series(z).groupby(keys).transform('mean').to_numpy()
    return 1/(1+np.exp(-(z-means)))

def summarize(frame,col):
    rr=frame.rename(columns={'subject':'subject_id','group':'health_group'})
    return base.summarize(frame.y.to_numpy(),frame[col].to_numpy(),rr,frame.fold.to_numpy())
def scalars(r): return e.scalars(r)

def aggregate(runs,c):
    out={}
    for subset in ['all_rows']+[str(i) for i in c['positions']]:
        out[subset]={}
        for arm in ('control','calibrated','delta'):
            vals=[]
            for s in c['repeat_seeds']:
                r=runs[f'repeat{s}'][subset]; vals.append(scalars(r[arm]) if arm!='delta' else {k:scalars(r['calibrated'])[k]-scalars(r['control'])[k] for k in KEYS})
            out[subset][arm]={k:{'mean':float(np.mean([v[k] for v in vals])),'std':float(np.std([v[k] for v in vals],ddof=1)),'values':[v[k] for v in vals]} for k in KEYS}
    return out
def acceptance(a,c):
    d=a['all_rows']['delta']; x=c['acceptance']; checks={'mean_delta_auc':d['auc']['mean']>=x['mean_delta_auc_min'],'mean_delta_f1':d['f1']['mean']>=x['mean_delta_f1_min'],'auc_nonnegative':sum(v>=0 for v in d['auc']['values'])>=x['nonnegative_splits_min_each'],'f1_nonnegative':sum(v>=0 for v in d['f1']['values'])>=x['nonnegative_splits_min_each'],'hc_no_excess_drop':d['hc_auc']['mean']>=-x['health_group_mean_auc_drop_max'],'dep_no_excess_drop':d['dep_auc']['mean']>=-x['health_group_mean_auc_drop_max']}
    return {'accepted':all(checks.values()),'checks':checks}

def main():
    start=time.perf_counter(); c=json.loads((HERE/'config.json').read_text(encoding='utf8')); validate_config(c)
    for d in ('predictions','input_metadata'): (HERE/d).mkdir(exist_ok=True)
    raw,ref,fixed=load_data(c); subjects,y=ref.subject.to_numpy(),ref.y.to_numpy(); _,a8=contexts(raw,subjects,ref.position.to_numpy()); splits=splits_for(ref,fixed,c)
    sources=[HERE/n for n in ('run_experiment.py','test_090.py','config.json','test_protocol.txt')]
    sources += [DEP/n for n in ('run_experiment.py','test_070.py','config.json','test_protocol.txt')]
    sources += [Path(base.__file__),DEP.parents[2]/'src/emo_rec/normalization.py',DEP.parents[2]/'src/emo_rec/data.py',DEP.parents[2]/'src/emo_rec/submission.py',DEP.parents[2]/'pyproject.toml',DEP.parents[2]/'数据/数据集说明文档.md']
    inputs=[(HERE/c[k]).resolve() for k in ('spectral_cache','temporal_cache','pe_cache','fixed_reference')]
    prov={'inputs':{str(p):base.sha256(p) for p in inputs},'sources':{str(p):base.sha256(p) for p in sources},'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'raw_shape':list(raw.shape),'public_arrays_loaded':False,'model_fits':30,'cpu_only':'sklearn HGB has no GPU implementation; frozen-model CPU exception','n_jobs':c['n_jobs'],'threads_per_job':c['threads_per_job'],'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    for name,folds in splits:
        base.assert_subject_isolation(subjects,folds); ref.assign(fold=folds).to_csv(HERE/'input_metadata'/f'folds_{name}.csv',index=False)
    (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8')
    with threadpool_limits(limits=1):
        prov['active_threadpools']=threadpool_info(); results=Parallel(n_jobs=c['n_jobs'],prefer='threads')(delayed(job)(a8,y,n,f,fold,c) for n,f in splits for fold in range(5))
    runs={}; rows=[]; deltas=[]; checks={}
    for name,folds in splits:
        frame=ref.assign(fold=folds,p_control=np.nan,p_calibrated=np.nan)
        for n,fold,p,*_ in results:
            if n==name: frame.loc[folds==fold,'p_control']=p
        frame['p_calibrated']=center(frame); assert np.isfinite(frame[['p_control','p_calibrated']]).all().all()
        if name=='fixed002':
            old=pd.read_csv(DEP/'predictions/fixed002_oof.csv'); checks['control_max_abs_diff']=float(np.max(np.abs(frame.p_control-old.p_train8))); assert checks['control_max_abs_diff']<=1e-12
        frame.to_csv(HERE/'predictions'/f'{name}_oof.csv',index=False); runs[name]={}
        for subset in ['all_rows']+[str(i) for i in c['positions']]:
            sel=frame if subset=='all_rows' else frame[frame.position==int(subset)]; runs[name][subset]={}
            for arm,col in [('control','p_control'),('calibrated','p_calibrated')]:
                r=summarize(sel,col); runs[name][subset][arm]=r; rows.append({'split':name,'position':subset,'arm':arm,'validation_context':8,**scalars(r)})
            deltas.append({'split':name,'position':subset,**{k:scalars(runs[name][subset]['calibrated'])[k]-scalars(runs[name][subset]['control'])[k] for k in KEYS}})
    agg=aggregate(runs,c); decision=acceptance(agg,c); prov['model_fits']=len(results)
    metrics={'runs':runs,'aggregate':agg,'acceptance':decision,'fixed002_max_abs_diff':checks,'elapsed_seconds':time.perf_counter()-start,'fit_seconds':[{'split':r[0],'fold':r[1],'seconds':r[3],'n_train':r[4]} for r in results],'provenance':prov}
    (HERE/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8'); (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8'); pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False); pd.DataFrame(deltas).to_csv(HERE/'paired_delta.csv',index=False)
    print(json.dumps({'acceptance':decision,'all_rows':agg['all_rows'],'elapsed_seconds':metrics['elapsed_seconds']},indent=2),flush=True)
if __name__=='__main__': main()
