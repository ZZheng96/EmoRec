import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits, threadpool_info

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DEP = HERE.parent / '070_train_context_match_new_splits'
spec = importlib.util.spec_from_file_location('exp070_dependency', DEP / 'run_experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
base = e.base
KEYS = e.KEYS
sys.path.insert(0, str(HERE.parents[2] / 'src'))
from emo_rec.normalization import normalize_by_subject
REGIONS = {
    'frontal': ['FP1','FP2','F7','F3','FZ','F4','F8'],
    'frontocentral': ['FT7','FC3','FCZ','FC4','FT8'],
    'central': ['T3','C3','CZ','C4','T4'],
    'parietal': ['TP7','CP3','CPZ','CP4','TP8','T5','P3','PZ','P4','T6'],
    'occipital': ['O1','OZ','O2'],
}
CHANNELS = ['FP1','FP2','F7','F3','FZ','F4','F8','FT7','FC3','FCZ','FC4','FT8','T3','C3','CZ','C4','T4','TP7','CP3','CPZ','CP4','TP8','T5','P3','PZ','P4','T6','O1','OZ','O2']
BANDS = ['delta','theta','alpha','beta','gamma']
FEATURE_NAMES = np.asarray([f'regional_log_bandpower_{r}_{b}' for r in REGIONS for b in BANDS])


def load_data(config):
    raw, ref, fixed = e.load_data(config)
    spectral = np.load((HERE / config['spectral_cache']).resolve(), allow_pickle=False)
    names = spectral['feature_names']
    mask = np.asarray([str(n).startswith('log_bandpower_') for n in names])
    assert mask.sum() == 150
    expected = np.asarray([f'log_bandpower_{ch}_{band}' for ch in CHANNELS for band in BANDS])
    np.testing.assert_array_equal(names[mask], expected)
    s = raw[:, :150].reshape(len(raw), 30, 5)
    pooled = np.column_stack([s[:, [CHANNELS.index(ch) for ch in chans], b].mean(axis=1) for chans in REGIONS.values() for b in range(5)])
    treatment = np.column_stack([pooled, raw[:, 150:]])
    assert treatment.shape == (2760, 595) and np.isfinite(treatment).all()
    np.savez_compressed(HERE / 'cache/regional_features.npz', X_train=pooled, feature_names=FEATURE_NAMES, source_feature_names=names[mask], raw_spectral_hash=np.asarray(base.sha256((HERE / config['spectral_cache']).resolve())))
    return raw, treatment, ref, fixed


def contexts(treatment, subjects, positions):
    a40 = np.column_stack([normalize_by_subject(treatment[:, a:b], subjects, 'zscore') for a,b in ((0,25),(25,475),(475,595))])
    a8 = np.empty_like(a40)
    for i in range(5):
        ix = positions == i
        a8[ix] = np.column_stack([normalize_by_subject(treatment[ix, a:b], subjects[ix], 'zscore') for a,b in ((0,25),(25,475),(475,595))])
    return a40, a8


def job(a40, a8, y, name, folds, fold, context, config):
    start = time.perf_counter(); train, valid = folds != fold, folds == fold
    x = a40 if context == 40 else a8
    model = base.make_model(config, config['outer_model_seed_base'] + fold)
    model.fit(x[train], y[train]); p = model.predict_proba(a8[valid])[:, 1]
    return name, fold, context, p, time.perf_counter() - start, int(train.sum())


def summarize(frame, context):
    rr = frame.rename(columns={'subject':'subject_id','group':'health_group'})
    return base.summarize(frame.y.to_numpy(), frame[f'p_treatment{context}'].to_numpy(), rr, frame.fold.to_numpy())


def scalars(r):
    return {**{k:r['overall'][k] for k in KEYS[:3]}, 'subject_auc_mean':r['subject_auc_mean'], 'hc_auc':r['health_groups']['HC']['pooled']['auc'], 'dep_auc':r['health_groups']['DEP']['pooled']['auc'], 'worst_subject_auc':min(v['auc'] for v in r['subjects'])}


def aggregate_runs(runs, config):
    out = {}
    for subset in ['all_rows'] + [str(i) for i in config['positions']]:
        out[subset] = {}
        for context in ('40','8','delta8_minus40'):
            vals=[]
            for seed in config['repeat_seeds']:
                r=runs[f'repeat{seed}'][subset]
                vals.append(scalars(r[context]) if context != 'delta8_minus40' else {k:scalars(r['8'])[k]-scalars(r['40'])[k] for k in KEYS})
            out[subset][context]={k:{'mean':float(np.mean([v[k] for v in vals])),'std':float(np.std([v[k] for v in vals],ddof=1)),'values':[v[k] for v in vals]} for k in KEYS}
    return out


def acceptance(aggregate, config):
    d=aggregate['all_rows']['delta8_minus40']; a=config['acceptance']
    checks={'mean_delta_auc':d['auc']['mean']>=a['mean_delta_auc_min'],'mean_delta_f1':d['f1']['mean']>=a['mean_delta_f1_min'],'auc_nonnegative':sum(v>=0 for v in d['auc']['values'])>=a['nonnegative_splits_min_each'],'f1_nonnegative':sum(v>=0 for v in d['f1']['values'])>=a['nonnegative_splits_min_each'],'hc_no_excess_drop':d['hc_auc']['mean']>=-a['health_group_mean_auc_drop_max'],'dep_no_excess_drop':d['dep_auc']['mean']>=-a['health_group_mean_auc_drop_max']}
    return {'accepted':all(checks.values()),'checks':checks}


def main():
    start=time.perf_counter(); config=json.loads((HERE/'config.json').read_text(encoding='utf8'))
    for folder in ('predictions','input_metadata'): (HERE/folder).mkdir(exist_ok=True)
    raw,treatment,ref,fixed=load_data(config); subjects=ref.subject.to_numpy(); y=ref.y.to_numpy(); control40,control8=e.contexts(raw,subjects,ref.position.to_numpy()); a40,a8=contexts(treatment,subjects,ref.position.to_numpy())
    splits=[('fixed002',fixed)]+[(f'repeat{s}',base.make_subject_folds(subjects,ref.group.to_numpy(),s)) for s in config['repeat_seeds']]
    for name,folds in splits: base.assert_subject_isolation(subjects,folds); ref.assign(fold=folds).to_csv(HERE/'input_metadata'/f'folds_{name}.csv',index=False)
    sources=[HERE/n for n in ('run_experiment.py','test_079.py','config.json','test_protocol.txt')]+[DEP/n for n in ('run_experiment.py','test_070.py','config.json','test_protocol.txt')]+[Path(base.__file__),Path(e.e.__file__),HERE.parents[2]/'src/emo_rec/normalization.py',HERE.parents[2]/'src/emo_rec/data.py',HERE.parents[2]/'pyproject.toml',HERE.parents[2]/'数据/数据集说明文档.md']
    inputs=[(HERE/config[k]).resolve() for k in ('spectral_cache','temporal_cache','pe_cache','fixed_reference')]
    prov={'inputs':{str(p):base.sha256(p) for p in inputs},'sources':{str(p):base.sha256(p) for p in sources},'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'raw_shape':list(raw.shape),'treatment_shape':list(treatment.shape),'public_arrays_loaded':False,'model_fits':60,'cpu_only':'sklearn HGB has no GPU implementation; controlled CPU exception','n_jobs':config['n_jobs'],'threads_per_job':1,'regional_mapping':REGIONS,'spectral_feature_names':names.tolist() if (names:=np.load((HERE/config['spectral_cache']).resolve(),allow_pickle=False)['feature_names']) is not None else []}
    (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8')
    with threadpool_limits(limits=1):
        prov['active_threadpools']=threadpool_info(); results=Parallel(n_jobs=config['n_jobs'],prefer='threads')(delayed(job)(a40,a8,y,name,folds,fold,context,config) for name,folds in splits for fold in range(5) for context in (40,8))
    assert len(results)==60; runs={}; rows=[]; deltas=[]; checks={}
    for name,folds in splits:
        frame=ref.assign(fold=folds,p_train40=np.nan,p_train8=np.nan,p_treatment40=np.nan,p_treatment8=np.nan)
        for n,fold,context,p,seconds,n_train in results:
            if n==name: frame.loc[folds==fold,f'p_treatment{context}']=p
        old=pd.read_csv(DEP/'predictions'/f'{name}_oof.csv');
        for col in ('row_index','position','subject','group','trial','y','fold'): np.testing.assert_array_equal(frame[col],old[col])
        frame[['p_train40','p_train8']]=old[['p_train40','p_train8']]
        checks[name]=float(np.max(np.abs(frame.p_train40-old.p_train40))); assert checks[name]<=1e-12
        frame.to_csv(HERE/'predictions'/f'{name}_oof.csv',index=False); runs[name]={}
        for subset in ['all_rows']+[str(i) for i in config['positions']]:
            q=frame if subset=='all_rows' else frame[frame.position==int(subset)]; runs[name][subset]={}
            for context in (40,8): runs[name][subset][str(context)]=summarize(q,context); rows.append({'split':name,'position':subset,'train_context':context,'validation_context':8,**scalars(runs[name][subset][str(context)])})
            deltas.append({'split':name,'position':subset,**{k:scalars(runs[name][subset]['8'])[k]-scalars(runs[name][subset]['40'])[k] for k in KEYS}})
    aggregate=aggregate_runs(runs,config); decision=acceptance(aggregate,config); prov.update(cache_sha256=base.sha256(HERE/'cache/regional_features.npz'),elapsed_seconds=time.perf_counter()-start)
    metrics={'runs':runs,'aggregate':aggregate,'acceptance':decision,'control_070_max_abs_diff':checks,'fit_seconds':[{'split':r[0],'fold':r[1],'train_context':r[2],'seconds':r[4],'n_train':r[5]} for r in results],'provenance':prov}
    (HERE/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8'); (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8'); pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False); pd.DataFrame(deltas).to_csv(HERE/'paired_delta.csv',index=False)
    d=aggregate['all_rows']['delta8_minus40']; text=f"实验079：接受={decision['accepted']}；平均ΔAUC={d['auc']['mean']:.9f}；平均ΔF1={d['f1']['mean']:.9f}；HC ΔAUC={d['hc_auc']['mean']:.9f}；DEP ΔAUC={d['dep_auc']['mean']:.9f}；控制最大差={max(checks.values()):.3g}；60 fits。"
    (HERE/'summary.txt').write_text(text,encoding='utf8'); print(text,flush=True)

if __name__=='__main__': main()
