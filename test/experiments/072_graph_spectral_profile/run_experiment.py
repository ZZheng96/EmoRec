import importlib.util, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt
from threadpoolctl import threadpool_limits, threadpool_info

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('exp070', HERE.parent / '070_train_context_match_new_splits' / 'run_experiment.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
base, OLD = e.base, e.OLD
KEYS = e.KEYS
ALIGN_KEYS = ('subject_ids','health_groups','original_trial_ids','y_train')

def pearson_matrix(x):
    z = x - x.mean(axis=1, keepdims=True)
    den = np.sqrt((z*z).sum(axis=1, keepdims=True)); r = (z @ z.T) / np.maximum(den @ den.T, 1e-12)
    np.fill_diagonal(r, 0.0); return np.nan_to_num(r)

def graph_profile(segments, config):
    x = np.asarray(segments, dtype=np.float64)
    n, ch, samples = x.shape
    assert (ch, samples) == (30, config['sampling_rate'] * config['window_seconds'])
    edges = np.zeros((ch, ch), dtype=float)
    idx = np.arange(ch - 1)
    edges[idx, idx + 1] = edges[idx + 1, idx] = 1.0
    out = np.empty((n, 24), dtype=float)
    filters = [butter(config['filter_order'], band, btype='bandpass', fs=config['sampling_rate'], output='sos') for band in config['bands']]
    for row_i in range(n):
        feat = []
        for sos in filters:
            w = sosfiltfilt(sos, x[row_i], axis=-1)
            corr = pearson_matrix(w)
            weighted = edges * np.abs(corr)
            wd = weighted.sum(1)
            wl = np.eye(ch) - weighted / np.sqrt(np.maximum(wd[:, None] * wd[None, :], 1e-12))
            vals = np.linalg.eigvalsh(wl)[1:5]
            p = np.abs(vals) / max(np.abs(vals).sum(), 1e-12)
            feat.extend(vals.tolist() + [-np.sum(p * np.log(p + 1e-12)) / np.log(4), float(np.abs(vals[:2]).sum() / max(np.abs(vals).sum(), 1e-12))])
        out[row_i] = feat
    assert np.isfinite(out).all()
    return out

def build_cache(config, path):
    from emo_rec.data import load_train_segments
    files=sorted((ROOT/'数据/训练集').glob('*/*.mat')); feats=[]; ys=[]; subs=[]; groups=[]; trials=[]
    for f in files:
        seg,y,t=load_train_segments(f); feats.append(graph_profile(seg,config)); ys.append(y); trials.append(t); subs.extend([f.stem]*len(seg)); groups.extend(['DEP' if f.parent.name=='抑郁症患者' else 'HC']*len(seg))
    path.parent.mkdir(exist_ok=True); np.savez_compressed(path,X_train=np.concatenate(feats),y_train=np.concatenate(ys),subject_ids=np.asarray(subs),health_groups=np.asarray(groups),original_trial_ids=np.concatenate(trials),feature_names=np.asarray([f'graph_{i}' for i in range(24)]),config_sha256=np.asarray(base.sha256(HERE/'config.json')))

def load_train_cache(path):
    keys=('X_train','subject_ids','health_groups','original_trial_ids','y_train','feature_names')
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in keys}

def load_data(config):
    s,t,pe=[load_train_cache((HERE/config[k]).resolve()) for k in ('spectral_cache','temporal_cache','pe_cache')]
    for a in (t,pe):
        for key in ALIGN_KEYS: np.testing.assert_array_equal(s[key],a[key])
    cache=HERE/'cache/graph_features.npz'
    if not cache.exists(): build_cache(config,cache)
    g=np.load(cache,allow_pickle=False); assert str(g['config_sha256'])==base.sha256(HERE/'config.json')
    for key in ALIGN_KEYS: np.testing.assert_array_equal(s[key],g[key])
    keep=s['subject_ids']!=config['excluded_subject']; mask=np.asarray([str(n).startswith('log_bandpower_') for n in s['feature_names']]); raw=np.column_stack([s['X_train'][keep][:,mask],t['X_train'][keep],pe['X_train'][keep],g['X_train'][keep]])
    ref=pd.DataFrame({'row_index':np.arange(keep.sum()),'subject':s['subject_ids'][keep],'group':s['health_groups'][keep],'trial':s['original_trial_ids'][keep],'y':s['y_train'][keep]}); fixed=pd.read_csv(HERE/config['fixed_reference']); ref['position']=ref.groupby(['subject','y','trial'],sort=False).cumcount().to_numpy(); np.testing.assert_array_equal(ref.subject,fixed.subject_id); assert raw.shape==(2760,744)
    return raw,ref,fixed.fold.to_numpy()

def normalize(raw,subjects):
    from emo_rec.normalization import normalize_by_subject
    blocks=[normalize_by_subject(raw[:,a:b],subjects,'zscore') for a,b in ((0,150),(150,600),(600,720),(720,744))]
    return np.column_stack(blocks)
def contexts(raw,subjects,pos):
    a8=np.empty_like(raw)
    for i in range(5):
        ix=pos==i; a8[ix]=normalize(raw[ix],subjects[ix])
    return a8,a8.copy()
def summarize(frame, column):
    rr=frame.rename(columns={'subject':'subject_id','group':'health_group'}); return base.summarize(frame.y.to_numpy(),frame[column].to_numpy(),rr,frame.fold.to_numpy())
def scalars(r): return {**{k:r['overall'][k] for k in KEYS[:3]},'subject_auc_mean':r['subject_auc_mean'],'hc_auc':r['health_groups']['HC']['pooled']['auc'],'dep_auc':r['health_groups']['DEP']['pooled']['auc'],'worst_subject_auc':min(v['auc'] for v in r['subjects'])}
def job(a,y,name,folds,fold,config):
    st=time.perf_counter(); tr=folds!=fold; va=folds==fold
    control=base.make_model(config,config['outer_model_seed_base']+fold); control.fit(a[tr,:720],y[tr])
    plus=base.make_model(config,config['outer_model_seed_base']+fold); plus.fit(a[tr],y[tr])
    return name,fold,control.predict_proba(a[va,:720])[:,1],plus.predict_proba(a[va])[:,1],time.perf_counter()-st,int(tr.sum())
def aggregate(runs,config):
    out={}
    for subset in ['all_rows']+[str(i) for i in config['positions']]:
        vals=[{k:scalars(runs[f'repeat{s}'][subset])[k] for k in KEYS} for s in config['repeat_seeds']]; out[subset]={k:{m:float(getattr(np,m)([v[k] for v in vals],ddof=1)) if m=='std' else float(getattr(np,m)([v[k] for v in vals])) for m in ('mean','std')}|{'values':[v[k] for v in vals]} for k in KEYS};
    return out
def main():
    st=time.perf_counter(); config=json.loads((HERE/'config.json').read_text()); raw,ref,fixed=load_data(config); subjects=ref.subject.to_numpy(); y=ref.y.to_numpy(); a8,_=contexts(raw,subjects,ref.position.to_numpy()); splits=[('fixed002',fixed)]+[(f'repeat{s}',base.make_subject_folds(subjects,ref.group.to_numpy(),s)) for s in config['repeat_seeds']];
    with threadpool_limits(1): results=Parallel(n_jobs=config['n_jobs'],prefer='threads')(delayed(job)(a8,y,n,f,fold,config) for n,f in splits for fold in range(5))
    runs={}; rows=[]
    for name,folds in splits:
        frame=ref.assign(fold=folds,p_control=np.nan,p_plus=np.nan)
        for n,fold,p_control,p_plus,*_ in results:
            if n==name: frame.loc[folds==fold,'p_control']=p_control; frame.loc[folds==fold,'p_plus']=p_plus
        if name=='fixed002': np.testing.assert_allclose(frame.p_control,pd.read_csv(OLD/'predictions'/f'{name}_oof.csv').p_train8,atol=1e-12)
        frame.to_csv(HERE/'predictions'/f'{name}_oof.csv',index=False); runs[name]={}
        for sub in ['all_rows']+[str(i) for i in config['positions']]:
            q=frame if sub=='all_rows' else frame[frame.position==int(sub)]; r_control=summarize(q,'p_control'); r=summarize(q,'p_plus'); runs[name][sub]=r; rows.append({'split':name,'position':sub,**scalars(r)})
    agg=aggregate(runs,config); d={k:agg['all_rows'][k] for k in KEYS}; decision={'accepted':False,'checks':{'frozen_result_pending':True}}
    prov={'inputs':{str((HERE/config[k]).resolve()):base.sha256((HERE/config[k]).resolve()) for k in ('spectral_cache','temporal_cache','pe_cache','fixed_reference')},'sources':{str(HERE/n):base.sha256(HERE/n) for n in ('run_experiment.py','test_protocol.txt','config.json')},'public_arrays_loaded':False,'raw_shape':list(raw.shape),'cpu_only':'HGB has no GPU implementation','elapsed_seconds':time.perf_counter()-st,'model_fits':60,'active_threadpools':threadpool_info()}
    metrics={'runs':runs,'aggregate':agg,'acceptance':decision,'fit_seconds':[{'split':r[0],'fold':r[1],'seconds':r[4],'n_train':r[5]} for r in results],'elapsed_seconds':prov['elapsed_seconds'],'provenance':prov}; (HERE/'metrics.json').write_text(json.dumps(metrics,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else str(o))); (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,indent=2)); pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False); pd.DataFrame([{'split':n,**{k:scalars(runs[n]['all_rows'])[k] for k in KEYS}} for n,_ in splits]).to_csv(HERE/'paired_delta.csv',index=False); (HERE/'summary.md').write_text(f'# 实验072图谱特征\n\n耗时：{prov["elapsed_seconds"]:.2f}秒；744维；控制fixed002逐行复现070 p_train8；未访问公开数组。\n\n接受判断：未通过（本实验实现固定处理，结果见metrics.json）。\n',encoding='utf8')
if __name__=='__main__': main()
