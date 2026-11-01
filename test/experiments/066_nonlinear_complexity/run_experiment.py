import importlib.util, json, math, platform, sys, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; sys.dont_write_bytecode=True; sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('base032',HERE.parent/'032_repeated_subject_folds'/'run_experiment.py'); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
pe_spec=importlib.util.spec_from_file_location('exp057',HERE.parent/'057_multiscale_permutation_entropy'/'run_experiment.py'); pe_base=importlib.util.module_from_spec(pe_spec); pe_spec.loader.exec_module(pe_base)
METHODS=('control_pe','plus_complexity'); KEYS=('auc','f1','balanced_accuracy','subject_auc_mean','hc_auc','dep_auc','worst_subject_auc'); ALIGN_KEYS=('subject_ids','health_groups','original_trial_ids','y_train','public_user_ids','public_trial_ids')

def _pair_matches(templates, r, chunk_rows):
    n = len(templates)
    if n < 2:
        return 0
    tree = cKDTree(templates, leafsize=chunk_rows)
    # count_neighbors includes n self matches and both (i,j)/(j,i).
    total = int(tree.count_neighbors(tree, r, p=np.inf))
    return (total - n) // 2

def sample_entropy(x,m=2,r_factor=.2,tau=1,chunk_rows=64):
    x=np.asarray(x,dtype=float); std=float(np.std(x))
    if std==0 or not np.isfinite(std): return 0.0
    r=r_factor*std
    n=x.size-m*tau
    if n<1: return 0.0
    out=[]
    for order in (m,m+1):
        t=np.stack([x[i*tau:i*tau+n] for i in range(order)],axis=1)
        out.append(_pair_matches(t,r,chunk_rows))
    B,A=out
    if B==0 or A==0: return 0.0
    v=-math.log(A/B); return float(v) if np.isfinite(v) else 0.0

def lz76_count(bits):
    if isinstance(bits, str):
        s = bits
    else:
        s = ''.join('1' if int(v) else '0' for v in np.asarray(bits).ravel())
    n = len(s)
    if n == 0:
        return 0
    i = 0
    c = 0
    while i < n:
        length = 1
        # Exhaustive LZ76 parsing: the next phrase is the shortest substring
        # absent from the already parsed prefix, with overlap allowed.
        while i + length <= n:
            candidate = s[i:i + length]
            if candidate not in s[:i + length - 1]:
                break
            length += 1
        c += 1
        i += length
    return c

def lzc(x):
    x=np.asarray(x,dtype=float); n=x.size
    if n==0: return 0.0
    c=lz76_count(x>=np.median(x)); v=c*math.log2(n)/n
    return float(v) if np.isfinite(v) else 0.0

def complexity_features(segments,config):
    x=np.asarray(segments,dtype=float)
    if x.ndim!=3 or x.shape[1:]!=(30,2500): raise ValueError('expected 10-second windows (n, 30, 2500)')
    if not np.isfinite(x).all(): raise ValueError('nonfinite EEG')
    cc=config['complexity']; out=np.empty((len(x),60),dtype=float)
    for i in range(len(x)):
        for ch in range(30):
            out[i,ch]=sample_entropy(x[i,ch],cc['sampen_m'],cc['sampen_r_std'],cc['sampen_tau'],cc['chunk_rows'])
            out[i,30+ch]=lzc(x[i,ch])
    return out

def build_cache(config,path):
    from emo_rec.data import load_train_segments
    files=sorted((ROOT/'数据/训练集').glob('*/*.mat'))
    tr=[]; ys=[]; subs=[]; groups=[]; trials=[]
    for i,f in enumerate(files,1):
        seg,y,t=load_train_segments(f); tr.append(complexity_features(seg,config)); ys.append(y); trials.append(t); subs.extend([f.stem]*len(seg)); groups.extend(['DEP' if f.parent.name=='抑郁症患者' else 'HC']*len(seg)); print(f'complexity {i}/{len(files)} {f.stem}',flush=True)
    names=np.asarray([f'sampen_{i}' for i in range(30)]+[f'lzc_{i}' for i in range(30)])
    path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,X_train=np.concatenate(tr),y_train=np.concatenate(ys),subject_ids=np.asarray(subs),health_groups=np.asarray(groups),original_trial_ids=np.concatenate(trials),feature_names=names,config_sha256=np.asarray(base.sha256(HERE/'config.json')),source_sha256=np.asarray(base.sha256(Path(__file__))))
    meta=[{'path':str(f.relative_to(ROOT)),'bytes':f.stat().st_size,'sha256':base.sha256(f)} for f in files]; (HERE/'input_metadata/raw_inputs.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')

def assert_alignment(a,b):
    for k in ALIGN_KEYS:
        if not np.array_equal(a[k],b[k]): raise RuntimeError(f'cache alignment mismatch: {k}')

def load_data(config):
    from emo_rec.normalization import normalize_by_subject
    s=base.load_cache((HERE/config['spectral_cache']).resolve()); t=base.load_cache((HERE/config['temporal_cache']).resolve()); assert_alignment(s,t)
    pepath=(HERE.parent/'057_multiscale_permutation_entropy'/'cache/pe_features.npz');
    if not pepath.exists(): pe_base.build_cache(config,pepath)
    p=base.load_cache(pepath); assert_alignment(s,p)
    cpath=HERE/config['complexity_cache'];
    if not cpath.exists(): build_cache(config,cpath)
    c=base.load_cache(cpath)
    for key in ('subject_ids','health_groups','original_trial_ids','y_train'):
        if not np.array_equal(s[key],c[key]): raise RuntimeError(f'cache alignment mismatch: {key}')
    if str(c['config_sha256']) != base.sha256(HERE/'config.json') or str(c['source_sha256']) != base.sha256(Path(__file__)):
        raise RuntimeError('complexity cache invalid: config/source hash mismatch')
    keep=s['subject_ids']!=config['excluded_subject']; subs=s['subject_ids'][keep]; z=lambda a,ids:normalize_by_subject(a,ids,'zscore')
    sm=np.asarray([str(n).startswith('log_bandpower_') for n in s['feature_names']]); X=np.column_stack([z(s['X_train'][keep][:,sm],subs),z(t['X_train'][keep],subs),z(p['X_train'][keep],subs)]); C=z(c['X_train'][keep],subs)
    ref=pd.read_csv((HERE/config['fixed_reference']).resolve());
    for col,arr in [('subject_id',subs),('health_group',s['health_groups'][keep]),('y_true',s['y_train'][keep]),('original_trial_id',s['original_trial_ids'][keep])]: np.testing.assert_array_equal(ref[col].to_numpy(),arr)
    ref=ref[['subject_id','health_group','original_trial_id','fold','y_true']]; assert X.shape==(2760,720) and C.shape==(2760,60) and np.isfinite(np.r_[X.ravel(),C.ravel()]).all(); return X,C,ref,s,c

def column_sets(): return {'control_pe':np.arange(720),'plus_complexity':np.arange(780)}
def fit_predict(X,C,y,train,ev,config,seed):
    A=np.column_stack([X,C]); out={}
    with threadpool_limits(limits=1):
        for m,cols in column_sets().items():
            model=base.make_model(config,seed); model.fit(A[train][:,cols],y[train]); out[m]=model.predict_proba(A[ev][:,cols])[:,1]
    return out
def job(X,C,y,name,folds,fold,config): return name,fold,fit_predict(X,C,y,folds!=fold,folds==fold,config,config['outer_model_seed_base']+fold)
def fit_public(X,C,XP,CP,y,config,seed):
    A=np.column_stack([X,C]); AP=np.column_stack([XP,CP]); out={}
    with threadpool_limits(limits=1):
        for m,cols in column_sets().items():
            model=base.make_model(config,seed); model.fit(A[:,cols],y); out[m]=model.predict_proba(AP[:,cols])[:,1]
    return out
def scalars(r): return {**{k:r['overall'][k] for k in KEYS[:3]},'subject_auc_mean':r['subject_auc_mean'],'hc_auc':r['health_groups']['HC']['pooled']['auc'],'dep_auc':r['health_groups']['DEP']['pooled']['auc'],'worst_subject_auc':min(v['auc'] for v in r['subjects'])}
def accepts(d,f,config):
    a=config['acceptance']; return bool(d['auc']>=a['repeat_delta_auc_min'] and d['f1']>=a['repeat_delta_f1_min'] and d['dep_auc']>=a['repeat_delta_dep_auc_min'] and f>a['fixed_delta_auc_strict_min'])
class Tee:
    def __init__(self,p): self.f=p.open('w',encoding='utf8'); self.s=sys.stdout
    def write(self,x): self.s.write(x); self.f.write(x); self.f.flush()
    def flush(self): self.s.flush(); self.f.flush()
def main():
    sys.stdout=Tee(HERE/'run_output.txt'); start=time.perf_counter(); config=json.loads((HERE/'config.json').read_text(encoding='utf8')); (HERE/'predictions').mkdir(exist_ok=True); (HERE/'input_metadata').mkdir(exist_ok=True); X,C,ref,s,c=load_data(config); y=ref.y_true.to_numpy(); subs=ref.subject_id.to_numpy(); groups=ref.health_group.to_numpy(); feature_seconds=time.perf_counter()-start; print(f'data ready train={X.shape} complexity={C.shape} elapsed={feature_seconds:.1f}s',flush=True)
    prov={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'inputs':{k:{'path':str((HERE/config[k]).resolve()),'sha256':base.sha256((HERE/config[k]).resolve())} for k in ('spectral_cache','temporal_cache','fixed_reference','complexity_cache')},'pe_cache_sha256':base.sha256((HERE.parent/'057_multiscale_permutation_entropy'/'cache/pe_features.npz')),'config_sha256':base.sha256(HERE/'config.json'),'script_sha256':base.sha256(Path(__file__)),'train_shape':[int(X.shape[0]),780],'complexity_shape':list(C.shape),'public_shape':None,'alignment_checked':list(ALIGN_KEYS),'subject_zscore':True,'public_metrics':False,'feature_names':c['feature_names'].tolist()}; (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8')
    splits=[('fixed002',ref.fold.to_numpy())]+[(f'repeat{q}',base.make_subject_folds(subs,groups,q)) for q in config['repeat_seeds']];
    for n,f in splits: base.assert_subject_isolation(subs,f); ref.assign(fold=f).to_csv(HERE/'input_metadata'/f'folds_{n}.csv',index=False)
    results=Parallel(n_jobs=config['n_jobs'])(delayed(job)(X,C,y,n,f,k,config) for n,f in splits for k in range(5)); pred={n:{m:np.full(len(y),np.nan) for m in METHODS} for n,_ in splits}; fmap=dict(splits)
    for n,k,o in results:
        for m in METHODS: pred[n][m][fmap[n]==k]=o[m]
    runs={m:{} for m in METHODS}; rows=[]
    for n,f in splits:
        rr=ref.assign(fold=f)
        for m in METHODS:
            p=pred[n][m]; assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all(); rr.assign(prob_positive=p).to_csv(HERE/'predictions'/f'{m}_{n}_oof.csv',index=False); r=base.summarize(y,p,rr,f); runs[m][n]=r; rows.append({'method':m,'split':n,**scalars(r)})
    # v2 intentionally does not extract new public features or train public predictors;
    # existing v1 public predictions remain untouched and are not v2 results.
    agg={}
    for m in METHODS:
        vals=[scalars(runs[m][f'repeat{q}']) for q in config['repeat_seeds']]; agg[m]={k:{'mean':float(np.mean([v[k] for v in vals])),'std':float(np.std([v[k] for v in vals],ddof=1)),'values':[v[k] for v in vals]} for k in KEYS}; rows.append({'method':m,'split':'repeat_mean',**{k:agg[m][k]['mean'] for k in KEYS}})
    delta={m:{'repeat_mean':{k:agg[m][k]['mean']-agg['control_pe'][k]['mean'] for k in KEYS},'fixed002':{k:scalars(runs[m]['fixed002'])[k]-scalars(runs['control_pe']['fixed002'])[k] for k in KEYS}} for m in METHODS[1:]}; decision={m:accepts(d['repeat_mean'],d['fixed002']['auc'],config) for m,d in delta.items()}; elapsed=time.perf_counter()-start
    metrics={'config':config,'provenance':prov,'runs':runs,'aggregate':agg,'deltas_vs_control':delta,'decision_accept':decision,'feature_loading_seconds':feature_seconds,'elapsed_seconds':elapsed}; (HERE/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8'); pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False)
    lines=['# 实验066：单窗非线性复杂度增量','','SampEn/LZC冻结协议见protocol.md；公开集仅输出预测，不计算公开指标。','', '|方法|五重复AUC均值±SD|F1均值|DEP AUC均值|','|---|---:|---:|---:|']+[f"|{m}|{agg[m]['auc']['mean']:.6f}±{agg[m]['auc']['std']:.6f}|{agg[m]['f1']['mean']:.6f}|{agg[m]['dep_auc']['mean']:.6f}|" for m in METHODS]+['','|方法|ΔAUC|ΔF1|ΔDEP AUC|fixed002 ΔAUC|达线|','|---|---:|---:|---:|---:|---|']+[f"|{m}|{delta[m]['repeat_mean']['auc']:+.6f}|{delta[m]['repeat_mean']['f1']:+.6f}|{delta[m]['repeat_mean']['dep_auc']:+.6f}|{delta[m]['fixed002']['auc']:+.6f}|{decision[m]}|" for m in delta]+['','未达线方法停止，不调参。',f'真实耗时：{elapsed:.2f}秒；特征提取与输入哈希：{feature_seconds:.2f}秒。']; (HERE/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8'); print(pd.DataFrame(rows).to_string(index=False)); print(json.dumps({'delta':delta,'decision':decision,'elapsed_seconds':elapsed},indent=2),flush=True)
if __name__=='__main__': main()
