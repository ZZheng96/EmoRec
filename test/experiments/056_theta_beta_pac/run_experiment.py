import hashlib, importlib.util, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt, hilbert
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('base032', HERE.parent/'032_repeated_subject_folds'/'run_experiment.py'); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
METHODS=('control_600','pac_only','plus_pac'); KEYS=('auc','f1','balanced_accuracy','subject_auc_mean','hc_auc','dep_auc','worst_subject_auc')
ALIGN_KEYS=('subject_ids','health_groups','original_trial_ids','y_train','public_user_ids','public_trial_ids')

def pac_mi(segments, config, return_surrogates=False):
    x=np.asarray(segments,dtype=float)
    if x.ndim!=3 or x.shape[1]!=30: raise ValueError('expected segments with shape (n, 30, samples)')
    fs=config['sampling_rate']; pc=config['pac']; edge=int(pc['edge_seconds']*fs); n=x.shape[-1]
    if n<=2*edge+int(pc['min_shift_seconds']*fs): raise ValueError('window too short for PAC edge/shift')
    theta=sosfiltfilt(butter(pc['filter_order'],pc['theta_hz'],btype='band',fs=fs,output='sos'),x,axis=-1)
    beta=sosfiltfilt(butter(pc['filter_order'],pc['beta_hz'],btype='band',fs=fs,output='sos'),x,axis=-1)
    phase=np.angle(hilbert(theta,axis=-1))[...,edge:-edge]; amp=np.abs(hilbert(beta,axis=-1))[...,edge:-edge]
    bins=np.linspace(-np.pi,np.pi,pc['n_phase_bins']+1); mi=np.empty(x.shape[:2]); z=np.empty_like(mi)
    valid_len=phase.shape[-1]; min_shift=int(pc['min_shift_seconds']*fs); shifts=[min_shift + i*max(1,(valid_len-2*min_shift)//pc['surrogates']) for i in range(pc['surrogates'])]
    def calc(a,ph):
        means=np.array([a[(ph>=bins[i])&(ph<bins[i+1])].mean() if np.any((ph>=bins[i])&(ph<bins[i+1])) else 0.0 for i in range(len(bins)-1)])
        p=means/np.maximum(means.sum(),1e-30); return float(np.sum(p*np.log(np.maximum(p*len(p),1e-30)))/np.log(len(p)))
    for i in range(x.shape[0]):
        for c in range(x.shape[1]):
            mi[i,c]=calc(amp[i,c],phase[i,c]); sur=[]
            for shift in shifts: sur.append(calc(np.roll(amp[i,c],shift),phase[i,c]))
            z[i,c]=(mi[i,c]-np.mean(sur))/max(np.std(sur,ddof=1),1e-12)
    return (z,mi) if return_surrogates else z

def build_cache(config,path):
    from emo_rec.data import load_train_segments,load_public_test_segments
    fs=[]; ys=[]; subs=[]; groups=[]; trials=[]; names=[f'pac_mi_z_{c}' for c in range(30)]
    files=sorted((ROOT/'数据'/'训练集').glob('*/*.mat'))
    for i,f in enumerate(files,1):
        seg,y,t=load_train_segments(f); fs.append(pac_mi(seg,config)); ys.append(y); trials.append(t); subs.extend([f.stem]*len(seg)); groups.extend(['DEP' if f.parent.name=='抑郁症患者' else 'HC']*len(seg)); print(f'PAC {i}/{len(files)} {f.stem}',flush=True)
    pub=[]; pu=[]; pt=[]
    for f in sorted((ROOT/'数据'/'公开测试集').glob('*.mat'),key=lambda p:int(p.stem.replace('P_test',''))):
        v=pac_mi(load_public_test_segments(f),config); pub.append(v); pu.extend([f.stem]*len(v)); pt.extend(np.arange(1,9))
    path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,X_train=np.concatenate(fs),y_train=np.concatenate(ys),subject_ids=np.asarray(subs),health_groups=np.asarray(groups),original_trial_ids=np.concatenate(trials),feature_names=np.asarray(names),X_public=np.concatenate(pub),public_user_ids=np.asarray(pu),public_trial_ids=np.asarray(pt))

def assert_alignment(a,b):
    for k in ALIGN_KEYS:
        if not np.array_equal(a[k],b[k]): raise RuntimeError(f'cache alignment mismatch: {k}')

def load_data(config):
    from emo_rec.normalization import normalize_by_subject
    s=base.load_cache((HERE/config['spectral_cache']).resolve()); t=base.load_cache((HERE/config['temporal_cache']).resolve()); base.assert_cache_alignment(s,t)
    path=HERE/config['pac_cache'];
    if not path.exists(): build_cache(config,path)
    p=base.load_cache(path); assert_alignment(s,p); keep=s['subject_ids']!=config['excluded_subject']; subs=s['subject_ids'][keep]; pub=s['public_user_ids']
    sm=np.asarray([str(n).startswith('log_bandpower_') for n in s['feature_names']]); assert sm.sum()==150 and t['X_train'].shape[1]==450 and p['X_train'].shape[1]==30
    z=lambda M,ids: normalize_by_subject(M,ids,'zscore')
    X=np.column_stack([z(s['X_train'][keep][:,sm],subs),z(t['X_train'][keep],subs)]); XP=np.column_stack([z(s['X_public'][:,sm],pub),z(t['X_public'],pub)])
    PX=z(p['X_train'][keep],subs); PP=z(p['X_public'],pub)
    ref=pd.read_csv((HERE/config['fixed_reference']).resolve()); assert np.array_equal(ref.subject_id.to_numpy(),subs) and np.array_equal(ref.y_true.to_numpy(),s['y_train'][keep]) and np.array_equal(ref.original_trial_id.to_numpy(),s['original_trial_ids'][keep]); ref=ref[['subject_id','health_group','original_trial_id','fold','y_true']]
    assert X.shape==(2760,600) and XP.shape==(80,600) and PX.shape==(2760,30) and PP.shape==(80,30) and np.isfinite(X).all() and np.isfinite(PX).all()
    return X,XP,PX,PP,ref,s,p

def column_sets(): return {'control_600':np.arange(600),'pac_only':np.arange(600,630),'plus_pac':np.arange(630)}
def fit_predict(X,P,y,train,ev,config,seed):
    out={}
    with threadpool_limits(limits=1):
        for m,c in column_sets().items():
            A=np.column_stack([X,P]); model=base.make_model(config,seed); model.fit(A[train][:,c],y[train]); out[m]=model.predict_proba(A[ev][:,c])[:,1]
    return out
def job(X,P,y,name,folds,fold,config):
    tr,va=folds!=fold,folds==fold; return name,fold,fit_predict(X,P,y,tr,va,config,config['outer_model_seed_base']+fold)
def fit_public(X,P,XP,PP,y,config,seed):
    A=np.column_stack([X,P]); AP=np.column_stack([XP,PP]); out={}
    with threadpool_limits(limits=1):
        for m,c in column_sets().items():
            model=base.make_model(config,seed); model.fit(A[:,c],y); out[m]=model.predict_proba(AP[:,c])[:,1]
    return out
def scalars(r): return {**{k:r['overall'][k] for k in KEYS[:3]},'subject_auc_mean':r['subject_auc_mean'],'hc_auc':r['health_groups']['HC']['pooled']['auc'],'dep_auc':r['health_groups']['DEP']['pooled']['auc'],'worst_subject_auc':min(x['auc'] for x in r['subjects'])}
def summarize(y,p,ref,folds): return base.summarize(y,p,ref,folds)
def diagnostics(P,y,subs,groups):
    rows=[]
    for c in range(P.shape[1]):
        ds=[]
        for s in np.unique(subs):
            a=P[(subs==s)&(y==1),c]; b=P[(subs==s)&(y==0),c]; sd=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/max(len(a)+len(b)-2,1)); ds.append(((a.mean()-b.mean())/max(sd,1e-12),groups[subs==s][0]))
        d=np.asarray([q[0] for q in ds]); g=np.asarray([q[1] for q in ds]); sign=np.sign(d.mean()); rows.append({'channel':c,'mean_d':float(d.mean()),'abs_d_HC':float(np.abs(d[g=='HC']).mean()),'abs_d_DEP':float(np.abs(d[g=='DEP']).mean()),'direction_consistency_HC':float((np.sign(d[g=='HC'])==sign).mean()),'direction_consistency_DEP':float((np.sign(d[g=='DEP'])==sign).mean())})
    return pd.DataFrame(rows)
class Tee:
    def __init__(self,p): self.f=open(p,'w',encoding='utf8'); self.s=sys.stdout
    def write(self,x): self.s.write(x); self.f.write(x); self.f.flush()
    def flush(self): self.s.flush(); self.f.flush()
def main():
    sys.stdout=Tee(HERE/'run_output.txt'); start=time.perf_counter(); config=json.loads((HERE/'config.json').read_text(encoding='utf8')); (HERE/'predictions').mkdir(exist_ok=True); (HERE/'input_metadata').mkdir(exist_ok=True)
    X,XP,P,PP,ref,s,pc=load_data(config); y=ref.y_true.to_numpy(); subs=ref.subject_id.to_numpy(); groups=ref.health_group.to_numpy(); print(f'data ready X={X.shape} PAC={P.shape}',flush=True)
    prov={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'train_shape':list(X.shape),'pac_shape':list(P.shape),'public_shape':list(XP.shape),'alignment_checked':list(ALIGN_KEYS),'subject_zscore':True,'public_metrics':False}; (HERE/'input_metadata'/'provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8')
    splits=[('fixed002',ref.fold.to_numpy())]+[(f'repeat{q}',base.make_subject_folds(subs,groups,q)) for q in config['repeat_seeds']]
    for _,f in splits: base.assert_subject_isolation(subs,f)
    tasks=[delayed(job)(X,P,y,n,f,k,config) for n,f in splits for k in range(5)]; results=Parallel(n_jobs=config['n_jobs'])(tasks); pred={n:{m:np.full(len(y),np.nan) for m in METHODS} for n,_ in splits}; fmap=dict(splits)
    for n,k,o in results:
        for m in METHODS: pred[n][m][fmap[n]==k]=o[m]
    runs={m:{} for m in METHODS}; rows=[]
    for n,f in splits:
        rr=ref.copy(); rr['fold']=f
        for m in METHODS:
            p=pred[n][m]; assert np.isfinite(p).all(); rr.assign(prob_positive=p).to_csv(HERE/'predictions'/f'{m}_{n}_oof.csv',index=False); r=summarize(y,p,rr,f); runs[m][n]=r; rows.append({'method':m,'split':n,**scalars(r)})
    public=fit_public(X,P,XP,PP,y,config,config['outer_model_seed_base'])
    for m in METHODS: pd.DataFrame({'user_id':s['public_user_ids'],'trial_id':s['public_trial_ids'],'prob. of positive emotion':public[m]}).to_csv(HERE/'predictions'/f'{m}_public.csv',index=False)
    agg={};
    for m in METHODS:
        vals=[scalars(runs[m][f'repeat{q}']) for q in config['repeat_seeds']]; agg[m]={k:{'mean':float(np.mean([v[k] for v in vals])),'std':float(np.std([v[k] for v in vals],ddof=1)),'values':[v[k] for v in vals]} for k in KEYS}; rows.append({'method':m,'split':'repeat_mean',**{k:agg[m][k]['mean'] for k in KEYS}})
    delta={m:{'repeat_mean':{k:agg[m][k]['mean']-agg['control_600'][k]['mean'] for k in KEYS},'fixed002':{k:scalars(runs[m]['fixed002'])[k]-scalars(runs['control_600']['fixed002'])[k] for k in KEYS}} for m in METHODS[1:]}; decision={m:bool(d['repeat_mean']['auc']>=.005 and d['repeat_mean']['f1']>=0 and d['repeat_mean']['dep_auc']>=0 and d['fixed002']['auc']>0) for m,d in delta.items()}
    diag=diagnostics(P,y,subs,groups); diag.to_csv(HERE/'pac_diagnostics.csv',index=False); elapsed=time.perf_counter()-start; metrics={'config':config,'provenance':prov,'aggregate':agg,'deltas_vs_control':delta,'decision_accept':decision,'pac_diagnostics':diag.to_dict('records'),'elapsed_seconds':elapsed}; (HERE/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8'); table=pd.DataFrame(rows); table.to_csv(HERE/'summary.csv',index=False); write_summary(metrics,table); print(table.to_string(index=False)); print(f'elapsed_seconds={elapsed:.1f}',flush=True)
def write_summary(metrics,table):
    a=metrics['aggregate']; d=metrics['deltas_vs_control']; lines=['# 实验056：单窗 theta-beta PAC 筛查','','训练69被试2760窗，排除HC43；复用001/009/032/055缓存、固定002折、HGB、阈值0.5、训练seed=2052+fold及2027..2031重复。control_600为历史协议逐被试无标签zscore；PAC为每通道4–8Hz相位与15–30Hz幅度、Butterworth+Hilbert，去两端0.5秒，18相位bin归一化KL，8次本窗循环移位surrogate校正z-score。公开集只输出预测，不计算指标。','','这是固定筛查配置，不是全PAC搜索；PAC阳性不能解释为机制证明，伪PAC、滤波边缘、波形非正弦和surrogate有限次数均可能影响结果。严格单窗预处理未作为主结果，不能与历史协议直接比较。','','## 预注册接受线','相对control：五重复均值ΔAUC≥+0.005、ΔF1≥0、ΔDEP AUC≥0，且fixed002 ΔAUC>0；不达线停止不调参。','','## 结果','']
    lines += ['|方法|分割|AUC|F1|BAcc|subject AUC|HC AUC|DEP AUC|','|---|---|---:|---:|---:|---:|---:|---:|']+[f"|{r['method']}|{r['split']}|{r['auc']:.4f}|{r['f1']:.4f}|{r['balanced_accuracy']:.4f}|{r['subject_auc_mean']:.4f}|{r['hc_auc']:.4f}|{r['dep_auc']:.4f}|" for _,r in table.iterrows()]
    lines += ['','## 相对control及诊断','']+[f"- {m}: ΔAUC={d[m]['repeat_mean']['auc']:+.4f}, ΔF1={d[m]['repeat_mean']['f1']:+.4f}, ΔDEP AUC={d[m]['repeat_mean']['dep_auc']:+.4f}, fixed002 ΔAUC={d[m]['fixed002']['auc']:+.4f}；达线={'是' if metrics['decision_accept'][m] else '否'}。" for m in d]+['- 每通道HC/DEP被试内平均|d|与方向一致率见pac_diagnostics.csv；仅描述性诊断。','',f"耗时：{metrics['elapsed_seconds']:.1f}秒。"]
    (HERE/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__': main()
