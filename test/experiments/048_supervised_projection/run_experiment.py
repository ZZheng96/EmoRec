import hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE=Path(__file__).resolve().parent; OUT=HERE/'predictions'

def load(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k] for k in z.files}
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def align(a,b):
    for k in ('subject_ids','health_groups','original_trial_ids','y_train','public_user_ids','public_trial_ids'):
        if not np.array_equal(a[k],b[k]): raise RuntimeError(f'cache alignment mismatch: {k}')
def folds(subjects, groups, seed):
    us=np.unique(subjects); ug=np.array([groups[np.flatnonzero(subjects==s)[0]] for s in us]); out={}
    for f,(_,v) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(us,ug)):
        for s in us[v]: out[s]=f
    return np.array([out[s] for s in subjects], dtype=int)
def hgb(cfg, seed): return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**cfg['hgb'],random_state=seed))
def metric(y,p):
    q=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
    return {'auc':float(roc_auc_score(y,p)),'f1':float(f1_score(y,q,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,q)),'threshold':.5,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
def summary(y,p,ref,fs):
    sr=[]
    for s in np.unique(ref.subject_id):
        m=ref.subject_id.to_numpy()==s; sr.append({'subject_id':s,'health_group':ref.loc[m,'health_group'].iloc[0],**metric(y[m],p[m])})
    gm={}
    for g in ('HC','DEP'):
        rows=[r for r in sr if r['health_group']==g]; m=ref.health_group.to_numpy()==g
        gm[g]={'n_subjects':len(rows),'pooled':metric(y[m],p[m]),'subject_auc_mean':float(np.mean([r['auc'] for r in rows])),'subject_auc_std':float(np.std([r['auc'] for r in rows],ddof=1))}
    return {'overall':metric(y,p),'folds':[{'fold':f,'n_subjects':int(ref.loc[fs==f,'subject_id'].nunique()),'n_samples':int((fs==f).sum()),**metric(y[fs==f],p[fs==f])} for f in range(5)],'subject_auc_mean':float(np.mean([r['auc'] for r in sr])),'subject_auc_std':float(np.std([r['auc'] for r in sr],ddof=1)),'health_groups':gm,'subjects':sr}
def public(ids,trials,p):
    x=pd.DataFrame({'user_id':ids,'trial_id':trials,'prob. of positive emotion':p}); validate_submission(x,[f'P_test{i}' for i in range(1,11)]); return x
def fit_project(Xtr,ytr,Xv,Xp):
    pipe=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(n_components=1,solver='lsqr',shrinkage='auto')); pipe.fit(Xtr,ytr)
    def score(x): return pipe.decision_function(x).reshape(-1,1)
    return np.concatenate([Xtr,score(Xtr)],1),np.concatenate([Xv,score(Xv)],1),np.concatenate([Xp,score(Xp)],1),pipe

def run_method(name, kind, Xs,Xt,Ps,Pt,y,ref,allfolds,cfg,public_ids,public_trials):
    repeats=[]; audits=[]; publics=[]
    for ri,fs in enumerate(allfolds,1):
        p=np.full(len(y),np.nan)
        for f in range(5):
            tr=fs!=f; va=~tr; seed=2026+f
            if kind=='spectral': model=hgb(cfg,seed); model.fit(Xs[tr],y[tr]); p[va]=model.predict_proba(Xs[va])[:,1]
            elif kind=='fusion':
                a=hgb(cfg,seed); b=hgb(cfg,seed); a.fit(Xs[tr],y[tr]); b.fit(Xt[tr],y[tr]); p[va]=.5*a.predict_proba(Xs[va])[:,1]+.5*b.predict_proba(Xt[va])[:,1]
            else:
                ztr,zv,_,proj=fit_project(np.concatenate([Xs[tr],Xt[tr]],1),y[tr],np.concatenate([Xs[va],Xt[va]],1),np.zeros((1,Xs.shape[1]+Xt.shape[1])))
                model=hgb(cfg,seed); model.fit(ztr,y[tr]); p[va]=model.predict_proba(zv)[:,1]
                audits.append({'repeat':ri,'fold':f,'training_rows':int(tr.sum()),'training_subjects':sorted(np.unique(ref.subject_id.to_numpy()[tr]).tolist()),'projection_coefficients':proj[-1].coef_.ravel().tolist(),'projection_intercept':float(proj[-1].intercept_[0]),'projection_scaler_mean':proj[0].mean_.tolist(),'projection_scaler_scale':proj[0].scale_.tolist()})
        if not np.isfinite(p).all(): raise RuntimeError(f'incomplete OOF: {name}')
        rr=ref.copy(); rr['fold']=fs; rr['prob_positive']=p; rr.to_csv(OUT/f'{name}_repeat{ri:02d}_oof.csv',index=False)
        r=summary(y,p,rr,fs); r.update(repeat=ri,seed=(cfg['fixed_fold_seed'] if ri == 1 else cfg['repeat_seeds'][ri-2]),fold_set=('fixed_002' if ri == 1 else 'repeat')); repeats.append(r)
        if kind=='spectral': m=hgb(cfg,2026);m.fit(Xs,y); publics.append(m.predict_proba(Ps)[:,1])
        elif kind=='fusion':
            a=hgb(cfg,2026);b=hgb(cfg,2026);a.fit(Xs,y);b.fit(Xt,y);publics.append(.5*a.predict_proba(Ps)[:,1]+.5*b.predict_proba(Pt)[:,1])
        else:
            ztr,_,zp,_=fit_project(np.concatenate([Xs,Xt],1),y,np.zeros((1,Xs.shape[1]+Xt.shape[1])),np.concatenate([Ps,Pt],1));m=hgb(cfg,2026);m.fit(ztr,y);publics.append(m.predict_proba(zp)[:,1])
    agg={k:{'mean':float(np.mean([r['overall'][k] for r in repeats])),'std':float(np.std([r['overall'][k] for r in repeats],ddof=1)),'values':[r['overall'][k] for r in repeats]} for k in ('auc','f1','balanced_accuracy')}
    for k in ('subject_auc_mean',): agg[k]={'mean':float(np.mean([r[k] for r in repeats])),'std':float(np.std([r[k] for r in repeats],ddof=1)),'values':[r[k] for r in repeats]}
    for g in ('HC','DEP'): 
        v=[r['health_groups'][g]['pooled']['auc'] for r in repeats];agg[g.lower()+'_auc']={'mean':float(np.mean(v)),'std':float(np.std(v,ddof=1)),'values':v}
    public(public_ids,public_trials,np.mean(publics,axis=0)).to_csv(OUT/f'{name}_public.csv',index=False)
    return {'n_features':601 if kind=='projection' else (150 if kind=='spectral' else 600),'repeats':repeats,'aggregate':agg,'projection_audit':audits}

def main():
    t=time.time(); cfg=json.loads((HERE/'config.json').read_text()); OUT.mkdir(exist_ok=True)
    s=load((HERE/cfg['spectral_cache']).resolve()); q=load((HERE/cfg['temporal_cache']).resolve());align(s,q)
    keep=s['subject_ids']!=cfg['excluded_subject']; subjects=s['subject_ids'][keep]; y=s['y_train'][keep]; groups=s['health_groups'][keep]
    if len(y)!=cfg['expected_train_rows']: raise RuntimeError('unexpected row count')
    ref=pd.DataFrame({'subject_id':subjects,'health_group':groups,'original_trial_id':s['original_trial_ids'][keep],'fold':-1,'y_true':y})
    sx=normalize_by_subject(s['X_train'][keep,:150],subjects,'zscore'); tx=normalize_by_subject(q['X_train'][keep],subjects,'zscore'); sp=normalize_by_subject(s['X_public'][:,:150],s['public_user_ids'],'zscore');tp=normalize_by_subject(q['X_public'],q['public_user_ids'],'zscore')
    fs=[folds(subjects,groups,x) for x in [cfg['fixed_fold_seed']]+cfg['repeat_seeds']]; [pd.DataFrame({'subject_id':subjects,'health_group':groups,'fold':f}).to_csv(OUT/f'folds_seed{seed}.csv',index=False) for seed,f in zip([2026]+cfg['repeat_seeds'],fs)]
    runs={}; summ=[]
    with threadpool_limits(limits=1):
        for n,k in [('spectral_control','spectral'),('spectral_temporal_control','fusion'),('supervised_projection','projection')]:
            r=run_method(n,k,sx,tx,sp,tp,y,ref,fs,cfg,s['public_user_ids'],s['public_trial_ids']);runs[n]=r;summ.append({'run_name':n,**{x:r['aggregate'][x]['mean'] for x in ('auc','f1','balanced_accuracy','subject_auc_mean','hc_auc','dep_auc')}})
    result={'experiment_config':cfg,'provenance':{'spectral_cache_sha256':sha((HERE/cfg['spectral_cache']).resolve()),'temporal_cache_sha256':sha((HERE/cfg['temporal_cache']).resolve()),'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'elapsed_seconds':time.time()-t},'protocol':{'labels_only_projection_fit':True,'subjectwise_unlabeled_zscore':True,'subject_isolation':True,'fixed_threshold':.5,'independent_repeats':cfg['repeat_seeds']},'runs':runs}
    (HERE/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));pd.DataFrame(summ).to_csv(HERE/'summary.csv',index=False)
    best=runs['supervised_projection']['aggregate']; ctl=runs['spectral_temporal_control']['aggregate']; verdict='提升' if best['auc']['mean']>ctl['auc']['mean'] and best['dep_auc']['mean']>ctl['dep_auc']['mean'] else '未提升，按协议停止，不调参'
    (HERE/'实验报告.md').write_text('# 实验048：训练折内标签条件监督投影\n\n固定LDA一维判别投影仅在每个训练折拟合，投影分数与600维subject-zscore频谱+时序特征拼接后使用固定HGB。验证/公开仅transform；控制在同一脚本逐折重训，五次重复独立OOF。\n\n## 结果\n\n'+pd.DataFrame(summ).to_string(index=False)+f'\n\n结论：{verdict}。DEP为主要判据。\n',encoding='utf-8')
    print(pd.DataFrame(summ).to_string(index=False));print(f'elapsed_seconds={time.time()-t:.3f}')
if __name__=='__main__': main()
