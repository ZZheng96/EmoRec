import hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import softmax, expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; sys.path.insert(0, str(ROOT / 'src'))
from emo_rec.normalization import normalize_by_subject
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
def make_model(kind, fold):
    if kind == 'lr': e = LogisticRegression(C=CFG['linear_C'], max_iter=CFG['max_iter'], random_state=CFG['seed'] + fold)
    elif kind == 'svc': e = LinearSVC(C=CFG['linear_C'], dual=False, max_iter=CFG['max_iter'], random_state=CFG['seed'] + fold)
    else: e = HistGradientBoostingClassifier(**CFG['hgb'], random_state=CFG['seed'] + fold)
    return make_pipeline(StandardScaler(), e)
def prob(m, x): return expit(m.decision_function(x)) if isinstance(m[-1], LinearSVC) else m.predict_proba(x)[:, 1]
def metric(y, p): return {'auc': float(roc_auc_score(y, p)), 'f1': float(f1_score(y, p >= CFG['threshold'], zero_division=0))}
def soft_route(desc, tr_desc, tr_clusters, temperature):
    scale = np.maximum(np.std(tr_desc, axis=0), 1e-8); d = ((desc[:, None, :] - tr_desc[None, :, :]) / scale) ** 2
    centers = np.stack([tr_desc[tr_clusters == k].mean(0) for k in range(CFG['prototype_clusters'])])
    dist = ((desc[:, None, :] - centers[None, :, :]) / scale) ** 2
    return softmax(-dist.mean(2) / temperature, axis=1), centers, scale
def choose_weights(Xs, y, train_mask, folds):
    # Inner OOF is restricted to outer-training subjects; no outer validation labels enter selection.
    z = np.zeros((len(y), len(Xs))); subfold = folds.copy(); usable = train_mask
    for f in range(5):
        a, b = usable & (folds != f), usable & (folds == f)
        if not b.any(): continue
        for j, x in enumerate(Xs): z[b, j] = make_model('svc', 700 + f).fit(x[a], y[a]).decision_function(x[b])
    best, best_auc = CFG['score_weight_grid'][0], -1
    for w in CFG['score_weight_grid']:
        p = expit((z[usable] / np.maximum(z[usable].std(0), 1e-8)) @ np.asarray(w))
        a = roc_auc_score(y[usable], p)
        if a > best_auc: best, best_auc = w, a
    return best, best_auc
def main():
    start = time.perf_counter(); print('experiment 208 starting', flush=True)
    for d in ('predictions', 'input_metadata'): (HERE / d).mkdir(exist_ok=True)
    paths = {k: (HERE / CFG[k]).resolve() for k in ('de_cache','spectral_cache','temporal_cache','fixed_reference','baseline_oof')}
    arrays = {}
    for k in ('de_cache','spectral_cache','temporal_cache'):
        with np.load(paths[k], allow_pickle=False) as z: arrays[k] = {n: z[n] for n in ('X_train','subject_ids','health_groups','original_trial_ids','y_train')}
    d,s,t = [arrays[k] for k in ('de_cache','spectral_cache','temporal_cache')]
    for q in (s,t):
        for n in ('subject_ids','health_groups','original_trial_ids','y_train'): assert np.array_equal(d[n], q[n]), n
    keep = d['subject_ids'] != CFG['excluded_subject']; subs, groups, y = d['subject_ids'][keep], d['health_groups'][keep], d['y_train'][keep]
    ref = pd.read_csv(paths['fixed_reference']); keys=['subject_id','health_group','original_trial_id','y_true','fold']
    for col, arr in [('subject_id',subs),('health_group',groups),('original_trial_id',d['original_trial_ids'][keep]),('y_true',y)]: assert np.array_equal(ref[col].to_numpy(), arr), col
    ref=ref[keys].copy(); ref['row_index']=np.arange(len(ref)); ref['physical_trial_index']=ref.groupby('subject_id',sort=False).cumcount()//5
    folds=ref.fold.to_numpy(); subjects=np.unique(subs); sf=np.array([folds[subs==q][0] for q in subjects])
    z=lambda x: normalize_by_subject(x, subs, 'zscore')
    Xs=[z(d['X_train'][keep]), z(s['X_train'][keep,:150]), z(t['X_train'][keep])]; names=['de','spectral','temporal']
    raw=d['X_train'][keep]; desc=np.stack([np.r_[raw[subs==q].mean(0), np.log(np.maximum(raw[subs==q].std(0),1e-8))] for q in subjects])
    pred_names=['unified_hgb','prototype_hgb','hcdep_hgb','soft_routing_fusion','multikernel_oof_weights','multikernel_fixed_equal','baseline_057_plus_pe']
    P={n:np.full(len(y),np.nan) for n in pred_names}; audits=[]; scales=[]; selected=[]
    with threadpool_limits(limits=CFG['threads']):
      for fold in range(CFG['folds']):
        tr,va=folds!=fold,folds==fold; assert not set(subs[tr]) & set(subs[va])
        train_trials=set(zip(subs[tr],ref.physical_trial_index.to_numpy()[tr])); valid_trials=set(zip(subs[va],ref.physical_trial_index.to_numpy()[va])); assert not train_trials & valid_trials
        print(f'fold {fold+1}/5: train {len(set(subs[tr]))} subjects / valid {len(set(subs[va]))}',flush=True)
        unified=make_model('hgb',fold).fit(Xs[0][tr],y[tr]); pu=prob(unified,Xs[0][va]); P['unified_hgb'][va]=pu
        # hard prototype training models and unlabeled soft routing for validation subjects
        train_desc=desc[sf!=fold]; train_subs=subjects[sf!=fold]; scaler=StandardScaler().fit(train_desc); td=scaler.transform(train_desc); all_desc=scaler.transform(desc)
        from sklearn.cluster import KMeans
        km=KMeans(n_clusters=CFG['prototype_clusters'],n_init=20,random_state=CFG['seed']+fold).fit(td); cluster_all=km.predict(all_desc)
        dist=((all_desc[:,None,:]-km.cluster_centers_[None,:,:])**2).mean(2); rw=softmax(-dist/CFG['routing_temperature'],axis=1); route=np.array([rw[np.where(subjects==q)[0][0]] for q in subs])
        pp=np.zeros(va.sum()); valid_w=route[np.array([np.where(subjects==q)[0][0] for q in subs[va]])]
        for k in range(CFG['prototype_clusters']):
            ti=tr & (cluster_all[np.searchsorted(subjects,subs)]==k)
            if ti.sum() and len(np.unique(y[ti]))==2:
                pp += valid_w[:,k]*prob(make_model('hgb',fold+20+k).fit(Xs[0][ti],y[ti]),Xs[0][va])
            else: pp += valid_w[:,k]*pu
        P['prototype_hgb'][va]=pp
        ph=np.empty(va.sum())
        for g in ('HC','DEP'):
            ti=tr&(groups==g); vi=va&(groups==g); ph[groups[va]==g]=prob(make_model('hgb',fold+40).fit(Xs[0][ti],y[ti]),Xs[0][vi])
        P['hcdep_hgb'][va]=ph
        ps=[]
        for x in Xs:
            m=make_model('svc',fold).fit(x[tr],y[tr]); q=m.decision_function(x[va]); qtr=m.decision_function(x[tr]); mu,sd=qtr.mean(),max(qtr.std(),1e-8); ps.append((q-mu)/sd); scales.append({'fold':fold,'mean':float(mu),'std':float(sd)})
        pstack=np.column_stack(ps); pscore=expit(pstack@np.asarray(CFG['routing_weights'].values())) if False else (CFG['routing_weights']['unified']*pu+CFG['routing_weights']['prototype']*pp+CFG['routing_weights']['health_expert']*ph); P['soft_routing_fusion'][va]=pscore
        w, inner_auc=choose_weights(Xs,y,tr,folds); selected.append({'fold':fold,'weights':w,'inner_auc':inner_auc}); P['multikernel_oof_weights'][va]=expit(pstack@np.asarray(w)); P['multikernel_fixed_equal'][va]=expit(pstack@np.ones(3)/3)
        audits.append({'fold':fold,'train_subjects':train_subs.tolist(),'valid_subjects':subjects[sf==fold].tolist(),'routing_weights':rw.tolist(),'cluster_centers':km.cluster_centers_.tolist(),'physical_trial_intersection':0,'subject_intersection':0})
        print(f'fold {fold+1}/5 complete ({time.perf_counter()-start:.1f}s)',flush=True)
    base=pd.read_csv(paths['baseline_oof']); assert np.array_equal(ref[keys].to_numpy(),base[keys].to_numpy()); P['baseline_057_plus_pe']=base.prob_positive.to_numpy()
    rows=[]; details={}
    for n,p in P.items():
      assert np.isfinite(p).all(); frame=ref.assign(prob_positive=p); trial=frame.groupby(['subject_id','health_group','physical_trial_index','fold'],as_index=False).agg(y_true=('y_true','first'),prob_positive=('prob_positive','mean')); r={'method':n,**metric(y,p)}
      for g in ('HC','DEP'): r.update({g.lower()+'_'+k:v for k,v in metric(y[groups==g],p[groups==g]).items()})
      r.update({'trial_'+k:v for k,v in metric(trial.y_true.to_numpy(),trial.prob_positive.to_numpy()).items()}); fm=[metric(y[folds==f],p[folds==f]) for f in range(5)]; r.update({'worst_fold_auc':min(q['auc'] for q in fm),'worst_fold_f1':min(q['f1'] for q in fm)}); rows.append(r); details[n]={'summary':r,'folds':[{'fold':f,**fm[f]} for f in range(5)]}; frame.to_csv(HERE/'predictions'/f'{n}_oof.csv',index=False); trial.to_csv(HERE/'predictions'/f'{n}_trials.csv',index=False)
    ref.to_csv(HERE/'input_metadata/folds.csv',index=False); dump(HERE/'input_metadata/prototypes.json',audits); pd.DataFrame(selected).to_csv(HERE/'input_metadata/oof_score_weights.csv',index=False); pd.DataFrame(scales).to_csv(HERE/'input_metadata/score_scaling.csv',index=False)
    source=list(paths.values())+[Path(__file__),HERE/'config.json',ROOT/'排行.md',ROOT/'数据/数据集说明文档.md',HERE.parent/'207_leaderboard_method_transfer/run_experiment.py']; dump(HERE/'input_metadata/provenance.json',{'inputs':{str(q.relative_to(ROOT)):sha(q) for q in source},'versions':{'python':platform.python_version(),'numpy':np.__version__},'validation_labels_used_for_routing':False,'validation_labels_used_for_weights':False,'strict_subjectwise_oof':True,'physical_trial_splits_disjoint':True})
    pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False); elapsed=time.perf_counter()-start; dump(HERE/'metrics.json',{'config':CFG,'runs':details,'selected_oof_weights':selected,'elapsed_seconds':elapsed,'public_auc':None,'public_f1':None})
    tbl=['|方法|AUC|F1|HC AUC/F1|DEP AUC/F1|trial AUC/F1|最差折AUC/F1|','|---|---:|---:|---|---|---|---|']+[f"|{r['method']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['hc_auc']:.6f}/{r['hc_f1']:.6f}|{r['dep_auc']:.6f}/{r['dep_f1']:.6f}|{r['trial_auc']:.6f}/{r['trial_f1']:.6f}|{r['worst_fold_auc']:.6f}/{r['worst_fold_f1']:.6f}|" for r in rows]
    best=max(rows,key=lambda q:q['auc']); lines=['# 实验208：无标签相似度软路由与多特征SVM分数融合','','## 协议与防泄漏','','严格subject-wise固定5折，排除HC43；每个被试只在一个验证折，物理trial的5窗口不跨折。DE、频谱、时域均按被试无标签zscore。训练fold拟合原型标准化/KMeans，验证被试仅用自身无标签DE均值/log标准差计算距离并以温度softmax路由；HC/DEP专家仅作已知健康元数据对照。软融合权重固定于训练协议，SVM多特征权重仅由外层训练被试内部OOF网格选择，验证标签未参与路由、温度或权重。057 plus_pe为冻结对照。','','## 结果','']+tbl+['',f"事后最高窗口AUC：{best['method']}，AUC={best['auc']:.6f}，F1={best['f1']:.6f}；仅作描述性结果，不生成公开候选。",'公开集真实AUC/F1未知；本轮不加载公开数组、不生成公开预测。完整逐折、trial级、路由和权重审计见metrics.json与input_metadata。',f'总耗时：{elapsed:.2f}秒。']; (HERE/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8'); print(pd.DataFrame(rows).to_string(index=False),flush=True); print(f'COMPLETE 5/5 folds; {elapsed:.1f}s',flush=True)
if __name__=='__main__': main()
