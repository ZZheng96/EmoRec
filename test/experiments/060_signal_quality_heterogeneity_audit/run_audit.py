import hashlib, json, math, platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE/'config.json').read_text(encoding='utf8'))
SEED = CFG['seed']
rng = np.random.default_rng(SEED)

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<3 or np.all(x==x[0]) or np.all(y==y[0]): return np.nan
    return float(np.corrcoef(rankdata(x,method='average'),rankdata(y,method='average'))[0,1])

def quality_table():
    src=(HERE/CFG['quality_source']).resolve(); d=json.loads(src.read_text(encoding='utf8'))
    rows=[]
    for f in d['files']:
        if f.get('role')!='train': continue
        sid=Path(f['path']).stem; vals=[]
        for key in ('EEG_data_neu','EEG_data_pos'):
            a=f['arrays'][key]; std=float(a['std']); center=max(abs(float(a['min'])-float(a['mean'])),abs(float(a['max'])-float(a['mean']))) / std
            vals.append((std,center))
        rows.append({'subject_id':sid,'health_group':'HC' if f['group']=='正常人' else 'DEP', 'std_proxy':np.mean([v[0] for v in vals]), 'range_sd_proxy':np.mean([v[1] for v in vals]), 'hc43_reference_only':sid=='HC43'})
    q=pd.DataFrame(rows).sort_values('subject_id').reset_index(drop=True)
    assert len(q)==70 and q.subject_id.nunique()==70
    return q

def load_set(directory, splits, suffix):
    base=None; preds={m:{} for m in CFG['methods']}
    for split in splits:
        split_meta=None
        for method in CFG['methods']:
            p=HERE/CFG[directory]/f'{method}_{suffix}{split}_oof.csv'
            x=pd.read_csv(p)
            invariant=['subject_id','health_group','original_trial_id','y_true']
            if base is None: base=x[invariant].copy()
            else: assert base[invariant].equals(x[invariant]), f'metadata mismatch: {p}'
            paired = invariant + ['fold']
            if split_meta is None: split_meta=x[paired].copy()
            else: assert split_meta.equals(x[paired]), f'paired metadata mismatch: {p}'
            assert x.groupby('subject_id').size().eq(40).all()
            assert set(x.y_true.unique()) == {0, 1}
            assert np.isfinite(x.prob_positive).all() and x.prob_positive.between(0, 1).all()
            assert len(x)==2760 and x.subject_id.nunique()==69
            assert all(x.groupby('fold').subject_id.nunique().gt(0))
            assert all(x.groupby('subject_id').fold.nunique()==1)
            preds[method][split]=x.prob_positive.to_numpy(float)
    assert base.subject_id.nunique()==69 and len(base)==2760
    assert base.groupby('subject_id').health_group.nunique().eq(1).all()
    assert base.groupby('subject_id').original_trial_id.nunique().eq(4).all()
    return base,preds

def subject_metrics(meta,preds,splits):
    subs=sorted(meta.subject_id.unique()); out=pd.DataFrame({'subject_id':subs})
    out['health_group']=out.subject_id.map(meta.drop_duplicates('subject_id').set_index('subject_id').health_group)
    for method in CFG['methods']:
        aucs=[]
        for split in splits:
            x=meta.assign(p=preds[method][split]); vals=x.groupby('subject_id',sort=True).apply(lambda z: roc_auc_score(z.y_true,z.p),include_groups=False)
            aucs.append(vals.reindex(subs).to_numpy())
        out[f'{method}_auc']=np.mean(aucs,axis=0)
    out['gain']=out.plus_pe_auc-out.control_600_auc
    return out

def bh(p):
    p=np.asarray(p,float); q=np.full(len(p),np.nan); ok=np.isfinite(p); inds=np.where(ok)[0]; order=inds[np.argsort(p[ok])]; vals=p[order]; adj=np.minimum.accumulate((vals*len(vals)/np.arange(1,len(vals)+1))[::-1])[::-1]; q[order]=np.minimum(adj,1); return q

def infer(df, quality_col, endpoint, group):
    z=df if group=='all' else df[df.health_group==group]
    x=z[quality_col].to_numpy(); y=z[endpoint].to_numpy(); observed=spearman(x,y)
    n=len(z); boot=[]
    for _ in range(CFG['bootstrap_reps']):
        if group=='all':
            ix=np.concatenate([rng.choice(np.flatnonzero(df.health_group.to_numpy()==g),np.sum(df.health_group.to_numpy()==g),replace=True) for g in ('HC','DEP')])
            boot.append(spearman(df[quality_col].to_numpy()[ix],df[endpoint].to_numpy()[ix]))
        else:
            ix=rng.integers(0,n,n); boot.append(spearman(x[ix],y[ix]))
    ci=np.nanpercentile(boot,[2.5,97.5])
    perm=[]
    for _ in range(CFG['permutation_reps']):
        if group=='all':
            yp=y.copy()
            for g in ('HC','DEP'):
                mask=z.health_group.to_numpy()==g; yp[mask]=rng.permutation(yp[mask])
            perm.append(spearman(x,yp))
        else: perm.append(spearman(x,rng.permutation(y)))
    perm=np.asarray(perm)
    # Stratified permutations preserve group differences: their null need not center at zero.
    p=min(1.0, 2*min((1+np.sum(perm>=observed))/(len(perm)+1), (1+np.sum(perm<=observed))/(len(perm)+1))) if np.isfinite(observed) else np.nan
    return {'quality_proxy':quality_col,'endpoint':endpoint,'group':group,'n_subjects':n,'rho':observed,'ci_low':float(ci[0]),'ci_high':float(ci[1]),'p_perm_two_sided':float(p)}

def main():
    q=quality_table(); q69=q[q.subject_id!='HC43'].copy()
    m58,p58=load_set('primary_predictions',CFG['primary_splits'],'split'); s58=subject_metrics(m58,p58,CFG['primary_splits'])
    m57,p57=load_set('repeat_predictions',CFG['repeat_splits'],'repeat'); s57=subject_metrics(m57,p57,CFG['repeat_splits'])
    assert m58[['subject_id','health_group']].drop_duplicates().sort_values('subject_id').reset_index(drop=True).equals(m57[['subject_id','health_group']].drop_duplicates().sort_values('subject_id').reset_index(drop=True))
    table=q69.merge(s58,on='subject_id').merge(s57[['subject_id','control_600_auc','plus_pe_auc','gain']],on='subject_id',suffixes=('_primary','_repeat')).rename(columns={'health_group_x':'health_group'})
    table.to_csv(HERE/'tables/subject_quality_vs_oof.csv',index=False)
    results=[]
    for label,df in [('primary',table)]:
        for proxy in ('std_proxy','range_sd_proxy'):
            for endpoint in ('plus_pe_auc_primary','gain_primary'):
                for group in ('all','HC','DEP'): results.append({'set':label,**infer(df,proxy,endpoint,group)})
    res=pd.DataFrame(results); res['q_bh']=bh(res.p_perm_two_sided.to_numpy())
    repeat=[]
    for proxy in ('std_proxy','range_sd_proxy'):
        for endpoint in ('plus_pe_auc_repeat','gain_repeat'):
            for group in ('all','HC','DEP'): repeat.append({'set':'repeat_direction_only',**infer(table,proxy,endpoint,group)})
    rep=pd.DataFrame(repeat); rep['q_bh']=bh(rep.p_perm_two_sided.to_numpy())
    cfg_hash=sha256(HERE/'config.json'); metrics={'config':CFG,'provenance':{'python':platform.python_version(),'config_sha256':cfg_hash,'script_sha256':sha256(HERE/'run_audit.py'),'quality_source_sha256':sha256((HERE/CFG['quality_source']).resolve()),'prediction_sha256':{str(p):sha256(p) for d,ss,suf in [('primary_predictions',CFG['primary_splits'],'split'),('repeat_predictions',CFG['repeat_splits'],'repeat')] for s in ss for m in CFG['methods'] for p in [(HERE/CFG[d]/f'{m}_{suf}{s}_oof.csv')]}},'n_subjects':69,'hc43':'retained in source audit but excluded from analysis and recorded as reference only','no_public_predictions':True,'primary':res.to_dict('records'),'repeat_direction_only':rep.to_dict('records')}
    (HERE/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 060 信号质量异质性审计','','本轮获得用户授权后运行；固定 seed=2060。仅使用已生成的058 split OOF与057 repeat OOF，不训练、不选择阈值、不删分析对象、不使用公开预测。质量代理是粗粒度信号尺度/极值相对标准差指标，不是确诊 artifact，也不报告伪迹率。','',f'主分析：69名被试；HC43仅作为mat审计参照记录，不进入相关分析。12个相关（2代理×2 endpoint×3群组）在本检验family内做BH校正；Spearman，2000次分组bootstrap CI，4999次组内置换双侧p。all组置换在HC/DEP内进行，bootstrap按组抽样。057 repeat仅作方向稳定性，不是独立验证。','']
    lines += ['## 主分析结果','',res[['quality_proxy','endpoint','group','rho','ci_low','ci_high','p_perm_two_sided','q_bh']].to_string(index=False),'','## 057 repeat方向稳定性（非独立验证）','',rep[['quality_proxy','endpoint','group','rho','ci_low','ci_high','p_perm_two_sided','q_bh']].to_string(index=False),'','## 方法动机与边界','', 'PREP原论文讨论多维坏通道检测与鲁棒参考：https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2015.00016/full；Autoreject原论文采用逐传感器峰峰值阈值并交叉验证：https://arxiv.org/html/1612.08194v3。本文整段极值/std代理不能替代逐窗逐通道检测，弱关联不能排除伪迹；没有0.95的新性能证据，也不构成伪迹率估计。因果与上限解释边界沿用053理论，不修改053。','', '结果与源文件hash见 `metrics.json`；逐被试合并表见 `tables/subject_quality_vs_oof.csv`。']
    (HERE/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__': main()
