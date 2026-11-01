import hashlib, importlib.util, json, platform, time
from pathlib import Path
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; SRC=ROOT/'test/experiments/064_spectral_shape_dynamics'
spec=importlib.util.spec_from_file_location('e064', SRC/'run_experiment.py'); e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); a=a-a.mean(); b=b-b.mean(); den=np.sqrt((a*a).sum()*(b*b).sum()); return np.nan if den==0 else float((a*b).sum()/den)
def cohend(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); den=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2); return np.nan if den==0 else float((x.mean()-y.mean())/den)
def main():
    t=time.perf_counter(); c=json.loads((SRC/'config.json').read_text(encoding='utf8')); A,ref,s,o=e.load_data(c)
    assert A.shape==(2760,780) and np.isfinite(A).all()
    ids=ref.subject_id.to_numpy(); y=ref.y_true.to_numpy(); groups=ref.health_group.to_numpy(); js=A[:,720:]; base=A[:,:720]; names=o['feature_names'].tolist()[0:60]
    assert len(np.unique(ids))==69
    subs=np.unique(ids); assert all(np.sum(ids==u)==40 for u in subs)
    controls=np.empty((60,720)); pooled=np.empty((60,720)); subject_diffs=[]; effects=[]
    for j in range(60):
      for k in range(720):
        pooled[j,k]=corr(js[:,j],base[:,k])
        vals=[corr(js[ids==u,j],base[ids==u,k]) for u in subs]; controls[j,k]=np.nanmean(vals)
      for g in ('HC','DEP'):
        sel=(groups==g); effects.append({'feature':names[j],'group':g,'n_subjects':int(len(np.unique(ids[sel]))),'window_cohen_d':cohend(js[sel,j][y[sel]==1],js[sel,j][y[sel]==0])})
    # per-subject diagnostic effects and leave-one-subject-out direction consistency
    for j in range(60):
      row={'feature':names[j]}
      for g in ('HC','DEP'):
        us=[u for u in subs if np.any((ids==u)&(groups==g))]; ds=[]
        for u in us:
          q=(ids==u); ds.append(cohend(js[q,j][y[q]==1],js[q,j][y[q]==0]))
        ds=np.asarray(ds,float); loo=[]; agree=[]; zero=0
        for i,u in enumerate(us):
          other=np.delete(ds,i); refd=np.nanmean(other); d=ds[i]
          if not np.isfinite(d) or not np.isfinite(refd) or d==0 or refd==0: zero+=1
          else: agree.append(float(np.sign(d)==np.sign(refd)))
        row.update({f'{g}_mean_subject_d':float(np.nanmean(ds)),f'{g}_median_abs_subject_d':float(np.nanmedian(np.abs(ds))),f'{g}_direction_consistency':float(np.mean(agree)) if agree else np.nan,f'{g}_zero_or_invalid_count':zero})
      subject_diffs.append(row)
    maxp=np.nanmax(np.abs(pooled),axis=1); maxi=np.nanargmax(np.abs(pooled),axis=1); maxs=np.nanmax(np.abs(controls),axis=1); maxsi=np.nanargmax(np.abs(controls),axis=1)
    fd=pd.DataFrame({'feature':names,'pooled_max_abs_corr':maxp,'pooled_control_feature':[int(x) for x in maxi],'mean_subject_max_abs_corr':maxs,'mean_subject_control_feature':[int(x) for x in maxsi],'pooled_vs_mean_subject_max_abs_diff':np.nanmax(np.abs(pooled-controls),axis=1)})
    fd.to_csv(HERE/'results/feature_diagnostics.csv',index=False); pd.DataFrame(subject_diffs).to_csv(HERE/'results/subject_feature_effects.csv',index=False); pd.DataFrame(effects).to_csv(HERE/'results/group_effects.csv',index=False)
    prov={'python':platform.python_version(),'numpy':np.__version__,'source':[{'path':str(SRC/'run_experiment.py'),'sha256':sha(SRC/'run_experiment.py')},{'path':str(SRC/'config.json'),'sha256':sha(SRC/'config.json')},{'path':str(SRC/'cache/spectral_js_features.npz'),'sha256':sha(SRC/'cache/spectral_js_features.npz')}],'protocol_sha256':sha(HERE/'protocol.md'),'shape':list(A.shape),'subjects':len(subs),'rows_per_subject':40,'label_use':'descriptive diagnostics only','elapsed_seconds':time.perf_counter()-t}
    out={'provenance':prov,'alignment':{'finite':True,'subjects':list(subs),'group_counts':{g:int(np.sum(groups==g)) for g in ('HC','DEP')},'rows_per_subject':{u:int(np.sum(ids==u)) for u in subs}},'correlation':{'pooled_max_abs':float(np.nanmax(maxp)),'mean_subject_max_abs':float(np.nanmax(maxs)),'max_matrix_abs_diff':float(np.nanmax(np.abs(pooled-controls))),'nonfinite_pooled':int(np.sum(~np.isfinite(pooled))),'nonfinite_subject_mean':int(np.sum(~np.isfinite(controls)))},'effects':effects}
    (HERE/'results/results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    (HERE/'summary.md').write_text('# 065 谱形JS冗余与效应一致性诊断\n\n本轮仅描述性诊断，不训练、不选特征、不生成公开预测。\n\n- 输入为064的780维逐被试无标签z-score，69人2760窗。\n- 池化相关与逐被试相关均值在该标准化下是同一层面的冗余证据；不能把两者当独立验证。完整值见 `results/results.json` 和CSV。\n- HC/DEP效应及LOSO方向一致率只用于判断下一假设，不能证明机制、显著性或泛化。若后续用结果挑选特征，必须另立冻结协议并做独立重复。\n\n参考：Bakdash & Marusich 2017（重复测量相关）；Kriegeskorte等2009（循环分析边界）。\n',encoding='utf8')
    print(json.dumps(out['correlation'],indent=2)); print(f'elapsed={prov["elapsed_seconds"]:.2f}s')
if __name__=='__main__': main()
