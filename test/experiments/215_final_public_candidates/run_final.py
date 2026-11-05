import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'predictions'
SRC=Path(r'D:\BCIC2026\EmoRec')
base=pd.read_csv(SRC/'test/experiments/057_multiscale_permutation_entropy/predictions/plus_pe_public.csv')
stft=pd.read_csv(SRC/'test/experiments/213_public_submission_candidates/predictions/repeated_seed_mean_residual_0.2_public.csv')
cons=pd.read_csv(SRC/'test/experiments/037_trial_consistency_robustness/predictions/consistency_spectral_seed2027_public.csv')
key=['user_id','trial_id']
for x in (base,stft,cons):
    assert list(x.columns)==key+['prob. of positive emotion']
    assert len(x)==80 and not x.duplicated(key).any()
base=base.merge(stft,on=key,suffixes=('_base','_stft')).merge(cons,on=key)
pb=base['prob. of positive emotion_base'].to_numpy(float)
ps=base['prob. of positive emotion_stft'].to_numpy(float)
pc=base['prob. of positive emotion'].to_numpy(float)

def write(name,p):
    out=base[key].copy(); out['prob. of positive emotion']=np.clip(p,0,1)
    out.to_csv(OUT/name,index=False)
    assert len(out)==80 and np.isfinite(out.iloc[:,-1]).all()
    return {'file':name,'rows':len(out),'sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest(),'min':float(out.iloc[:,-1].min()),'max':float(out.iloc[:,-1].max())}
results=[]
results.append(write('candidate_1_consistency_spectral.csv',pc))
results.append(write('candidate_2_baseline_plus_pe.csv',pb))
results.append(write('candidate_3_stft_residual.csv',ps))
results.append(write('candidate_4_consistency_stft_blend_0.8_0.2.csv',0.8*pc+0.2*ps))
# Fixed-quota auxiliary candidate: preserve score ordering with tiny within-subject values.
q=np.empty_like(pc)
for user,g in base.groupby('user_id',sort=False):
    idx=g.index.to_numpy(); order=idx[np.argsort(-pc[idx],kind='mergesort')]
    q[order]=np.array([0.75+0.05*(3-i) for i in range(4)]+[0.25-0.05*i for i in range(4)])
results.append(write('candidate_5_fixed_quota_consistency.csv',q))
(HERE/'metrics.json').write_text(json.dumps({'public_labels_used':False,'selection':'predeclared candidates; no public score used','candidates':results},ensure_ascii=False,indent=2),encoding='utf-8')
(HERE/'实验报告.md').write_text('''# 实验215：初赛最后一轮公开候选\n\n根据初赛只提交公开测试集预测、评价AUC和F1的要求，整理5个预声明候选。候选1使用实验037 consistency_spectral公开预测；该方向严格OOF均值AUC=0.833320、F1=0.749930，高于当前057和STFT残差。候选4为固定0.8×consistency_spectral+0.2×STFT残差。候选5使用公开集已知每个被试4正4中的固定quota辅助策略，仅作为最后一次F1风险候选。公开标签未读取，未使用平台反馈。\n\n## 候选\n\n1. `candidate_1_consistency_spectral.csv`：首选AUC/F1候选。\n2. `candidate_2_baseline_plus_pe.csv`：稳健基线。\n3. `candidate_3_stft_residual.csv`：STFT残差候选。\n4. `candidate_4_consistency_stft_blend_0.8_0.2.csv`：固定融合候选。\n5. `candidate_5_fixed_quota_consistency.csv`：每个公开被试固定4高分/4低分的F1辅助候选。\n\n每个文件80行，列名和公开提交顺序与赛题要求一致；AUC/F1真实公开成绩未知。\n''',encoding='utf-8')
print(pd.DataFrame(results).to_string(index=False))
