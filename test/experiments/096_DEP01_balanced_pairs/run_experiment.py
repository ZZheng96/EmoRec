import hashlib, importlib.util, json, platform, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

sys_path = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e095', sys_path.parent/'095_DEP01_individual_oof'/'run_experiment.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
HERE = sys_path
ROOT = HERE.parents[2]
BLOCKS = ('all','spectral','temporal','PE')
MODELS = ('linear_svc','logistic_regression')
CS = (0.01,0.1,1.0,10.0)
CANDIDATES = tuple((m,b,c) for m in MODELS for b in BLOCKS for c in CS)

def load_data(config): return base.load_data(config)
def trial_keys(ref): return np.asarray([f'{int(y)}::{t}' for y,t in zip(ref.y,ref.trial)], dtype=object)
def block_x(x,b): return x if b=='all' else x[:, {'spectral':slice(0,150),'temporal':slice(150,600),'PE':slice(600,720)}[b]]
def metric(y,p): return {'auc':float(roc_auc_score(y,p)), 'f1':float(f1_score(y,p>=.5,zero_division=0))}
def fit_predict(x,y,train,valid,cand):
    m,b,c=cand; est = LinearSVC(C=c, dual='auto', max_iter=5000, random_state=2026) if m=='linear_svc' else LogisticRegression(C=c,solver='liblinear',max_iter=3000,random_state=2026)
    pipe=make_pipeline(StandardScaler(),est); z=block_x(x,b); pipe.fit(z[train],y[train]); s=pipe.decision_function(z[valid]); s=np.clip(s,-60,60); return 1/(1+np.exp(-s))
def aggregate_pair_predictions(pair_predictions, pair_masks):
    sums=np.zeros(pair_masks.shape[1],dtype=float); counts=np.zeros(pair_masks.shape[1],dtype=int)
    for values,mask in zip(pair_predictions,pair_masks): sums[mask]+=values; counts[mask]+=1
    assert np.all(counts==4)
    return sums/counts, counts
def inner_score(x,y,keys,pos,neg,cand):
    pairs=[(p,n) for p in pos for n in neg]; predictions=[]; masks=[]
    for p,n in pairs:
        va=np.isin(keys,[p,n]); predictions.append(fit_predict(x,y,~va,va,cand)); masks.append(va)
    masks=np.asarray(masks); counts=masks.sum(axis=0); assert np.all(counts==3)
    sums=np.zeros(len(keys),dtype=float)
    for values,mask in zip(predictions,masks): sums[mask]+=values
    p=sums/counts; return metric(y,p),p

def main():
    start=time.perf_counter(); config=json.loads((HERE/'config.json').read_text()); x,ref=load_data(config)
    sr=ref[ref.subject==config['target_subject']].reset_index(drop=True); x=x[ref.subject.to_numpy()==config['target_subject']]; y=sr.y.to_numpy(); keys=trial_keys(sr)
    pos=list(dict.fromkeys(keys[y==1])); neg=list(dict.fromkeys(keys[y==0])); assert len(pos)==len(neg)==4
    pair_predictions=[]; pair_masks=[]; fold_rows=[]; overlap=[]
    with threadpool_limits(limits=1):
      for fold,(hp,hn) in enumerate((p,n) for p in pos for n in neg):
        va=np.isin(keys,[hp,hn]); train=~va; remp=[k for k in pos if k!=hp]; remn=[k for k in neg if k!=hn]
        scores=[]
        for i,cand in enumerate(CANDIDATES):
            im,pred_inner=inner_score(x[train],y[train],keys[train],remp,remn,cand)
            scores.append((im['auc'],im['f1'],-i,cand,im))
        best=max(scores); cand=best[3]; inner_record=[{'candidate':str(s[3]),'auc':s[4]['auc'],'f1':s[4]['f1']} for s in scores]
        pair_predictions.append(fit_predict(x,y,train,va,cand)); pair_masks.append(va)
        fold_rows.append({'fold':fold,'held_positive_trial':hp,'held_negative_trial':hn,'train_rows':int(train.sum()),'valid_rows':int(va.sum()),'train_positive':int(y[train].sum()),'train_negative':int((1-y[train]).sum()),'selected_candidate':str(cand),'inner_scores':inner_record})
        overlap.append({'fold':fold,'train_indices':np.flatnonzero(train).tolist(),'valid_indices':np.flatnonzero(va).tolist()})
    pair_masks=np.asarray(pair_masks); p, prediction_count=aggregate_pair_predictions(pair_predictions,pair_masks)
    assert np.all(prediction_count==4)
    fold_metrics=[]
    for i,r in enumerate(fold_rows):
        va=pair_masks[i]; fold_metrics.append({'fold':i,**metric(y[va],pair_predictions[i])})
    prior_sum=np.zeros(40); reverse_sum=np.zeros(40)
    for i,r in enumerate(fold_rows):
        va=pair_masks[i]; train=~va; prior_sum[va]+=y[train].mean(); reverse_sum[va]+=1-y[train].mean()
    prior=prior_sum/prediction_count; rev=reverse_sum/prediction_count
    long_rows=[]
    for i in range(16):
        indices=np.flatnonzero(pair_masks[i])
        for local,j in enumerate(indices): long_rows.append({'row_index':int(sr.iloc[j].row_index),'fold':i,'y':int(y[j]),'prob':float(pair_predictions[i][local])})
    pd.DataFrame(long_rows).to_csv(HERE/'predictions/fold_predictions.csv',index=False)
    pooled=metric(y,p); final_fold=np.argmax(pair_masks,axis=0)
    out=sr.copy(); out['fold']=final_fold; out['prob_positive']=p; out['prediction_count']=prediction_count; out.to_csv(HERE/'predictions/DEP01_balanced_pairs_oof.csv',index=False)
    (HERE/'input_metadata/folds.json').write_text(json.dumps(fold_rows,indent=2),encoding='utf8'); (HERE/'input_metadata/overlap_indices.json').write_text(json.dumps(overlap,indent=2),encoding='utf8')
    counts={str(r['selected_candidate']):sum(r['selected_candidate']==q['selected_candidate'] for q in fold_rows) for r in fold_rows}
    sources=[HERE/'run_experiment.py',HERE/'config.json',HERE/'test_096.py',ROOT/'数据/数据集说明文档.md']
    provenance={'script_sha256':hashlib.sha256((HERE/'run_experiment.py').read_bytes()).hexdigest(),'inputs':{str((HERE/config[k]).resolve()):hashlib.sha256((HERE/config[k]).resolve().read_bytes()).hexdigest() for k in ('spectral_cache','temporal_cache','pe_cache')},'normalization':'base.load_data raw features; no pre-40-window zscore in 096; StandardScaler fit per training fold','threads':1}
    (HERE/'input_metadata/provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    metrics={'pooled_oof':pooled,'fold_mean':{'auc':float(np.mean([r['auc'] for r in fold_metrics])),'f1':float(np.mean([r['f1'] for r in fold_metrics]))},'fold_metrics':fold_metrics,'train_prior_baseline':metric(y,prior),'reverse_prior_baseline':metric(y,rev),'selection_counts':counts,'prediction_count':4,'n_rows':40,'elapsed_seconds':time.perf_counter()-start,'provenance':provenance}
    (HERE/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8'); print(json.dumps(metrics,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
