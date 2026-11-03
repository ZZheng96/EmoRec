import hashlib, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import h5py, numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
BANDS=((1,4),(4,8),(8,13),(13,30),(30,45)); CHANNELS=("FP1","FP2","F7","F3","FZ","F4","F8","FT7","FC3","FCZ","FC4","FT8","T3","C3","CZ","C4","T4","TP7","CP3","CPZ","CP4","TP8","T5","P3","PZ","P4","T6","O1","OZ","O2")
PAIR=( (0,1),(2,6),(3,5),(7,11),(8,10),(12,16),(13,15),(17,21),(18,20),(22,26),(23,25),(27,29) )
MIRROR=np.arange(30); SIGN=np.ones(30)
for a,b in PAIR: MIRROR[a]=b; MIRROR[b]=a; SIGN[a]=1; SIGN[b]=-1
BLOCKS=("power","asymmetry","all"); MORPHOLOGIES=("mean","full"); COMPONENTS=(1,2,3); CS=(.1,1,10)
CANDIDATES=tuple((m,b,k,c) for m in MORPHOLOGIES for b in BLOCKS for k in COMPONENTS for c in CS)
FIXED_THRESHOLD=.5; QUOTA=4

def metric(y,p): return {"auc":float(roc_auc_score(y,p)),"f1":float(f1_score(y,np.asarray(p)>=.5,zero_division=0))}
def quota(p):
 o=np.zeros(8,dtype=int); o[np.argsort(-np.asarray(p),kind="stable")[:4]]=1; return o

def load_subject(subject):
 paths=[p for p in (ROOT/"数据"/"训练集").glob("*/*.mat") if p.stem==subject]; assert len(paths)==1
 with h5py.File(paths[0],"r") as mat:
  pieces=[]
  for key in ("EEG_data_neu","EEG_data_pos"):
   d=np.asarray(mat[key][()],dtype=np.float64).T; assert d.shape==(30,50000); pieces.append(d.reshape(30,20,2500).transpose(1,0,2))
 return np.concatenate(pieces),np.r_[np.zeros(20,dtype=int),np.ones(20,dtype=int)],np.r_[np.repeat(np.arange(4),5),4+np.repeat(np.arange(4),5)],paths[0]

def window_features(x):
 powers=[]
 for lo,hi in BANDS:
  z=sosfiltfilt(butter(4,[lo,hi],btype="bandpass",fs=250,output="sos"),x,axis=-1)
  powers.append(np.log(np.maximum(np.var(z,axis=-1),1e-30)))
 power=np.stack(powers,axis=1); asym=power[:,:,] - power[:,:,MIRROR]
 asym=asym*SIGN[None,None,:]
 return power.reshape(len(x),-1),asym.reshape(len(x),-1)

def dynamic(base):
 # base: windows x 150, original order is retained
 t=np.arange(-2,3,dtype=float); mean=base.mean(0); std=base.std(0); slope=((t-t.mean())[:,None]*base).sum(0)/((t-t.mean())**2).sum(); diff=base[-1]-base[0]; halves=base[:2].mean(0)-base[-2:].mean(0)
 return {"mean":mean,"full":np.concatenate([mean,std,slope,diff,halves])}

def trial_features(window_power,window_asym,trial_ids,morph,block):
 rows=[]
 for tid in range(int(np.max(trial_ids)) + 1):
  p,a=dynamic(window_power[trial_ids==tid])[morph],dynamic(window_asym[trial_ids==tid])[morph]
  rows.append({"power":p,"asymmetry":a,"all":np.concatenate([p,a])}[block])
 return np.asarray(rows)

def pairs(y):
 pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
 for p in pos:
  for n in neg:
   va=np.sort([p,n]); yield np.setdiff1d(np.arange(len(y)),va),va

def predict(train,ytr,valid,candidate):
 m,b,k,c=candidate; a=train[(m,b)]; v=valid[(m,b)]
 model=make_pipeline(StandardScaler(),PCA(n_components=k,svd_solver="full"),LogisticRegression(C=c,solver="liblinear",max_iter=3000,random_state=2026)); model.fit(a,ytr); s=model.decision_function(v); return 1/(1+np.exp(-np.clip(s,-60,60)))

def build_trial_sets(power,asym,ids,trial_indices):
 out={}
 for m in MORPHOLOGIES:
  for b in BLOCKS:
   allf=trial_features(power,asym,ids,m,b); out[(m,b)]=allf[trial_indices]
 return out

def select(power,asym,y):
 rec=[]; ids=np.repeat(np.arange(len(y)),5); full=build_trial_sets(power,asym,ids,np.arange(len(y)))
 for ci,cand in enumerate(CANDIDATES):
  vals=[]; ys=[]
  for tr,va in pairs(y):
   train={k:v[tr] for k,v in full.items()}; valid={k:v[va] for k,v in full.items()}
   vals.extend(predict(train,y[tr],valid,cand)); ys.extend(y[va])
  for di,d in enumerate(("raw","reverse")):
   p=np.asarray(vals); p=p if d=="raw" else 1-p; rec.append({"candidate_index":ci,"candidate":cand,"direction":d,"direction_index":di,**metric(np.asarray(ys),p)})
 return sorted(rec,key=lambda r:(-r["auc"],-r["f1"],r["candidate_index"],r["direction_index"]))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 started=time.perf_counter(); x,yw,win_trial,source=load_subject("DEP02"); wp,wa=window_features(x); assert wp.shape==(40,150) and wa.shape==(40,150)
 trial_keys=list(dict.fromkeys([f"{y}::{t%4+1}" for y,t in zip(yw,win_trial)])); ids=np.asarray([trial_keys.index(f"{y}::{t%4+1}") for y,t in zip(yw,win_trial)]); y=np.asarray([int(k.split("::")[0]) for k in trial_keys]); assert y.tolist()==[0,0,0,0,1,1,1,1] and [int((ids==i).sum()) for i in range(8)]==[5]*8
 values=[]; rows=[]; selections=[]
 with threadpool_limits(limits=1):
  for pid,(tr,va) in enumerate(pairs(y)):
   ranked=select(wp[ids*0+np.arange(40)],wa[ids*0+np.arange(40)],y) if False else None
   # inner operates only outer-training windows, with local trial ids
   local_ids=np.asarray([np.flatnonzero(tr==i)[0] if i in tr else -1 for i in ids]); keep=np.isin(ids,tr); lmap={old:j for j,old in enumerate(tr)}; lid=np.asarray([lmap[i] for i in ids[keep]])
   p0,a0=wp[keep],wa[keep]; yy=y[tr]; ranked=select(p0,a0,yy); best=ranked[0]
   test_ids=np.repeat(np.arange(len(va)),5); train_sets=build_trial_sets(p0,a0,lid,np.arange(len(tr))); valid_sets=build_trial_sets(wp[np.isin(ids,va)],wa[np.isin(ids,va)],test_ids, np.arange(len(va))); p=predict(train_sets,yy,valid_sets,tuple(best["candidate"])); p=p if best["direction"]=="raw" else 1-p
   values.append((va,p)); selections.append({"pair_id":pid,"train_trial_indices":tr.tolist(),"valid_trial_indices":va.tolist(),"selected_candidate":best["candidate"],"selected_direction":best["direction"],"inner_scores":ranked,"selection_source":"inner 3x3 OOF only"})
   for i,v in zip(va,p): rows.append({"pair_id":pid,"trial_index":int(i),"trial_key":trial_keys[i],"y":int(y[i]),"prob_positive":float(v)})
 sums=np.zeros(8); counts=np.zeros(8,dtype=int)
 for va,p in values: sums[va]+=p; counts[va]+=1
 p=sums/counts; out=pd.DataFrame({"trial_index":range(8),"trial_key":trial_keys,"y":y,"prob_positive":p,"prediction_count":counts}); out["prediction_quota4"]=quota(p); out["prediction_fixed05"]=(p>=.5).astype(int)
 out.to_csv(HERE/"predictions/DEP02_trial_window_dynamics_oof.csv",index=False); pd.DataFrame(rows).to_csv(HERE/"predictions/pair_predictions.csv",index=False); (HERE/"input_metadata/selections.json").write_text(json.dumps(selections,ensure_ascii=False,indent=2),encoding="utf8")
 freeze={"started_at_utc":datetime.now(timezone.utc).isoformat(),"hashes":{n:sha(HERE/n) for n in ("run_experiment.py","config.json","冻结协议.md")}}; (HERE/"input_metadata/freeze.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding="utf8")
 prov={"source":str(source),"source_sha256":sha(source),"script_sha256":sha(HERE/"run_experiment.py"),"data_description_sha256":sha(ROOT/"数据/数据集说明文档.md"),"python":platform.python_version(),"threads":1,"window_order":"raw 40 windows, 5 per physical trial","outer_validation_labels_used_for_selection":False,"candidate_count":len(CANDIDATES)*2,"feature_dimensions":{"power":150,"asymmetry":150,"all":300},"freeze":freeze}; (HERE/"input_metadata/provenance.json").write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding="utf8")
 pairm=[metric(y[va],pp) for va,pp in values]; result={"experiment_id":"124_DEP02_trial_window_dynamics_pairs","subject":"DEP02","pooled_oof":{"auc":float(roc_auc_score(y,p)),"f1":float(f1_score(y,out.prediction_quota4,zero_division=0))},"fixed_0.5":metric(y,p),"pair_mean":{k:float(np.mean([m[k] for m in pairm])) for k in ("auc","f1")},"pair_metrics":pairm,"n_trials":8,"n_windows":40,"windows_per_trial":5,"candidate_model_count":len(CANDIDATES),"candidate_count":len(CANDIDATES)*2,"morphologies":MORPHOLOGIES,"blocks":BLOCKS,"pca_components":COMPONENTS,"C":CS,"elapsed_seconds":time.perf_counter()-started,"outer_validation_labels_used_for_selection":False}; (HERE/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8"); (HERE/"实验报告.md").write_text(f"# 实验124：DEP02 trial窗口动态特征配对\n\n- pooled AUC：**{result['pooled_oof']['auc']:.6f}**；quota4 F1：**{result['pooled_oof']['f1']:.6f}**\n- fixed0.5 AUC/F1：**{result['fixed_0.5']['auc']:.6f} / {result['fixed_0.5']['f1']:.6f}**\n- pair均值 AUC/F1：**{result['pair_mean']['auc']:.6f} / {result['pair_mean']['f1']:.6f}**\n- 候选：54模型×raw/reverse=108；耗时：**{result['elapsed_seconds']:.3f}秒**。\n",encoding="utf8"); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
