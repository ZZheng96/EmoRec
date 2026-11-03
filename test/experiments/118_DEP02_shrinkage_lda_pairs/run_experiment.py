import hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BLOCKS = ("all", "spectral", "temporal", "PE")
SELECTORS = (("none", None), ("mutual_info", mutual_info_classif), ("f_classif", f_classif))
CANDIDATES = []
for block in BLOCKS:
    for model, values in (("lda", ("auto", 0.1, 0.5, 0.9)), ("logistic_regression", (1.0,))):
        for value in values:
            for selector, _ in SELECTORS:
                for k in ((None,) if selector == "none" else (5, 10, 20, 50)):
                    CANDIDATES.append((block, model, value, selector, k))
CANDIDATES = tuple(CANDIDATES)

def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, np.asarray(p) >= .5, zero_division=0))}

def quota_predictions(p):
    out = np.zeros(8, dtype=int); out[np.argsort(-np.asarray(p), kind="stable")[:4]] = 1; return out

def block_x(x, block):
    return x if block == "all" else x[:, {"spectral": slice(0,150), "temporal": slice(150,600), "PE": slice(600,720)}[block]]

def load_cache(path):
    with np.load(path, allow_pickle=False) as c:
        return {k: c[k] for k in ("X_train", "subject_ids", "health_groups", "original_trial_ids", "y_train", "feature_names")}

def load_subject(config):
    cs = [load_cache((HERE / config[k]).resolve()) for k in ("spectral_cache", "temporal_cache", "pe_cache")]
    for c in cs[1:]:
        for k in ("subject_ids", "health_groups", "original_trial_ids", "y_train"): np.testing.assert_array_equal(cs[0][k], c[k])
    s,t,pe = cs; mask = s["subject_ids"] == config["target_subject"]
    sm = np.asarray([str(n).startswith("log_bandpower_") for n in s["feature_names"]])
    x = np.column_stack([s["X_train"][mask][:,sm], t["X_train"][mask], pe["X_train"][mask]])
    ref = pd.DataFrame({"subject":s["subject_ids"][mask],"group":s["health_groups"][mask],"trial":s["original_trial_ids"][mask],"y":s["y_train"][mask]})
    assert x.shape == (40,720) and ref.groupby(["y","trial"]).size().eq(5).all()
    return x, ref, [str((HERE / config[k]).resolve()) for k in ("spectral_cache","temporal_cache","pe_cache")]

def trial_pairs(y):
    pos,neg = np.flatnonzero(np.asarray(y)==1),np.flatnonzero(np.asarray(y)==0)
    for p in pos:
        for n in neg:
            va=np.asarray([p,n]); yield np.setdiff1d(np.arange(len(y)),va),va

def predict(a, y, b, candidate):
    block, model, value, selector, k = candidate; xa,xb=block_x(a,block),block_x(b,block)
    steps=[("scale",StandardScaler())]
    if selector != "none":
        fn = mutual_info_classif if selector == "mutual_info" else f_classif
        steps.append(("select",SelectKBest(fn,k=k)))
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=value) if model == "lda" else LogisticRegression(C=float(value), max_iter=2000, solver="liblinear", random_state=0)
    steps.append(("model",clf)); pipe=Pipeline(steps); pipe.fit(xa,y)
    return pipe.predict_proba(xb)[:,1]

def select_inner(x,y):
    inner=list(trial_pairs(y)); records=[]
    for idx,c in enumerate(CANDIDATES):
        pp=np.zeros(len(y)); count=np.zeros(len(y),dtype=int)
        for tr,va in inner: pp[va]+=predict(x[tr],y[tr],x[va],c); count[va]+=1
        raw=pp/count
        for di,direction in enumerate(("raw","reverse")):
            p=raw if direction=="raw" else 1-raw
            records.append({"candidate_index":idx,"candidate":c,"direction":direction,"direction_index":di,**metric(y,p)})
    best=min(records,key=lambda r:(-r["auc"],-r["f1"],r["candidate_index"],r["direction_index"]))
    return CANDIDATES[best["candidate_index"]],best["direction"],records

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    started=time.perf_counter(); config=json.loads((HERE/"config.json").read_text(encoding="utf8")); x,ref,inputs=load_subject(config)
    keys=np.asarray([f"{int(a)}::{int(b)}" for a,b in zip(ref.y,ref.trial)],dtype=object); unique=list(dict.fromkeys(keys))
    trial_x=np.asarray([x[keys==k].mean(axis=0) for k in unique]); trial_y=np.asarray([ref.y.to_numpy()[keys==k][0] for k in unique])
    trial_ref=pd.DataFrame([{"trial_index":i,"trial_key":k,"y":int(trial_y[i]),"window_rows":json.dumps(np.flatnonzero(keys==k).tolist())} for i,k in enumerate(unique)])
    values=[]; masks=[]; rows=[]; selections=[]
    with threadpool_limits(limits=1):
        for pair_id,(tr,va) in enumerate(trial_pairs(trial_y)):
            candidate,direction,scores=select_inner(trial_x[tr],trial_y[tr]); p=predict(trial_x[tr],trial_y[tr],trial_x[va],candidate)
            if direction=="reverse": p=1-p
            values.append(p); masks.append(va); selections.append({"pair_id":pair_id,"train_indices":tr.tolist(),"valid_indices":va.tolist(),"selected_candidate":candidate,"selected_direction":direction,"inner_scores":scores})
            rows.extend({"pair_id":pair_id,"trial_index":int(i),"trial_key":unique[i],"y":int(trial_y[i]),"prob_positive":float(q),"window_count":5} for i,q in zip(va,p))
    sums,counts=np.zeros(8),np.zeros(8,dtype=int)
    for value,va in zip(values,masks): sums[va]+=value; counts[va]+=1
    p=sums/counts; trial_ref["prob_positive"]=p; trial_ref["prediction_count"]=counts; trial_ref["prediction_quota4"]=quota_predictions(p); trial_ref["prediction_fixed05"]=(p>=.5).astype(int)
    trial_ref.to_csv(HERE/"predictions/DEP02_trial_shrinkage_lda_oof.csv",index=False); pd.DataFrame(rows).to_csv(HERE/"predictions/pair_predictions.csv",index=False)
    (HERE/"input_metadata/selections.json").write_text(json.dumps(selections,ensure_ascii=False,indent=2),encoding="utf8")
    provenance={"inputs":{q:sha(q) for q in inputs},"sources":{str(HERE/n):sha(HERE/n) for n in ("run_experiment.py","config.json","冻结协议.md")},"data_description_sha256":sha(ROOT/"数据/数据集说明文档.md"),"python":platform.python_version(),"threads":1,"metadata_rows":40,"trial_window_counts":{k:int((keys==k).sum()) for k in unique},"outer_validation_labels_used_for_selection":False}
    (HERE/"input_metadata/provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf8")
    pair_metrics=[metric(trial_y[va],pv) for va,pv in zip(masks,values)]; result={"experiment_id":config["experiment_id"],"subject":"DEP02","pooled_oof":{"auc":float(roc_auc_score(trial_y,p)),"f1":float(f1_score(trial_y,trial_ref.prediction_quota4,zero_division=0))},"fixed_0.5":metric(trial_y,p),"pair_mean":{k:float(np.mean([m[k] for m in pair_metrics])) for k in ("auc","f1")},"pair_metrics":pair_metrics,"n_trials":8,"n_windows":40,"windows_per_trial":5,"feature_dimensions":config["blocks"],"candidate_count":len(CANDIDATES)*2,"positive_quota":4,"outer_validation_labels_used_for_selection":False,"simultaneously_reached_0.95":False,"elapsed_seconds":time.perf_counter()-started}
    (HERE/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    status="达到" if result["pooled_oof"]["auc"]>=.95 and result["pooled_oof"]["f1"]>=.95 else "未达到"
    (HERE/"实验报告.md").write_text(f"# 实验118：DEP02收缩LDA/低维监督特征配对OOF\n\n- 主 pooled AUC：**{result['pooled_oof']['auc']:.6f}**\n- 主F1（预冻结top-4 quota）：**{result['pooled_oof']['f1']:.6f}**\n- fixed 0.5 F1：**{result['fixed_0.5']['f1']:.6f}**；pair均值 AUC/F1：**{result['pair_mean']['auc']:.6f} / {result['pair_mean']['f1']:.6f}**。\n- 候选数：**{len(CANDIDATES)*2}**；结论：双指标0.95要求**{status}**，若任一指标低于0.95不宣称达标。\n",encoding="utf8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
