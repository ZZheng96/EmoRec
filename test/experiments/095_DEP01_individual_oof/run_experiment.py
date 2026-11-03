import hashlib, importlib.util, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('exp091', HERE.parent / '091_within_subject_trial_oof' / 'run_experiment.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
CANDIDATES = tuple(f'{m}|{b}|{s}|{d}|c{c}|t{t}' for m in ('linear_svc','logistic_regression','hist_gradient_boosting') for b in ('all','spectral','temporal','permutation_entropy') for s in ('raw','selectkbest100','pca095') for d in ('raw','reverse') for c in (0.1,1.0,10.0) for t in (0.5,))

def load_data(config):
    x, ref, _ = base.base.load_data(config) if hasattr(base, 'base') else (*base.load_data(config), None)
    return x, ref

def trial_keys(ref): return np.asarray([f'{y}::{t}' for y,t in zip(ref.y, ref.trial)], dtype=object)
def normalize_subject(x):
    out=x.astype(float,copy=True)
    for a,b in ((0,150),(150,600),(600,720)):
        z=out[:,a:b]; out[:,a:b]=(z-z.mean(axis=0))/z.std(axis=0)
    assert np.isfinite(out).all()
    return out
def metric(y,p,t): return {'auc': float(roc_auc_score(y,p)),'f1':float(f1_score(y,p>=t,zero_division=0))}
def make_model(name, c):
    if name=='linear_svc': return LinearSVC(C=c, dual='auto', max_iter=5000, random_state=2026)
    if name=='logistic_regression': return LogisticRegression(C=c, max_iter=3000, solver='liblinear', random_state=2026)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=5, l2_regularization=c, early_stopping=False, random_state=2026)
def probability(model,x, direction):
    z=model.decision_function(x) if hasattr(model,'decision_function') else model.predict_proba(x)[:,1]
    z=np.clip(z,-60,60); p=1/(1+np.exp(-z)); return p if direction=='raw' else 1-p
def pipeline(name, block, selector, c):
    steps=[('scale',StandardScaler())]
    if selector=='selectkbest100': steps.append(('select',SelectKBest(f_classif,k=100)))
    elif selector=='pca095': steps.append(('pca',PCA(n_components=0.95,svd_solver='full',random_state=2026)))
    steps.append(('model',make_model(name,c))); return Pipeline(steps)
def block_x(x,b): return x if b=='all' else x[:, {'spectral':slice(0,150),'temporal':slice(150,600),'permutation_entropy':slice(600,720)}[b]]
def choose(x,y,keys,train_keys,config):
    scores=[]
    for i,cand in enumerate(CANDIDATES):
        name,b,s,d,cs,ts=cand.split('|'); threshold=float(ts[1:]); c=float(cs[1:])
        pp=[]; yy=[]; unique=list(dict.fromkeys(train_keys))
        for held in unique:
            tr=(keys!=held)&np.isin(keys,unique); va=keys==held; model=pipeline(name,b,s,c); model.fit(block_x(x,b)[tr],y[tr]); pp.extend(probability(model,block_x(x,b)[va],d)); yy.extend(y[va])
        m=metric(np.asarray(yy),np.asarray(pp),threshold); scores.append((m['auc'],m['f1'],-i,cand,m))
    best=max(scores); return best[3], {s[3]:s[4] for s in scores}

def main():
    start=time.perf_counter(); config=json.loads((HERE/'config.json').read_text()); x,ref=load_data(config)
    assert ref.subject.nunique()==69 and (ref.groupby('subject').size()==40).all() and not (ref.subject==config['excluded_subject']).any()
    sr=ref[ref.subject==config['target_subject']].reset_index(drop=True); sx=normalize_subject(x[ref.subject.to_numpy()==config['target_subject']]); y=sr.y.to_numpy(); keys=trial_keys(sr); unique=list(dict.fromkeys(keys)); assert len(unique)==8 and all((keys==k).sum()==5 for k in unique)
    p=np.full(40,np.nan); folds=np.full(40,-1); metadata=[]
    with threadpool_limits(limits=1):
        for fold,held in enumerate(unique):
            tr=keys!=held; va=~tr; selected,inner=choose(sx,y,keys,keys[tr],config); name,b,s,d,cs,ts=selected.split('|'); model=pipeline(name,b,s,float(cs[1:])); model.fit(block_x(sx,b)[tr],y[tr]); p[va]=probability(model,block_x(sx,b)[va],d); folds[va]=fold; metadata.append({'fold':fold,'held_out_trial_key':held,'selected_candidate':selected,'inner_scores':inner})
    out=sr.assign(fold=folds,prob_positive=p); out.to_csv(HERE/'predictions/DEP01_oof.csv',index=False); m=metric(y,p,.5); pd.DataFrame(metadata).to_json(HERE/'input_metadata/fold_selection.json',orient='records',indent=2); (HERE/'metrics.json').write_text(json.dumps({'subject':'DEP01','auc':m['auc'],'f1':m['f1'],'simultaneously_reached_0.95':m['auc']>=.95 and m['f1']>=.95,'elapsed_seconds':time.perf_counter()-start,'n_rows':40},indent=2)); print(json.dumps(m,ensure_ascii=False))
if __name__=='__main__': main()
