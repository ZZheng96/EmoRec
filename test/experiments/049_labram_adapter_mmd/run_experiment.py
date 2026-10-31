import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from emo_rec.normalization import normalize_by_subject

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = HERE / 'cache'
PRED = HERE / 'predictions'
EXP004 = ROOT / 'test/experiments/004_labram_frozen'
EXP001 = ROOT / 'test/experiments/001_spectral_logpower'
BASELINE = ROOT / 'test/experiments/002_subject_normalization'


def folds(subjects, groups, seed=2026):
    from sklearn.model_selection import StratifiedKFold
    names = np.unique(subjects)
    health = np.array([groups[np.flatnonzero(subjects == s)[0]] for s in names])
    out = {}
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(names, health)):
        for i in va: out[names[i]] = k
    return np.array([out[s] for s in subjects])


def classification_metrics(y, p, threshold=.5):
    q = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, q, labels=[0, 1]).ravel()
    return {'roc_auc': float(roc_auc_score(y, p)), 'binary_f1': float(f1_score(y, q, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, q)), 'threshold': threshold,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def load_data():
    with np.load(EXP004 / 'cache/embeddings.npz') as z: lab = {k: z[k] for k in z.files}
    with np.load(EXP001 / 'cache/features.npz') as z: spec = {k: z[k] for k in z.files}
    keep = lab['subject_ids'] != 'HC43'
    keys = ['X_train','y_train','subject_ids','health_groups','folds']
    data = {k: lab[k][keep] for k in keys}
    data['X_public'], data['public_user_ids'], data['public_trial_ids'] = lab['X_public'], lab['public_user_ids'], lab['public_trial_ids']
    sk = spec['subject_ids'] != 'HC43'
    spectral_mask = np.asarray([str(n).startswith('log_bandpower_') for n in spec['feature_names']])
    if spectral_mask.sum() != 150: raise RuntimeError(f'Expected 150 spectral features, got {spectral_mask.sum()}')
    data['S_train'] = normalize_by_subject(spec['X_train'][sk][:, spectral_mask], data['subject_ids'], 'zscore', 1e-12)
    data['S_public'] = normalize_by_subject(spec['X_public'][:, spectral_mask], data['public_user_ids'], 'zscore', 1e-12)
    return data


def public_frame(ids, trials, p):
    return pd.DataFrame({'user_id': ids, 'trial_id': trials, 'prob. of positive emotion': p})


def train_adapter(Xtr, ytr, Xva, Xpub, subjects, train_mask, seed, mmd=False):
    torch.manual_seed(seed); np.random.seed(seed)
    scaler = StandardScaler().fit(Xtr[train_mask])
    a = torch.tensor(scaler.transform(Xtr), dtype=torch.float32)
    av = torch.tensor(scaler.transform(Xva), dtype=torch.float32)
    ap = torch.tensor(scaler.transform(Xpub), dtype=torch.float32)
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.adapter = torch.nn.Sequential(torch.nn.Linear(200, 32), torch.nn.Tanh(), torch.nn.Linear(32, 200)); self.head = torch.nn.Linear(200, 1)
        def forward(self, x): return self.head(x + self.adapter(x)).squeeze(1)
    net = Net(); net.train(); opt = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=1e-4)
    idx = np.flatnonzero(train_mask); yy = torch.tensor(ytr[idx], dtype=torch.float32)
    # fixed subject-domain batches; MMD is label-free and restricted to training subjects
    domain = np.array([subjects[i] for i in idx])
    rng = np.random.default_rng(seed)
    for _ in range(25):
        order = rng.permutation(len(idx))
        for st in range(0, len(idx), 128):
            take = order[st:st+128]; xb, yb = a[idx[take]], yy[take]
            logits = net(xb); loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            if mmd:
                chosen = idx[take]
                ds = domain[take]
                uniq = np.unique(ds)
                if len(uniq) > 1:
                    means = torch.stack([net.adapter(a[chosen[ds == u]]).mean(0) for u in uniq])
                    loss = loss + .05 * ((means - means.mean(0)).pow(2).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pv = torch.sigmoid(net(av)).numpy(); pp = torch.sigmoid(net(ap)).numpy()
    frozen_ok = True
    return pv, pp, {'adapter_parameters': sum(p.numel() for p in net.adapter.parameters()), 'head_parameters': sum(p.numel() for p in net.head.parameters()), 'backbone_frozen': frozen_ok}


def subject_result(y, p, subjects, groups, folds_, name):
    rows=[]
    for s in np.unique(subjects):
        m=subjects==s; rows.append({'run_name':name,'subject_id':s,'health_group':groups[m][0],**classification_metrics(y[m],p[m])})
    sm=pd.DataFrame(rows); hg={}
    for g in ('HC','DEP'):
        m=groups==g; sub=sm[sm.health_group==g]
        hg[g]={'n_subjects':len(sub),'pooled':classification_metrics(y[m],p[m]),'subject_auc_mean':float(sub.roc_auc.mean())}
    fm=[{'fold':k,**classification_metrics(y[folds_==k],p[folds_==k])} for k in range(5)]
    return {'n_subjects':int(np.unique(subjects).size),'n_samples':len(y),'overall':classification_metrics(y,p),'folds':fm,
            'fold_auc_mean':float(np.mean([x['roc_auc'] for x in fm])),'subject_auc_mean':float(sm.roc_auc.mean()),'health_groups':hg,'subject_metrics':rows}


def run_condition(name, data, fold_arr, seed):
    y, s, g = data['y_train'], data['subject_ids'], data['health_groups']; p=np.full(len(y), np.nan)
    pp=np.zeros(len(data['X_public']))
    audits=[]
    for k in range(5):
        tr=fold_arr != k; va=fold_arr == k
        if name == 'spectral_hgb_control':
            sc=StandardScaler().fit(data['S_train'][tr]); model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1.,random_state=seed)
            model.fit(sc.transform(data['S_train'][tr]),y[tr]); p[va]=model.predict_proba(sc.transform(data['S_train'][va]))[:,1]
            if k == 0: pp=model.predict_proba(sc.transform(data['S_public']))[:,1]
        elif name == 'frozen_labram_linear':
            sc=StandardScaler().fit(data['X_train'][tr]); model=torch.nn.Linear(200, 1)
            clf=torch.optim.LBFGS(model.parameters(), max_iter=100); xt=torch.tensor(sc.transform(data['X_train'][tr]),dtype=torch.float32); yt=torch.tensor(y[tr],dtype=torch.float32)
            def closure():
                clf.zero_grad(); loss=torch.nn.functional.binary_cross_entropy_with_logits(model(xt).squeeze(1),yt); loss.backward(); return loss
            clf.step(closure); model.eval()
            with torch.no_grad(): p[va]=torch.sigmoid(model(torch.tensor(sc.transform(data['X_train'][va]),dtype=torch.float32))).squeeze(1).numpy(); pub=model(torch.tensor(sc.transform(data['X_public']),dtype=torch.float32)).sigmoid().squeeze(1).numpy()
            pp=pub if k==0 else pp; audits.append({'backbone_frozen':True,'classifier':'linear'})
        else:
            p[va], pub, audit=train_adapter(data['X_train'],y,data['X_train'][va],data['X_public'],s,tr,seed+k,name=='adapter_mmd'); pp=pub if k==0 else pp; audits.append(audit)
        print(f'{name} seed={seed} fold={k+1}/5',flush=True)
    # final public-only fit, using all labels
    if name == 'spectral_hgb_control':
        sc=StandardScaler().fit(data['S_train']); model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1.,random_state=seed); model.fit(sc.transform(data['S_train']),y); pp=model.predict_proba(sc.transform(data['S_public']))[:,1]
    else:
        _, pp, _=train_adapter(data['X_train'],y,data['X_train'][:1],data['X_public'],s,np.ones(len(y),bool),seed+99,name=='adapter_mmd')
    oof=pd.DataFrame({'subject_id':s,'health_group':g,'fold':fold_arr,'y_true':y,'prob_positive':p}); oof.to_csv(PRED/f'{name}_seed{seed}_oof.csv',index=False)
    public_frame(data['public_user_ids'],data['public_trial_ids'],pp).to_csv(PRED/f'{name}_seed{seed}_public.csv',index=False)
    result=subject_result(y,p,s,g,fold_arr,name); result['audit']=audits[0] if audits else {'backbone_frozen':True}; return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repeats',action='store_true'); args=ap.parse_args(); start=time.perf_counter(); PRED.mkdir(exist_ok=True)
    config=json.loads((HERE/'config.json').read_text(encoding='utf8')); data=load_data(); fold_arr=data['folds']; seeds=[2026]+(config['repeat_seeds'] if args.repeats else [])
    allm={'experiment_config':config,'protocol':{'public_labels_used':False,'hc43_excluded':True,'validation_label_independent':True},'runs':{},'runtime':{}}
    for seed in seeds:
        for name in config['conditions']:
            r=run_condition(name,data,fold_arr if seed==2026 else folds(data['subject_ids'],data['health_groups'],seed),seed); allm['runs'][f'{name}_seed{seed}']=r
            (HERE/'metrics.json').write_text(json.dumps(allm,ensure_ascii=False,indent=2),encoding='utf8')
    rows=[]
    for name,r in allm['runs'].items(): rows.append({'run_name':name,'oof_auc':r['overall']['roc_auc'],'oof_f1':r['overall']['binary_f1'],'oof_bacc':r['overall']['balanced_accuracy'],'subject_macro_auc':r['subject_auc_mean'],'hc_auc':r['health_groups']['HC']['pooled']['roc_auc'],'dep_auc':r['health_groups']['DEP']['pooled']['roc_auc']})
    pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False); allm['runtime']={'elapsed_seconds':time.perf_counter()-start}; (HERE/'metrics.json').write_text(json.dumps(allm,ensure_ascii=False,indent=2),encoding='utf8'); print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
