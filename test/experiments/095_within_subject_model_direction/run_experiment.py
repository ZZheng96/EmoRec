import hashlib, importlib.util, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, f1_score
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('exp091', HERE.parent / '091_within_subject_trial_oof' / 'run_experiment.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
MODELS = ('linear_svc', 'logistic_regression', 'hgb')
DIRECTIONS = ('raw', 'reverse')
CANDIDATES = tuple(f'{m}_{d}' for m in MODELS for d in DIRECTIONS)

def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

def make_model(name, config):
    if name == 'linear_svc': model = LinearSVC(**config['linear_svc'], random_state=config['random_state'])
    elif name == 'logistic_regression': model = LogisticRegression(**config['logistic_regression'], random_state=config['random_state'])
    else: model = HistGradientBoostingClassifier(**config['hgb'], random_state=config['random_state'])
    return make_pipeline(StandardScaler(), model)

def score(model, x, name):
    m, direction = name.rsplit('_', 1)
    p = sigmoid(model.decision_function(x)) if m == 'linear_svc' else model.predict_proba(x)[:, 1]
    return p if direction == 'raw' else 1.0 - p

def classification(y, p):
    return {'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None, 'f1': float(f1_score(y, p >= 0.5, zero_division=0))}

def load_data(config): return base.load_data(config)
def trial_keys(ref): return np.asarray([f'{y}::{t}' for y, t in zip(ref.y, ref.trial)], dtype=object)

def choose_candidate(x, y, keys, train_keys, config):
    unique = list(dict.fromkeys(train_keys)); scores = {}
    for candidate in CANDIDATES:
        inner_p, inner_y = [], []
        for held in unique:
            tr = np.isin(keys, unique) & (keys != held); va = keys == held
            model = make_model(candidate.rsplit('_', 1)[0], config); model.fit(x[tr], y[tr])
            inner_p.extend(score(model, x[va], candidate)); inner_y.extend(y[va])
        scores[candidate] = classification(np.asarray(inner_y), np.asarray(inner_p))
    best = max(CANDIDATES, key=lambda c: ((-np.inf if scores[c]['auc'] is None else scores[c]['auc']), scores[c]['f1'], -CANDIDATES.index(c)))
    return best, scores

def subject_run(subject, x, ref, config):
    ix = ref.subject.to_numpy() == subject; sx, sr = x[ix], ref.loc[ix].reset_index(drop=True)
    y, keys = sr.y.to_numpy(), trial_keys(sr); unique = list(dict.fromkeys(keys))
    assert len(unique) == 8 and all(np.sum(keys == k) == 5 for k in unique)
    adaptive = np.full(40, np.nan); fixed = np.full(40, np.nan); folds = np.full(40, -1, dtype=int); selected = []; meta = []
    for fold, held in enumerate(unique):
        train, valid = keys != held, keys == held
        candidate, inner = choose_candidate(sx, y, keys, keys[train], config)
        model = make_model(candidate.rsplit('_', 1)[0], config); model.fit(sx[train], y[train])
        fixed_model = make_model('linear_svc', config); fixed_model.fit(sx[train], y[train])
        adaptive[valid] = score(model, sx[valid], candidate); fixed[valid] = score(fixed_model, sx[valid], 'linear_svc_raw'); folds[valid] = fold; selected.append(candidate)
        meta.append({'subject': str(subject), 'fold': fold, 'held_out_trial_key': str(held), 'train_rows': int(train.sum()), 'valid_rows': int(valid.sum()), 'selected_candidate': candidate, 'inner_scores': inner})
    assert np.isfinite(adaptive).all() and np.isfinite(fixed).all()
    return sr, fixed, adaptive, folds, selected, meta

def main():
    start = time.perf_counter(); config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    (HERE / 'predictions').mkdir(exist_ok=True); (HERE / 'input_metadata').mkdir(exist_ok=True)
    x, ref = load_data(config); all_rows=[]; subject_rows=[]; fold_rows=[]; selection_rows=[]
    with threadpool_limits(limits=1):
        for subject in np.unique(ref.subject.to_numpy()):
            sr, fixed, adaptive, folds, selected, meta = subject_run(subject, x, ref, config)
            out=sr.copy(); out['fold']=folds; out['prob_linear_svc_raw']=fixed; out['prob_adaptive']=adaptive; out['subject']=str(subject); all_rows.append(out)
            fm, am = classification(sr.y.to_numpy(), fixed), classification(sr.y.to_numpy(), adaptive)
            row={'subject':str(subject),'group':str(sr.group.iloc[0]),'linear_svc_raw_auc':fm['auc'],'linear_svc_raw_f1':fm['f1'],'adaptive_auc':am['auc'],'adaptive_f1':am['f1']}
            row.update({c+'_count': selected.count(c) for c in CANDIDATES}); subject_rows.append(row); fold_rows.extend(meta); selection_rows.extend({'subject':str(subject),'fold':i,'selected_candidate':c} for i,c in enumerate(selected))
    pred=pd.concat(all_rows,ignore_index=True)[['row_index','subject','group','trial','position','y','fold','prob_linear_svc_raw','prob_adaptive']]; pred.to_csv(HERE/'predictions/within_subject_oof.csv',index=False)
    sdf=pd.DataFrame(subject_rows); sdf.to_csv(HERE/'per_subject.csv',index=False)
    raw=classification(pred.y.to_numpy(),pred.prob_linear_svc_raw.to_numpy()); adaptive=classification(pred.y.to_numpy(),pred.prob_adaptive.to_numpy())
    macro_raw={'auc':float(sdf.linear_svc_raw_auc.mean()),'f1':float(sdf.linear_svc_raw_f1.mean())}; macro_adaptive={'auc':float(sdf.adaptive_auc.mean()),'f1':float(sdf.adaptive_f1.mean())}
    counts={c:int(sdf[c+'_count'].sum()) for c in CANDIDATES}
    pd.DataFrame([{'arm':'linear_svc_raw','scope':'pooled_overall',**raw},{'arm':'adaptive','scope':'pooled_overall',**adaptive},{'arm':'linear_svc_raw','scope':'macro_subject',**macro_raw},{'arm':'adaptive','scope':'macro_subject',**macro_adaptive}]).to_csv(HERE/'summary.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(HERE/'input_metadata/folds.csv',index=False); pd.DataFrame(selection_rows).to_csv(HERE/'input_metadata/candidate_selection.csv',index=False)
    inputs=[(HERE/config[k]).resolve() for k in ('spectral_cache','temporal_cache','pe_cache')]; sources=[HERE/n for n in ('run_experiment.py','test_095.py','config.json','test_protocol.txt')]+[Path(base.__file__),ROOT/'数据/数据集说明文档.md']
    prov={'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'raw_shape':list(x.shape),'public_arrays_loaded':False,'n_subjects':int(ref.subject.nunique()),'rows':int(len(ref)),'normalization':config['normalization'],'threads':1,'python':platform.python_version(),'sklearn':__import__('sklearn').__version__}
    (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8')
    metrics={'linear_svc_raw':{'overall_pooled':raw,'macro_subject':macro_raw},'adaptive':{'overall_pooled':adaptive,'macro_subject':macro_adaptive},'candidate_selection_counts':counts,'subject_metrics':subject_rows,'folds':fold_rows,'elapsed_seconds':time.perf_counter()-start,'provenance':prov}; (HERE/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    print(json.dumps({'linear_svc_raw':metrics['linear_svc_raw'],'adaptive':metrics['adaptive'],'candidate_selection_counts':counts,'elapsed_seconds':metrics['elapsed_seconds']},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
