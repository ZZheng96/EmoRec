import hashlib, importlib.util, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('e096', HERE.parent/'096_DEP01_balanced_pairs'/'run_experiment.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
CANDIDATES = tuple((b,c,g) for b in ('all','spectral','temporal','PE') for c in (.1,1,10,100) for g in ('scale',.01))
metric = base.metric

def load_subject(config):
    assert config['target_subject'] == 'DEP01'
    x, ref = base.load_data(config)
    mask = ref.subject.to_numpy() == 'DEP01'
    ref = ref[mask].reset_index(drop=True)
    return x[mask], ref

def splits(y, keys):
    pos = list(dict.fromkeys(keys[y == 1])); neg = list(dict.fromkeys(keys[y == 0]))
    for p in pos:
        for n in neg:
            valid = np.isin(keys, [p,n])
            yield ~valid, valid

def aggregate(values, masks, expected_count):
    masks = np.asarray(masks)
    counts = masks.sum(axis=0)
    assert np.all(counts == expected_count)
    if expected_count == 4:
        return base.aggregate_pair_predictions(values, masks)
    sums = np.zeros(masks.shape[1])
    for value, mask in zip(values, masks): sums[mask] += value
    return sums/counts, counts

def fit_predict(x_train, y_train, x_valid, candidate):
    block, c, gamma = candidate
    model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=c, gamma=gamma))
    model.fit(base.block_x(x_train, block), y_train)
    score = model.decision_function(base.block_x(x_valid, block))
    return 1/(1+np.exp(-np.clip(score, -60, 60)))

def select_candidate(x_train, y_train, train_keys):
    inner = list(splits(y_train, train_keys))
    records = []; predictions = []
    for index, candidate in enumerate(CANDIDATES):
        values = [fit_predict(x_train[tr], y_train[tr], x_train[va], candidate) for tr,va in inner]
        p, counts = aggregate(values, [va for tr,va in inner], 3)
        records.append({'candidate_index':index, 'candidate':candidate, **metric(y_train,p)})
        predictions.append(p)
    best = max(records, key=lambda r:(r['auc'], r['f1'], -r['candidate_index']))
    return CANDIDATES[best['candidate_index']], records, predictions

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    start = time.perf_counter()
    config = json.loads((HERE/'config.json').read_text())
    x, ref = load_subject(config); y = ref.y.to_numpy(); keys = base.trial_keys(ref)
    assert x.shape == (40,720) and len(set(keys)) == 8
    for name in ('predictions','input_metadata'): (HERE/name).mkdir(exist_ok=True)
    source_files = [HERE/'run_experiment.py', HERE/'test_097.py', HERE/'config.json', HERE/'冻结协议.md', ROOT/'pyproject.toml', ROOT/'数据/数据集说明文档.md']
    source_files += [Path(module.__file__) for module in (base, base.base, base.base.base, base.base.base.base, base.base.base.base.base)]
    provenance = {'sources':{str(p):sha(p) for p in source_files},
        'inputs':{str((HERE/config[k]).resolve()):sha((HERE/config[k]).resolve()) for k in ('spectral_cache','temporal_cache','pe_cache','fixed_reference')},
        'python':platform.python_version(), 'numpy':np.__version__, 'pandas':pd.__version__, 'sklearn':sklearn.__version__, 'threads':1,
        'normalization':config['normalization'], 'compute_reason':config['compute_reason']}
    (HERE/'input_metadata/provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    values=[]; masks=[]; selections=[]; long=[]; inner_long=[]; pair_metrics=[]
    with threadpool_limits(limits=1):
        for pair_id,(tr,va) in enumerate(splits(y,keys)):
            # The selector receives only outer-training arrays, never outer labels.
            candidate, scores, inner_p = select_candidate(x[tr], y[tr], keys[tr])
            indices = np.flatnonzero(tr)
            inner_indices = [{'train_indices':indices[it].tolist(), 'valid_indices':indices[iv].tolist()} for it,iv in splits(y[tr],keys[tr])]
            selections.append({'pair_id':pair_id, 'train_indices':indices.tolist(), 'valid_indices':np.flatnonzero(va).tolist(), 'inner_splits':inner_indices, 'selected_candidate':candidate, 'inner_scores':scores})
            for ci, pp in enumerate(inner_p):
                for j,p in zip(indices,pp): inner_long.append({'pair_id':pair_id,'candidate_index':ci,'local_index':int(j),'row_index':int(ref.iloc[j].row_index),'y':int(y[j]),'prob_positive':float(p),'prediction_count':3})
            p = fit_predict(x[tr], y[tr], x[va], candidate)
            values.append(p); masks.append(va)
            pair_metrics.append({'pair_id':pair_id, **metric(y[va],p)})
            for j,prob in zip(np.flatnonzero(va),p): long.append({'pair_id':pair_id,'local_index':int(j),'row_index':int(ref.iloc[j].row_index),'y':int(y[j]),'prob_positive':float(prob)})
            print(json.dumps({'pair_id':pair_id,'selected_candidate':candidate,'inner_best':max(scores,key=lambda r:(r['auc'],r['f1'],-r['candidate_index']))}),flush=True)
    p, counts = aggregate(values,masks,4)
    ref.assign(prob_positive=p,prediction_count=counts).to_csv(HERE/'predictions/DEP01_nonlinear_pairs_oof.csv',index=False)
    pd.DataFrame(long).to_csv(HERE/'predictions/pair_predictions.csv',index=False)
    pd.DataFrame(inner_long).to_csv(HERE/'predictions/inner_mean_predictions.csv',index=False)
    (HERE/'input_metadata/selections.json').write_text(json.dumps(selections,indent=2),encoding='utf8')
    result = {'subject':'DEP01','pooled_oof':metric(y,p),'pair_mean':{k:float(np.mean([m[k] for m in pair_metrics])) for k in ('auc','f1')},'pair_metrics':pair_metrics,'prior_0.5':metric(y,np.full(len(y),.5)), 'n_rows':40,'long_rows':len(long),'prediction_count':4,'candidate_count':len(CANDIDATES),'elapsed_seconds':time.perf_counter()-start}
    (HERE/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result),flush=True)

if __name__ == '__main__': main()
