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
spec = importlib.util.spec_from_file_location('exp067', HERE.parent / '067_adaptation_context_size' / 'run_experiment.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

METHODS = ('hgb', 'logistic_regression', 'linear_svc')

def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

def make_model(name, config):
    if name == 'hgb':
        model = HistGradientBoostingClassifier(**config['hgb'], random_state=config['random_state'])
    elif name == 'logistic_regression':
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=config['random_state'], solver='lbfgs')
    else:
        model = LinearSVC(C=1.0, random_state=config['random_state'], dual='auto', max_iter=5000)
    return make_pipeline(StandardScaler(), model)

def predict_probability(model, x, method):
    if method == 'linear_svc':
        return sigmoid(model.decision_function(x))
    return model.predict_proba(x)[:, 1]

def classification(y, p):
    return {'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'f1': float(f1_score(y, p >= 0.5, zero_division=0))}

def load_data(config):
    raw, ref, _ = base.load_data(config)
    # base.load_data validates cache alignment, dimensions, HC43 exclusion, 69x40, and no public arrays.
    return raw, ref

def trial_keys(ref):
    return np.asarray([f'{y}::{t}' for y, t in zip(ref.y, ref.trial)], dtype=object)

def choose_method(x, y, keys, train_keys, config):
    scores = {}
    unique = list(dict.fromkeys(train_keys))
    for method in METHODS:
        inner_p, inner_y = [], []
        for held in unique:
            tr = np.isin(keys, unique) & (keys != held)
            va = keys == held
            model = make_model(method, config)
            model.fit(x[tr], y[tr])
            inner_p.extend(predict_probability(model, x[va], method)); inner_y.extend(y[va])
        metric = classification(np.asarray(inner_y), np.asarray(inner_p))
        scores[method] = metric
    best = max(METHODS, key=lambda m: ((-np.inf if scores[m]['auc'] is None else scores[m]['auc']), scores[m]['f1'], -METHODS.index(m)))
    return best, scores

def subject_run(subject, x, ref, config):
    ix = ref.subject.to_numpy() == subject
    sx, sr = x[ix], ref.loc[ix].reset_index(drop=True)
    y, keys = sr.y.to_numpy(), trial_keys(sr)
    unique = list(dict.fromkeys(keys))
    assert len(unique) == 8 and all(np.sum(keys == k) == 5 for k in unique)
    p = np.full(40, np.nan); folds = np.full(40, -1, dtype=int); selected = []
    fold_meta = []
    for fold, held in enumerate(unique):
        train = keys != held; valid = keys == held
        method, inner = choose_method(sx, y, keys, keys[train], config)
        model = make_model(method, config); model.fit(sx[train], y[train])
        p[valid] = predict_probability(model, sx[valid], method); folds[valid] = fold
        selected.append(method)
        fold_meta.append({'subject': str(subject), 'fold': fold, 'held_out_trial_key': str(held), 'train_rows': int(train.sum()), 'valid_rows': int(valid.sum()), 'selected_method': method, 'inner_scores': inner})
    assert np.isfinite(p).all() and np.array_equal(np.sort(folds), np.repeat(np.arange(8), 5))
    metric = classification(y, p)
    return sr, p, folds, metric, selected, fold_meta

def main():
    start = time.perf_counter(); config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    (HERE / 'predictions').mkdir(exist_ok=True); (HERE / 'input_metadata').mkdir(exist_ok=True)
    x, ref = load_data(config)
    subjects = ref.subject.to_numpy(); all_rows = []; subject_rows = []; fold_rows = []; selection_rows = []
    with threadpool_limits(limits=1):
        for subject in np.unique(subjects):
            sr, p, folds, metric, selected, meta = subject_run(subject, x, ref, config)
            out = sr.copy(); out['fold'] = folds; out['prob_positive'] = p; out['subject'] = str(subject)
            all_rows.append(out); subject_rows.append({'subject': str(subject), 'group': str(sr.group.iloc[0]), **metric, **{m + '_count': selected.count(m) for m in METHODS}})
            fold_rows.extend(meta)
            selection_rows.extend([{'subject': str(subject), 'fold': i, 'selected_method': m} for i, m in enumerate(selected)])
    pred = pd.concat(all_rows, ignore_index=True)
    pred = pred[['row_index','subject','group','trial','position','y','fold','prob_positive']]
    pred.to_csv(HERE / 'predictions/within_subject_oof.csv', index=False)
    subject_df = pd.DataFrame(subject_rows); subject_df.to_csv(HERE / 'per_subject.csv', index=False)
    overall = classification(pred.y.to_numpy(), pred.prob_positive.to_numpy())
    macro = {'auc': float(subject_df.auc.mean()), 'f1': float(subject_df.f1.mean())}
    counts = {m: int((subject_df[m + '_count']).sum()) for m in METHODS}
    summary = pd.DataFrame([{'scope':'pooled_overall', **overall}, {'scope':'macro_subject', **macro}]); summary.to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(HERE / 'input_metadata/folds.csv', index=False)
    pd.DataFrame(selection_rows).to_csv(HERE / 'input_metadata/method_selection.csv', index=False)
    inputs = [(HERE / config[k]).resolve() for k in ('spectral_cache','temporal_cache','pe_cache')]
    sources = [HERE / n for n in ('run_experiment.py','test_091.py','config.json','test_protocol.txt')] + [Path(base.__file__), ROOT / '数据/数据集说明文档.md']
    prov = {'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, 'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, 'raw_shape': list(x.shape), 'public_arrays_loaded': False, 'n_subjects': int(ref.subject.nunique()), 'rows': int(len(ref)), 'normalization': config['normalization'], 'threads': 1, 'python': platform.python_version(), 'sklearn': __import__('sklearn').__version__}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    metrics = {'overall_pooled': overall, 'macro_subject': macro, 'subject_metrics': subject_rows, 'method_selection_counts': counts, 'folds': fold_rows, 'elapsed_seconds': time.perf_counter()-start, 'provenance': prov}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    print(json.dumps({'overall_pooled': overall, 'macro_subject': macro, 'method_selection_counts': counts, 'elapsed_seconds': metrics['elapsed_seconds']}, ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
