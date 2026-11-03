import hashlib, importlib.util, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, f1_score
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('exp091', HERE.parent / '091_within_subject_trial_oof' / 'run_experiment.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
METHODS = ('raw', 'reverse')

def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

def make_model(config):
    return make_pipeline(StandardScaler(), LinearSVC(**config['linear_svc'], random_state=config['random_state']))

def score_probability(model, x, direction):
    p = sigmoid(model.decision_function(x))
    return p if direction == 'raw' else 1.0 - p

def classification(y, p):
    return {'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'f1': float(f1_score(y, p >= 0.5, zero_division=0))}

def load_data(config):
    return base.load_data(config)

def trial_keys(ref):
    return base.trial_keys(ref)

def choose_direction(x, y, keys, train_keys, config):
    unique = list(dict.fromkeys(train_keys)); scores = {}
    for direction in METHODS:
        inner_p, inner_y = [], []
        for held in unique:
            tr = np.isin(keys, unique) & (keys != held); va = keys == held
            model = make_model(config); model.fit(x[tr], y[tr])
            inner_p.extend(score_probability(model, x[va], direction)); inner_y.extend(y[va])
        scores[direction] = classification(np.asarray(inner_y), np.asarray(inner_p))
    best = max(METHODS, key=lambda m: ((-np.inf if scores[m]['auc'] is None else scores[m]['auc']), scores[m]['f1'], -METHODS.index(m)))
    return best, scores

def subject_run(subject, x, ref, config):
    ix = ref.subject.to_numpy() == subject; sx, sr = x[ix], ref.loc[ix].reset_index(drop=True)
    y, keys = sr.y.to_numpy(), trial_keys(sr); unique = list(dict.fromkeys(keys))
    assert len(unique) == 8 and all(np.sum(keys == k) == 5 for k in unique)
    raw_p, adaptive_p = np.full(40, np.nan), np.full(40, np.nan); folds = np.full(40, -1, dtype=int)
    fold_meta, selections = [], []
    for fold, held in enumerate(unique):
        train, valid = keys != held, keys == held
        direction, inner = choose_direction(sx, y, keys, keys[train], config)
        model = make_model(config); model.fit(sx[train], y[train])
        raw_p[valid] = score_probability(model, sx[valid], 'raw')
        adaptive_p[valid] = score_probability(model, sx[valid], direction)
        folds[valid] = fold; selections.append(direction)
        fold_meta.append({'subject': str(subject), 'fold': fold, 'held_out_trial_key': str(held), 'train_rows': int(train.sum()), 'valid_rows': int(valid.sum()), 'selected_direction': direction, 'inner_scores': inner})
    assert np.isfinite(raw_p).all() and np.isfinite(adaptive_p).all() and np.array_equal(np.sort(folds), np.repeat(np.arange(8), 5))
    return sr, raw_p, adaptive_p, folds, selections, fold_meta

def main():
    start = time.perf_counter(); config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    (HERE / 'predictions').mkdir(exist_ok=True); (HERE / 'input_metadata').mkdir(exist_ok=True)
    x, ref = load_data(config); subjects = ref.subject.to_numpy(); all_rows = []; subject_rows = []; fold_rows = []; selection_rows = []
    with threadpool_limits(limits=1):
        for subject in np.unique(subjects):
            sr, raw_p, adaptive_p, folds, selections, meta = subject_run(subject, x, ref, config)
            out = sr.copy(); out['fold'] = folds; out['prob_raw'] = raw_p; out['prob_direction_adaptive'] = adaptive_p; out['subject'] = str(subject); all_rows.append(out)
            rm, am = classification(sr.y.to_numpy(), raw_p), classification(sr.y.to_numpy(), adaptive_p)
            subject_rows.append({'subject': str(subject), 'group': str(sr.group.iloc[0]), 'raw_auc': rm['auc'], 'raw_f1': rm['f1'], 'direction_adaptive_auc': am['auc'], 'direction_adaptive_f1': am['f1'], 'reverse_count': selections.count('reverse'), 'raw_count': selections.count('raw')})
            fold_rows.extend(meta); selection_rows.extend([{'subject': str(subject), 'fold': i, 'selected_direction': d} for i, d in enumerate(selections)])
    pred = pd.concat(all_rows, ignore_index=True)[['row_index','subject','group','trial','position','y','fold','prob_raw','prob_direction_adaptive']]
    pred.to_csv(HERE / 'predictions/within_subject_oof.csv', index=False)
    subject_df = pd.DataFrame(subject_rows); subject_df.to_csv(HERE / 'per_subject.csv', index=False)
    raw = classification(pred.y.to_numpy(), pred.prob_raw.to_numpy()); adaptive = classification(pred.y.to_numpy(), pred.prob_direction_adaptive.to_numpy())
    macro_raw = {'auc': float(subject_df.raw_auc.mean()), 'f1': float(subject_df.raw_f1.mean())}; macro_adaptive = {'auc': float(subject_df.direction_adaptive_auc.mean()), 'f1': float(subject_df.direction_adaptive_f1.mean())}
    counts = {'raw': int(subject_df.raw_count.sum()), 'reverse': int(subject_df.reverse_count.sum())}
    summary = pd.DataFrame([{'arm':'raw','scope':'pooled_overall',**raw},{'arm':'direction_adaptive','scope':'pooled_overall',**adaptive},{'arm':'raw','scope':'macro_subject',**macro_raw},{'arm':'direction_adaptive','scope':'macro_subject',**macro_adaptive}]); summary.to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(HERE / 'input_metadata/folds.csv', index=False); pd.DataFrame(selection_rows).to_csv(HERE / 'input_metadata/direction_selection.csv', index=False)
    inputs = [(HERE / config[k]).resolve() for k in ('spectral_cache','temporal_cache','pe_cache')]
    sources = [HERE / n for n in ('run_experiment.py','test_094.py','config.json','test_protocol.txt')] + [Path(base.__file__), ROOT / '数据/数据集说明文档.md']
    prov = {'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, 'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, 'raw_shape': list(x.shape), 'public_arrays_loaded': False, 'n_subjects': int(ref.subject.nunique()), 'rows': int(len(ref)), 'normalization': config['normalization'], 'threads': 1, 'python': platform.python_version(), 'sklearn': __import__('sklearn').__version__}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    metrics = {'raw': {'overall_pooled': raw, 'macro_subject': macro_raw}, 'direction_adaptive': {'overall_pooled': adaptive, 'macro_subject': macro_adaptive}, 'direction_selection_counts': counts, 'subject_metrics': subject_rows, 'folds': fold_rows, 'elapsed_seconds': time.perf_counter()-start, 'provenance': prov}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    print(json.dumps({'raw': metrics['raw'], 'direction_adaptive': metrics['direction_adaptive'], 'direction_selection_counts': counts, 'elapsed_seconds': metrics['elapsed_seconds']}, ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
