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

BLOCKS = ('spectral150', 'temporal450', 'permutation_entropy120', 'all720')
SLICES = {'spectral150': slice(0, 150), 'temporal450': slice(150, 600), 'permutation_entropy120': slice(600, 720), 'all720': slice(0, 720)}

def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

def make_model(config):
    return make_pipeline(StandardScaler(), LinearSVC(**config['linear_svc'], random_state=config['random_state']))

def predict(model, x):
    return sigmoid(model.decision_function(x))

def classification(y, p):
    return {'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'f1': float(f1_score(y, p >= 0.5, zero_division=0))}

def load_data(config):
    return base.load_data(config)

def trial_keys(ref):
    return base.trial_keys(ref)

def choose_block(x, y, keys, train_keys, config):
    unique = list(dict.fromkeys(train_keys)); scores = {}
    for block in BLOCKS:
        inner_p, inner_y = [], []
        sl = SLICES[block]
        for held in unique:
            tr = np.isin(keys, unique) & (keys != held); va = keys == held
            model = make_model(config); model.fit(x[tr, sl], y[tr])
            inner_p.extend(predict(model, x[va, sl])); inner_y.extend(y[va])
        scores[block] = classification(np.asarray(inner_y), np.asarray(inner_p))
    best = max(BLOCKS, key=lambda b: ((-np.inf if scores[b]['auc'] is None else scores[b]['auc']), scores[b]['f1'], -BLOCKS.index(b)))
    return best, scores

def subject_run(subject, x, ref, config):
    ix = ref.subject.to_numpy() == subject; sx, sr = x[ix], ref.loc[ix].reset_index(drop=True)
    y, keys = sr.y.to_numpy(), trial_keys(sr); unique = list(dict.fromkeys(keys))
    assert len(unique) == 8 and all(np.sum(keys == k) == 5 for k in unique)
    selected_p, fixed_p = np.full(40, np.nan), np.full(40, np.nan); folds = np.full(40, -1, dtype=int)
    fold_meta, selections = [], []
    for fold, held in enumerate(unique):
        train, valid = keys != held, keys == held
        block, inner = choose_block(sx, y, keys, keys[train], config)
        selected_model = make_model(config); selected_model.fit(sx[train, SLICES[block]], y[train])
        fixed_model = make_model(config); fixed_model.fit(sx[train, SLICES['all720']], y[train])
        selected_p[valid] = predict(selected_model, sx[valid, SLICES[block]])
        fixed_p[valid] = predict(fixed_model, sx[valid, SLICES['all720']]); folds[valid] = fold
        selections.append(block)
        fold_meta.append({'subject': str(subject), 'fold': fold, 'held_out_trial_key': str(held), 'train_rows': int(train.sum()), 'valid_rows': int(valid.sum()), 'selected_block': block, 'inner_scores': inner})
    assert np.isfinite(selected_p).all() and np.isfinite(fixed_p).all() and np.array_equal(np.sort(folds), np.repeat(np.arange(8), 5))
    return sr, selected_p, fixed_p, folds, selections, fold_meta

def main():
    start = time.perf_counter(); config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    x, ref = load_data(config); subjects = ref.subject.to_numpy(); all_rows = []; subject_rows = []; fold_rows = []; selection_rows = []
    with threadpool_limits(limits=1):
        for subject in np.unique(subjects):
            sr, selected_p, fixed_p, folds, selections, meta = subject_run(subject, x, ref, config)
            out = sr.copy(); out['fold'] = folds; out['prob_selected_block'] = selected_p; out['prob_all720_fixed'] = fixed_p; out['subject'] = str(subject)
            all_rows.append(out)
            sm, fm = classification(sr.y.to_numpy(), selected_p), classification(sr.y.to_numpy(), fixed_p)
            subject_rows.append({'subject': str(subject), 'group': str(sr.group.iloc[0]), 'selected_auc': sm['auc'], 'selected_f1': sm['f1'], 'fixed_auc': fm['auc'], 'fixed_f1': fm['f1'], **{b + '_count': selections.count(b) for b in BLOCKS}})
            fold_rows.extend(meta); selection_rows.extend([{'subject': str(subject), 'fold': i, 'selected_block': b} for i, b in enumerate(selections)])
    pred = pd.concat(all_rows, ignore_index=True)[['row_index','subject','group','trial','position','y','fold','prob_selected_block','prob_all720_fixed']]
    pred.to_csv(HERE / 'predictions/within_subject_oof.csv', index=False)
    subject_df = pd.DataFrame(subject_rows); subject_df.to_csv(HERE / 'per_subject.csv', index=False)
    overall_selected = classification(pred.y.to_numpy(), pred.prob_selected_block.to_numpy()); overall_fixed = classification(pred.y.to_numpy(), pred.prob_all720_fixed.to_numpy())
    macro_selected = {'auc': float(subject_df.selected_auc.mean()), 'f1': float(subject_df.selected_f1.mean())}; macro_fixed = {'auc': float(subject_df.fixed_auc.mean()), 'f1': float(subject_df.fixed_f1.mean())}
    counts = {b: int(subject_df[b + '_count'].sum()) for b in BLOCKS}
    summary = pd.DataFrame([{'arm':'selected_block','scope':'pooled_overall', **overall_selected}, {'arm':'all720_fixed','scope':'pooled_overall', **overall_fixed}, {'arm':'selected_block','scope':'macro_subject', **macro_selected}, {'arm':'all720_fixed','scope':'macro_subject', **macro_fixed}]); summary.to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(HERE / 'input_metadata/folds.csv', index=False); pd.DataFrame(selection_rows).to_csv(HERE / 'input_metadata/block_selection.csv', index=False)
    inputs = [(HERE / config[k]).resolve() for k in ('spectral_cache','temporal_cache','pe_cache')]
    sources = [HERE / n for n in ('run_experiment.py','test_093.py','config.json','test_protocol.txt')] + [Path(base.__file__), ROOT / '数据/数据集说明文档.md']
    prov = {'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, 'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, 'raw_shape': list(x.shape), 'public_arrays_loaded': False, 'n_subjects': int(ref.subject.nunique()), 'rows': int(len(ref)), 'normalization': config['normalization'], 'threads': 1, 'python': platform.python_version(), 'sklearn': __import__('sklearn').__version__}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    metrics = {'selected_block': {'overall_pooled': overall_selected, 'macro_subject': macro_selected}, 'all720_fixed': {'overall_pooled': overall_fixed, 'macro_subject': macro_fixed}, 'block_selection_counts': counts, 'subject_metrics': subject_rows, 'folds': fold_rows, 'elapsed_seconds': time.perf_counter()-start, 'provenance': prov}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    print(json.dumps({'selected_block': metrics['selected_block'], 'all720_fixed': metrics['all720_fixed'], 'block_selection_counts': counts, 'elapsed_seconds': metrics['elapsed_seconds']}, ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
