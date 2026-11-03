import hashlib, importlib.util, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('exp067', HERE.parent / '067_adaptation_context_size' / 'run_experiment.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
ARMS = ('standard_scaler_lr', 'standard_scaler_pca10_lr')

def load_data(config):
    raw, ref, _ = base.load_data(config)
    return raw, ref

def trial_keys(ref):
    return np.asarray([f'{y}::{t}' for y, t in zip(ref.y, ref.trial)], dtype=object)

def model(arm, config):
    steps = [('scaler', StandardScaler())]
    if arm == 'standard_scaler_pca10_lr':
        steps.append(('pca', PCA(n_components=config['pca_components'], svd_solver='full', random_state=config['random_state'])))
    steps.append(('lr', LogisticRegression(C=config['lr_C'], max_iter=2000, solver='lbfgs', random_state=config['random_state'])))
    return make_pipeline(*[v for _, v in steps])

def classification(y, p):
    return {'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'f1': float(f1_score(y, p >= 0.5, zero_division=0))}

def subject_run(subject, x, ref, config):
    ix = ref.subject.to_numpy() == subject
    sx, sr = x[ix], ref.loc[ix].reset_index(drop=True)
    y, keys = sr.y.to_numpy(), trial_keys(sr)
    unique = list(dict.fromkeys(keys))
    assert len(unique) == 8 and all(np.sum(keys == k) == 5 for k in unique)
    probs = {a: np.full(40, np.nan) for a in ARMS}; folds = np.full(40, -1, dtype=int); meta = []
    for fold, held in enumerate(unique):
        train, valid = keys != held, keys == held
        folds[valid] = fold
        row = {'subject': str(subject), 'fold': fold, 'held_out_trial_key': str(held), 'train_rows': int(train.sum()), 'valid_rows': int(valid.sum())}
        for arm in ARMS:
            fitted = model(arm, config); fitted.fit(sx[train], y[train]); probs[arm][valid] = fitted.predict_proba(sx[valid])[:, 1]
        meta.append(row)
    assert all(np.isfinite(p).all() for p in probs.values()) and np.array_equal(np.sort(folds), np.repeat(np.arange(8), 5))
    metrics = {a: classification(y, probs[a]) for a in ARMS}
    return sr, probs, folds, metrics, meta

def main():
    start = time.perf_counter(); config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    x, ref = load_data(config); all_rows = []; subject_rows = []; fold_rows = []
    with threadpool_limits(limits=1):
        for subject in np.unique(ref.subject.to_numpy()):
            sr, probs, folds, metrics, meta = subject_run(subject, x, ref, config)
            out = sr.copy(); out['fold'] = folds; out['subject'] = str(subject)
            for arm in ARMS: out['prob_' + arm] = probs[arm]
            all_rows.append(out); fold_rows.extend(meta)
            row = {'subject': str(subject), 'group': str(sr.group.iloc[0])}
            for arm in ARMS:
                row.update({arm + '_' + k: v for k, v in metrics[arm].items()})
            subject_rows.append(row)
    pred = pd.concat(all_rows, ignore_index=True)
    pred = pred[['row_index','subject','group','trial','position','y','fold'] + ['prob_' + a for a in ARMS]]
    pred.to_csv(HERE / 'predictions/within_subject_oof.csv', index=False)
    subjects_df = pd.DataFrame(subject_rows); subjects_df.to_csv(HERE / 'per_subject.csv', index=False)
    summary_rows = []; overall = {}; macro = {}
    for arm in ARMS:
        overall[arm] = classification(pred.y.to_numpy(), pred['prob_' + arm].to_numpy())
        macro[arm] = {'auc': float(subjects_df[arm + '_auc'].mean()), 'f1': float(subjects_df[arm + '_f1'].mean())}
        summary_rows.extend([{'arm': arm, 'scope': 'pooled_overall', **overall[arm]}, {'arm': arm, 'scope': 'macro_subject', **macro[arm]}])
    pd.DataFrame(summary_rows).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(HERE / 'input_metadata/folds.csv', index=False)
    inputs = [(HERE / config[k]).resolve() for k in ('spectral_cache','temporal_cache','pe_cache','fixed_reference')]
    sources = [HERE / n for n in ('run_experiment.py','test_092.py','config.json','test_protocol.txt')] + [Path(base.__file__), ROOT / '数据/数据集说明文档.md']
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    prov = {'inputs': {str(p): sha(p) for p in inputs}, 'sources': {str(p): sha(p) for p in sources}, 'raw_shape': list(x.shape), 'public_arrays_loaded': False, 'n_subjects': int(ref.subject.nunique()), 'rows': int(len(ref)), 'outer_train_rows': 35, 'outer_valid_rows': 5, 'fit_policy': 'each arm independently fits scaler/PCA/LR on outer 35 windows only', 'threads': 1, 'python': platform.python_version(), 'sklearn': __import__('sklearn').__version__}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    metrics = {'overall_pooled': overall, 'macro_subject': macro, 'subject_metrics': subject_rows, 'folds': fold_rows, 'elapsed_seconds': time.perf_counter()-start, 'provenance': prov}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    print(json.dumps({'overall_pooled': overall, 'macro_subject': macro, 'elapsed_seconds': metrics['elapsed_seconds']}, ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
