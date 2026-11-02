import hashlib
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from emo_rec.covariance import regularized_covariances
from emo_rec.data import load_train_segments
from emo_rec.features import extract_spectral_features

spec = importlib.util.spec_from_file_location('exp070_084', HERE.parent / '070_train_context_match_new_splits' / 'run_experiment.py')
e070 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e070)
base = e070.base
KEYS = e070.KEYS


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def whiten_matrix(cov):
    cov = (cov + cov.T) / 2
    values, vectors = eigh(cov)
    if not np.isfinite(values).all() or values.min() <= 0: raise AssertionError('covariance is not positive definite')
    w = (vectors * (values ** -0.5)) @ vectors.T
    np.testing.assert_allclose(w, w.T, atol=1e-10)
    np.testing.assert_allclose(w @ cov @ w, np.eye(cov.shape[0]), atol=1e-7)
    return w


def positions(reference):
    return reference.groupby(['subject', 'y', 'trial'], sort=False).cumcount().to_numpy()


def load_cache(path):
    keys = ('X_train', 'subject_ids', 'health_groups', 'original_trial_ids', 'y_train', 'feature_names')
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in keys}


def load_data(config):
    c = HERE.parent / '070_train_context_match_new_splits'
    s, t, pe = [load_cache((c / config[k]).resolve()) for k in ('spectral_cache', 'temporal_cache', 'pe_cache')]
    for other in (t, pe):
        for key in ('subject_ids', 'health_groups', 'original_trial_ids', 'y_train'): np.testing.assert_array_equal(s[key], other[key])
    keep = s['subject_ids'] != config['excluded_subject']
    ref = pd.DataFrame({'row_index': np.arange(keep.sum()), 'subject': s['subject_ids'][keep], 'group': s['health_groups'][keep], 'trial': s['original_trial_ids'][keep], 'y': s['y_train'][keep]})
    ref['position'] = positions(ref)
    fixed = pd.read_csv((c / config['fixed_reference']).resolve())
    for col, old in [('subject','subject_id'),('group','health_group'),('trial','original_trial_id'),('y','y_true')]: np.testing.assert_array_equal(ref[col], fixed[old])
    raw_dir = ROOT / '数据' / '训练集'
    segments = []
    for subject in ref.subject.drop_duplicates():
        path = next(raw_dir.rglob(f'{subject}.mat'))
        x, _, _ = load_train_segments(path)
        source_rows = np.flatnonzero(s['subject_ids'] == subject)
        source_rows = source_rows[keep[source_rows]]
        # Cache order is the canonical neu/pos, trial, window order.
        assert len(source_rows) == 40
        segments.append((subject, x, source_rows))
    aligned = np.empty((len(ref), 30, 2500), dtype=np.float64)
    for subject, x, source_rows in segments:
        for pos in range(5):
            out = np.flatnonzero((ref.subject.to_numpy() == subject) & (ref.position.to_numpy() == pos))
            src = [i for i, r in enumerate(source_rows) if positions(pd.DataFrame({'subject': [subject]*40, 'y': s['y_train'][source_rows], 'trial': s['original_trial_ids'][source_rows]}))[i] == pos]
            cov = regularized_covariances(x[src], shrinkage=config['shrinkage']).mean(axis=0)
            w = whiten_matrix(cov)
            aligned[out] = np.einsum('ab,nbt->nat', w, x[src])
    spectral, names = extract_spectral_features(aligned, config['sampling_rate'])
    mask = np.asarray([str(n).startswith('log_bandpower_') for n in names])
    assert mask.sum() == 150
    raw = np.column_stack([spectral[:, mask], t['X_train'][keep], pe['X_train'][keep]])
    assert raw.shape == (2760, 720) and np.isfinite(raw).all()
    return raw, ref, fixed.fold.to_numpy(), {'source_files': [str(p) for p in sorted(raw_dir.glob('*/*.mat'))], 'raw_windows': int(len(ref)), 'public_arrays_loaded': False}


def fit_one(x, ref, name, folds, fold, config):
    train, valid = folds != fold, folds == fold
    start = time.perf_counter()
    with threadpool_limits(limits=config['threads_per_job']):
        model = base.make_model(config, config['outer_model_seed_base'] + fold)
        model.fit(x[train], ref.y.to_numpy()[train])
        p = model.predict_proba(x[valid])[:, 1]
    return name, fold, p, time.perf_counter() - start, int(train.sum())


def summarize(frame, col):
    rr = frame.rename(columns={'subject':'subject_id','group':'health_group'})
    return base.summarize(frame.y.to_numpy(), frame[col].to_numpy(), rr, frame.fold.to_numpy())


def main():
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8')); start = time.perf_counter()
    (HERE / 'predictions').mkdir(exist_ok=True)
    (HERE / 'input_metadata').mkdir(exist_ok=True)
    raw, ref, fixed, raw_info = load_data(config)
    splits = [('fixed002', fixed)] + [(f'repeat{s}', base.make_subject_folds(ref.subject.to_numpy(), ref.group.to_numpy(), s)) for s in config['repeat_seeds']]
    control_dir = HERE.parent / '070_train_context_match_new_splits'
    controls = {n: pd.read_csv(control_dir / 'predictions' / f'{n}_oof.csv') for n, _ in splits}
    for n, folds in splits:
        base.assert_subject_isolation(ref.subject.to_numpy(), folds)
    results = Parallel(n_jobs=config['n_jobs'], prefer='threads')(delayed(fit_one)(raw, ref, n, folds, fold, config) for n, folds in splits for fold in range(5))
    assert len(results) == 30
    runs, rows, deltas = {}, [], []
    for n, folds in splits:
        frame = ref.assign(fold=folds, p_train8=controls[n].p_train8.to_numpy(), p_alignment8=np.nan)
        for name, fold, p, _, _ in results:
            if name == n: frame.loc[folds == fold, 'p_alignment8'] = p
        frame.to_csv(HERE / 'predictions' / f'{n}_oof.csv', index=False)
        runs[n] = {}
        for subset in ['all_rows'] + [str(i) for i in config['positions']]:
            q = frame if subset == 'all_rows' else frame[frame.position == int(subset)]; runs[n][subset] = {}
            for arm, col in [('control8','p_train8'),('alignment8','p_alignment8')]:
                r = summarize(q, col); runs[n][subset][arm] = r; rows.append({'split':n,'position':subset,'arm':arm,**e070.scalars(r)})
            deltas.append({'split':n,'position':subset,**{k:e070.scalars(runs[n][subset]['alignment8'])[k]-e070.scalars(runs[n][subset]['control8'])[k] for k in KEYS}})
    inputs = [control_dir / 'predictions' / f'{n}_oof.csv' for n, _ in splits]
    sources = [HERE / n for n in ('run_experiment.py','test_084.py','config.json','test_protocol.txt')]
    sources += [ROOT / 'src/emo_rec/data.py', ROOT / 'src/emo_rec/features.py', ROOT / 'src/emo_rec/covariance.py', ROOT / '数据/数据集说明文档.md']
    prov = {'inputs': {str(p): sha256(p) for p in inputs}, 'sources': {str(p): sha256(p) for p in sources}, 'raw_sources': raw_info, 'raw_shape': list(raw.shape), 'public_arrays_loaded': False, 'model_fits': 30, 'control_source': '070 predictions/*_oof.csv p_train8', 'shrinkage': config['shrinkage'], 'alignment': 'per subject and physical position; mean of 8 window covariances; symmetric inverse square root', 'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'elapsed_seconds': time.perf_counter()-start}
    metrics = {'runs':runs,'fit_seconds':[{'split':r[0],'fold':r[1],'seconds':r[3],'n_train':r[4]} for r in results],'provenance':prov}
    (HERE/'predictions').mkdir(exist_ok=True); (HERE/'input_metadata').mkdir(exist_ok=True)
    (HERE/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8'); (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE/'summary.csv', index=False); pd.DataFrame(deltas).to_csv(HERE/'paired_delta.csv', index=False)

if __name__ == '__main__': main()
