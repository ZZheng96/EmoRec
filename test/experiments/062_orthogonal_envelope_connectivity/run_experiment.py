import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt, hilbert
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
SOURCE = HERE.parent / '057_multiscale_permutation_entropy'
spec = importlib.util.spec_from_file_location('exp057_source', SOURCE / 'run_experiment.py')
source = importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
base = source.base
METHODS = ('control_pe', 'plus_oAEC')
KEYS = ('auc', 'f1', 'balanced_accuracy', 'subject_auc_mean', 'hc_auc', 'dep_auc', 'worst_subject_auc')
ALIGN_KEYS = source.ALIGN_KEYS


def pearson_last(a, b):
    ac = a - a.mean(axis=-1, keepdims=True)
    bc = b - b.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(np.sum(ac * ac, axis=-1) * np.sum(bc * bc, axis=-1))
    return np.divide(np.sum(ac * bc, axis=-1), denominator, out=np.zeros_like(denominator), where=denominator > 0)


def orthogonal_envelope_matrix(z):
    z = np.asarray(z, dtype=np.complex128)
    amplitude = np.abs(z)
    directed = np.zeros((len(z), len(z)), dtype=np.float64)
    for i in range(len(z)):
        orthogonal = np.abs(np.imag(z * np.conj(z[i]) / np.maximum(amplitude[i], np.finfo(np.float64).eps)))
        directed[i] = np.abs(pearson_last(amplitude[i], orthogonal))
    symmetric = (directed + directed.T) / 2
    np.fill_diagonal(symmetric, 0)
    return symmetric


def oAEC_features(segments, config):
    x = np.asarray(segments, dtype=np.float64)
    n = config['sampling_rate'] * config['window_seconds']
    if x.ndim != 3 or x.shape[1:] != (30, n): raise ValueError(f'expected windows (n, 30, {n})')
    if not np.isfinite(x).all(): raise ValueError('nonfinite EEG')
    oc = config['oAEC']; trim = int(oc['trim_seconds'] * config['sampling_rate'])
    blocks = []
    for band in oc['bands']:
        sos = butter(oc['order'], band, btype='bandpass', fs=config['sampling_rate'], output='sos')
        z = hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)[..., trim:-trim]
        blocks.append(np.asarray([orthogonal_envelope_matrix(window).sum(axis=1) / 29 for window in z]))
    return np.concatenate(blocks, axis=1)


def feature_names(config):
    from emo_rec.features import CHANNEL_NAMES
    return [f'oAEC_{lo}_{hi}_{ch}_mean' for lo, hi in config['oAEC']['bands'] for ch in CHANNEL_NAMES]


def build_cache(config, path):
    from emo_rec.data import load_train_segments
    files = sorted((ROOT / '数据/训练集').glob('*/*.mat'))
    feats, ys, subs, groups, trials = [], [], [], [], []
    for i, f in enumerate(files, 1):
        seg, y, t = load_train_segments(f); feats.append(oAEC_features(seg, config)); ys.append(y); trials.append(t)
        subs.extend([f.stem] * len(seg)); groups.extend(['DEP' if f.parent.name == '抑郁症患者' else 'HC'] * len(seg)); print(f'oAEC {i}/{len(files)} {f.stem}', flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_cache = base.load_cache(SOURCE / 'cache/pe_features.npz')
    np.savez_compressed(path, X_train=np.concatenate(feats), y_train=np.concatenate(ys), subject_ids=np.asarray(subs), health_groups=np.asarray(groups), original_trial_ids=np.concatenate(trials), feature_names=np.asarray(feature_names(config)), public_user_ids=source_cache['public_user_ids'], public_trial_ids=source_cache['public_trial_ids'], config_sha256=np.asarray(base.sha256(HERE / 'config.json')))
    metadata = [{'path': str(f.relative_to(ROOT)), 'bytes': f.stat().st_size, 'sha256': base.sha256(f)} for f in files]
    (HERE / 'input_metadata').mkdir(exist_ok=True); (HERE / 'input_metadata/raw_inputs.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')


def assert_alignment(a, b):
    for key in ALIGN_KEYS:
        if not np.array_equal(a[key], b[key]): raise RuntimeError(f'cache alignment mismatch: {key}')


def load_data(config):
    old_config = json.loads((SOURCE / 'config.json').read_text(encoding='utf8'))
    assert (SOURCE / old_config['pe_cache']).exists()
    A, AP, ref, s, pe = source.load_data(old_config)
    assert AP.shape == (80, 720)
    path = HERE / config['oAEC_cache']
    if not path.exists(): build_cache(config, path)
    o = base.load_cache(path); assert str(o['config_sha256']) == base.sha256(HERE / 'config.json'); assert_alignment(s, o)
    keep = s['subject_ids'] != config['excluded_subject']; subs = s['subject_ids'][keep]
    from emo_rec.normalization import normalize_by_subject
    z = lambda a, ids: normalize_by_subject(a, ids, 'zscore')
    O = z(o['X_train'][keep], subs)
    ref = pd.read_csv((HERE / config['fixed_reference']).resolve())
    for col, arr in [('subject_id', subs), ('health_group', s['health_groups'][keep]), ('y_true', s['y_train'][keep]), ('original_trial_id', s['original_trial_ids'][keep])]: np.testing.assert_array_equal(ref[col].to_numpy(), arr)
    assert A.shape == (2760, 720) and O.shape == (2760, 120) and np.isfinite(np.column_stack([A, O])).all()
    return np.column_stack([A, O]), ref[['subject_id', 'health_group', 'original_trial_id', 'fold', 'y_true']], s, o


def column_sets(): return {'control_pe': np.arange(720), 'plus_oAEC': np.arange(840)}

def fit_predict(A, y, train, valid, config, seed):
    out = {}
    with threadpool_limits(limits=1):
        for method, cols in column_sets().items():
            model = base.make_model(config, seed); model.fit(A[train][:, cols], y[train]); out[method] = model.predict_proba(A[valid][:, cols])[:, 1]
    return out

def job(A, y, folds, fold, config): return fold, fit_predict(A, y, folds != fold, folds == fold, config, config['outer_model_seed_base'] + fold)

def scalars(r): return {**{k: r['overall'][k] for k in KEYS[:3]}, 'subject_auc_mean': r['subject_auc_mean'], 'hc_auc': r['health_groups']['HC']['pooled']['auc'], 'dep_auc': r['health_groups']['DEP']['pooled']['auc'], 'worst_subject_auc': min(v['auc'] for v in r['subjects'])}

def accepts(delta, fixed_auc, config):
    a = config['acceptance']; return bool(delta['auc'] >= a['repeat_delta_auc_min'] and delta['f1'] >= a['repeat_delta_f1_min'] and delta['dep_auc'] >= a['repeat_delta_dep_auc_min'] and fixed_auc > a['fixed_delta_auc_strict_min'])

class Tee:
    def __init__(self, path): self.file = path.open('w', encoding='utf8'); self.stdout = sys.stdout
    def write(self, text): self.stdout.write(text); self.file.write(text); self.file.flush()
    def flush(self): self.stdout.flush(); self.file.flush()


def main():
    sys.stdout = Tee(HERE / 'run_output.txt'); start = time.perf_counter(); config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    (HERE / 'predictions').mkdir(exist_ok=True); (HERE / 'input_metadata').mkdir(exist_ok=True)
    A, ref, s, o = load_data(config); y, subs, groups = ref.y_true.to_numpy(), ref.subject_id.to_numpy(), ref.health_group.to_numpy()
    prov = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'inputs': {'source_script': base.sha256(SOURCE / 'run_experiment.py'), 'config': base.sha256(HERE / 'config.json'), 'source_cache': base.sha256(SOURCE / 'cache/pe_features.npz')}, 'train_shape': list(A.shape), 'public_metrics': False, 'feature_names': o['feature_names'].tolist()}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding='utf8')
    splits = [('fixed002', ref.fold.to_numpy())] + [(f'repeat{s}', base.make_subject_folds(subs, groups, s)) for s in config['repeat_seeds']]; fmap = dict(splits)
    for name, folds in splits: base.assert_subject_isolation(subs, folds); ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
    results = Parallel(n_jobs=config['n_jobs'])(delayed(job)(A, y, folds, fold, config) for _, folds in splits for fold in range(5))
    pred = {name: {m: np.full(len(y), np.nan) for m in METHODS} for name, _ in splits}; runs, rows = {m: {} for m in METHODS}, []
    for (name, folds) in splits:
        for fold in range(5):
            out = results.pop(0)
            for method in METHODS: pred[name][method][folds == out[0]] = out[1][method]
        rr = ref.assign(fold=folds)
        for method in METHODS:
            frame = pd.DataFrame({**{c: rr[c] for c in rr.columns}, 'prob_positive': pred[name][method]}); frame.to_csv(HERE / 'predictions' / f'{method}_{name}_oof.csv', index=False)
            result = base.summarize(y, pred[name][method], rr, folds); runs[method][name] = result; rows.append({'method': method, 'split': name, **scalars(result)})
    aggregate = {}
    for method in METHODS:
        vals = [scalars(runs[method][f'repeat{s}']) for s in config['repeat_seeds']]; aggregate[method] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}; rows.append({'method': method, 'split': 'repeat_mean', **{k: aggregate[method][k]['mean'] for k in KEYS}})
    delta = {'plus_oAEC': {'repeat_mean': {k: aggregate['plus_oAEC'][k]['mean'] - aggregate['control_pe'][k]['mean'] for k in KEYS}, 'fixed002': {k: scalars(runs['plus_oAEC']['fixed002'])[k] - scalars(runs['control_pe']['fixed002'])[k] for k in KEYS}}}
    decision = {'plus_oAEC': accepts(delta['plus_oAEC']['repeat_mean'], delta['plus_oAEC']['fixed002']['auc'], config)}; elapsed = time.perf_counter() - start
    (HERE / 'metrics.json').write_text(json.dumps({'config': config, 'provenance': prov, 'runs': runs, 'aggregate': aggregate, 'deltas_vs_control': delta, 'decision_accept': decision, 'elapsed_seconds': elapsed}, ensure_ascii=False, indent=2), encoding='utf8'); pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    lines = ['# 实验062：跨通道正交包络相关', '', '本轮使用abs幅度，不是原Hipp log power方法的完整复现。Sensor orthogonalization不等于源空间消除混叠，也不能证明脑连接，不能消除所有ghost/间接混叠；10秒短窗可靠性未知。参考Hipp https://pmc.ncbi.nlm.nih.gov/articles/PMC3861400/ 、Nolte https://pmc.ncbi.nlm.nih.gov/articles/PMC7683718/ 、Palva2018 https://doi.org/10.1016/j.neuroimage.2018.02.032 。公开缓存仅复用057既有特征核对alignment，不计算公开新特征、不进行公开全训练及预测。', '', '|方法|五重复AUC均值±SD|F1均值|DEP AUC均值|', '|---|---:|---:|---:|']
    for m in METHODS: lines.append(f"|{m}|{aggregate[m]['auc']['mean']:.6f}±{aggregate[m]['auc']['std']:.6f}|{aggregate[m]['f1']['mean']:.6f}|{aggregate[m]['dep_auc']['mean']:.6f}|")
    d = delta['plus_oAEC']; lines += ['', '|方法|ΔAUC|ΔF1|ΔDEP AUC|fixed002 ΔAUC|达线|', '|---|---:|---:|---:|---:|---|', f"|plus_oAEC|{d['repeat_mean']['auc']:+.6f}|{d['repeat_mean']['f1']:+.6f}|{d['repeat_mean']['dep_auc']:+.6f}|{d['fixed002']['auc']:+.6f}|{decision['plus_oAEC']}|", '', f'真实耗时：{elapsed:.2f}秒；不调参。']; (HERE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf8'); print(pd.DataFrame(rows).to_string(index=False), flush=True)

if __name__ == '__main__': main()
