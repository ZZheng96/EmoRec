import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.signal import stft
from scipy.special import rel_entr
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
SOURCE = HERE.parent / '057_multiscale_permutation_entropy'
spec = importlib.util.spec_from_file_location('exp057_source', SOURCE / 'run_experiment.py')
source = importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
base = source.base
METHODS = ('control_pe', 'plus_spectral_js')
KEYS = source.KEYS
ALIGN_KEYS = source.ALIGN_KEYS
scalars = source.scalars
accepts = source.accepts
Tee = source.Tee


def spectral_js_features(segments, config):
    x = np.asarray(segments, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (30, 2500):
        raise ValueError('expected independent windows (n, 30, 2500)')
    if not np.isfinite(x).all():
        raise ValueError('nonfinite EEG')
    f, _, z = stft(x, fs=config['sampling_rate'], axis=-1, **config['stft'])
    power = np.abs(z[..., (f >= 1) & (f <= 45), :]) ** 2
    assert power.shape[-2:] == (45, 19)
    total = power.sum(axis=-2, keepdims=True)
    p = np.divide(power, total, out=np.full_like(power, 1 / 45), where=total > 0)
    a, b = p[..., :-1], p[..., 1:]
    m = (a + b) / 2
    js = (rel_entr(a, m) + rel_entr(b, m)).sum(axis=-2) / (2 * np.log(2))
    return np.concatenate([js.mean(axis=-1), js.std(axis=-1, ddof=0)], axis=1)


def feature_names():
    from emo_rec.features import CHANNEL_NAMES
    return [f'spectral_js_{stat}_{ch}' for stat in ('mean', 'std') for ch in CHANNEL_NAMES]


def build_cache(config, path):
    from emo_rec.data import load_train_segments
    files = sorted((ROOT / '数据/训练集').glob('*/*.mat'))
    assert len(files) == 70
    feats, ys, subs, groups, trials = [], [], [], [], []
    for i, f in enumerate(files, 1):
        seg, y, t = load_train_segments(f)
        feats.append(spectral_js_features(seg, config)); ys.append(y); trials.append(t)
        subs.extend([f.stem] * len(seg)); groups.extend(['DEP' if f.parent.name == '抑郁症患者' else 'HC'] * len(seg))
        print(f'spectral_js {i}/{len(files)} {f.stem}', flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = base.load_cache(SOURCE / 'cache/pe_features.npz')
    np.savez_compressed(path, X_train=np.concatenate(feats), y_train=np.concatenate(ys), subject_ids=np.asarray(subs), health_groups=np.asarray(groups), original_trial_ids=np.concatenate(trials), feature_names=np.asarray(feature_names()), public_user_ids=old['public_user_ids'], public_trial_ids=old['public_trial_ids'], config_sha256=np.asarray(base.sha256(HERE / 'config.json')))


def load_data(config):
    old_config = json.loads((SOURCE / 'config.json').read_text(encoding='utf8'))
    assert (SOURCE / old_config['pe_cache']).exists()
    A, AP, ref, s, pe = source.load_data(old_config)
    assert AP.shape == (80, 720)
    path = HERE / config['spectral_js_cache']
    if not path.exists(): build_cache(config, path)
    o = base.load_cache(path)
    assert str(o['config_sha256']) == base.sha256(HERE / 'config.json')
    source.assert_alignment(s, o)
    keep = s['subject_ids'] != config['excluded_subject']; subs = s['subject_ids'][keep]
    from emo_rec.normalization import normalize_by_subject
    O = normalize_by_subject(o['X_train'][keep], subs, 'zscore')
    fixed = pd.read_csv((HERE / config['fixed_reference']).resolve())
    pd.testing.assert_frame_equal(ref, fixed[ref.columns])
    assert A.shape == (2760, 720) and O.shape == (2760, 60)
    assert np.isfinite(np.column_stack([A, O])).all()
    return np.column_stack([A, O]), ref, s, o


def column_sets(): return {'control_pe': np.arange(720), 'plus_spectral_js': np.arange(780)}


def fit_predict(A, y, train, valid, config, seed):
    out = {}
    with threadpool_limits(limits=1):
        for method, cols in column_sets().items():
            model = base.make_model(config, seed)
            model.fit(A[train][:, cols], y[train])
            out[method] = model.predict_proba(A[valid][:, cols])[:, 1]
    return out


def job(A, y, folds, fold, config):
    return fold, fit_predict(A, y, folds != fold, folds == fold, config, config['outer_model_seed_base'] + fold)


def main():
    sys.stdout = Tee(HERE / 'run_output.txt'); start = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    for d in ('predictions', 'input_metadata'): (HERE / d).mkdir(exist_ok=True)
    raw = [{'path': str(f.relative_to(ROOT)), 'bytes': f.stat().st_size, 'sha256': base.sha256(f)} for f in sorted((ROOT / '数据/训练集').glob('*/*.mat'))]
    (HERE / 'input_metadata/raw_inputs.json').write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding='utf8')
    A, ref, s, o = load_data(config)
    y, subs, groups = ref.y_true.to_numpy(), ref.subject_id.to_numpy(), ref.health_group.to_numpy()
    old_config = json.loads((SOURCE / 'config.json').read_text(encoding='utf8'))
    inputs = [HERE / 'config.json', HERE / 'protocol.md', Path(__file__), SOURCE / 'run_experiment.py', SOURCE / 'config.json', HERE / config['spectral_js_cache']]
    inputs += [(SOURCE / old_config[k]).resolve() for k in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')]
    prov = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'inputs': [{'path': str(f), 'sha256': base.sha256(f)} for f in inputs], 'train_shape': list(A.shape), 'public_training': False, 'public_predictions': False, 'feature_names': o['feature_names'].tolist(), 'alignment_checked': list(ALIGN_KEYS), 'subject_isolation': True, 'subject_zscore': True}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding='utf8')
    splits = [('fixed002', ref.fold.to_numpy())] + [(f'repeat{seed}', base.make_subject_folds(subs, groups, seed)) for seed in config['repeat_seeds']]
    for name, folds in splits:
        assert set(folds) == set(range(5))
        base.assert_subject_isolation(subs, folds)
        ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
    print(f'data ready {A.shape}; frozen paired 30 folds / 60 models', flush=True)
    sys.path.insert(0, str(HERE))
    import run_experiment as importable
    results = Parallel(n_jobs=config['n_jobs'])(delayed(importable.job)(A, y, folds, fold, config) for _, folds in splits for fold in range(5))
    runs, rows, checks = {m: {} for m in METHODS}, [], {}
    for name, folds in splits:
        pred = {m: np.full(len(y), np.nan) for m in METHODS}
        for fold in range(5):
            out = results.pop(0); assert out[0] == fold
            for m in METHODS: pred[m][folds == fold] = out[1][m]
        rr = ref.assign(fold=folds)
        old = pd.read_csv(SOURCE / 'predictions' / f'plus_pe_{name}_oof.csv')
        pd.testing.assert_frame_equal(old[rr.columns], rr)
        checks[name] = float(np.max(np.abs(old.prob_positive.to_numpy() - pred['control_pe'])))
        assert checks[name] <= 1e-12
        for m in METHODS:
            assert np.isfinite(pred[m]).all() and ((pred[m] >= 0) & (pred[m] <= 1)).all()
            rr.assign(prob_positive=pred[m]).to_csv(HERE / 'predictions' / f'{m}_{name}_oof.csv', index=False)
            result = base.summarize(y, pred[m], rr, folds)
            runs[m][name] = result; rows.append({'method': m, 'split': name, **scalars(result)})
    aggregate = {}
    for m in METHODS:
        vals = [scalars(runs[m][f'repeat{s}']) for s in config['repeat_seeds']]
        aggregate[m] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
        rows.append({'method': m, 'split': 'repeat_mean', **{k: aggregate[m][k]['mean'] for k in KEYS}})
    delta = {'plus_spectral_js': {'repeat_mean': {k: aggregate['plus_spectral_js'][k]['mean'] - aggregate['control_pe'][k]['mean'] for k in KEYS}, 'fixed002': {k: scalars(runs['plus_spectral_js']['fixed002'])[k] - scalars(runs['control_pe']['fixed002'])[k] for k in KEYS}, 'repeat_paired_auc': [a-b for a, b in zip(aggregate['plus_spectral_js']['auc']['values'], aggregate['control_pe']['auc']['values'])]}}
    d = delta['plus_spectral_js']; decision = {'plus_spectral_js': accepts(d['repeat_mean'], d['fixed002']['auc'], config)}
    elapsed = time.perf_counter() - start
    metrics = {'config': config, 'provenance': prov, 'runs': runs, 'aggregate': aggregate, 'deltas_vs_control': delta, 'decision_accept': decision, 'control_reproduction_max_abs_diff_vs_057': checks, 'elapsed_seconds': elapsed}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf8')
    (HERE / 'input_metadata/alignment_audit.json').write_text(json.dumps({'cache_keys_checked': list(ALIGN_KEYS), 'rows': len(y), 'subjects': len(np.unique(subs)), 'excluded': config['excluded_subject'], 'paired_reference_columns': list(ref.columns), 'control_max_abs_diff_vs_057': checks, 'subject_isolation_all_splits': True}, indent=2), encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    lines = ['# 实验064：单窗谱形状动态', '', '冻结协议见protocol.md；不调参，不公开训练或预测。', '', '|方法|五重复AUC均值±SD|F1均值|DEP AUC均值|', '|---|---:|---:|---:|']
    for m in METHODS: lines.append(f"|{m}|{aggregate[m]['auc']['mean']:.6f}±{aggregate[m]['auc']['std']:.6f}|{aggregate[m]['f1']['mean']:.6f}|{aggregate[m]['dep_auc']['mean']:.6f}|")
    lines += ['', f"五重复ΔAUC={d['repeat_mean']['auc']:+.6f}，ΔF1={d['repeat_mean']['f1']:+.6f}，ΔDEP AUC={d['repeat_mean']['dep_auc']:+.6f}；fixed002 ΔAUC={d['fixed002']['auc']:+.6f}。接受：{decision['plus_spectral_js']}。未达线即停止。", f'真实运行耗时：{elapsed:.2f}秒；控制对057最大差：{max(checks.values()):.3g}。12份OOF见predictions；完整逐折/逐被试指标见metrics.json。', '', 'Shen/Lin2019（https://pmc.ncbi.nlm.nih.gov/articles/PMC6831623/ ）跨日非平稳仅作动机，不证明秒级JS有效。SciPy jensenshannon为sqrt(divergence)，本轮用base2 divergence。1秒窗频率分辨率和随机谱估计限制；50%重叠引入相邻相关；非平稳不等于情绪机制；mean/std丢失变化方向。']
    (HERE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(json.dumps({'delta': delta, 'decision': decision, 'elapsed_seconds': elapsed}, indent=2), flush=True)


if __name__ == '__main__': main()
