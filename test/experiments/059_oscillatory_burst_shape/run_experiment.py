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
from scipy.signal import butter, hilbert, sosfiltfilt
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
SOURCE = HERE.parent / '057_multiscale_permutation_entropy'
spec = importlib.util.spec_from_file_location('exp057_source', SOURCE / 'run_experiment.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
base = source.base
METHODS = ('control_pe', 'plus_burst')
KEYS = ('auc', 'f1', 'balanced_accuracy', 'subject_auc_mean', 'hc_auc', 'dep_auc', 'worst_subject_auc')
ALIGN_KEYS = source.ALIGN_KEYS


def _minimum_samples(limits, fs):
    return int(np.ceil(2.0 / np.mean(limits) * fs))


def _runs(mask, minimum):
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= minimum]


def burst_features(segments, config, return_details=False):
    x = np.asarray(segments, dtype=float)
    fs = config['sampling_rate']; n = fs * config['window_seconds']
    if x.ndim != 3 or x.shape[1:] != (30, n):
        raise ValueError(f'expected windows (n, 30, {n})')
    if not np.isfinite(x).all():
        raise ValueError('nonfinite EEG')
    bc = config['burst']; edge = int(bc['edge_seconds'] * fs)
    outputs = []; details = []
    for band, limits in bc['bands'].items():
        sos = butter(bc['filter_order'], limits, btype='bandpass', fs=fs, output='sos')
        filtered = sosfiltfilt(sos, x, axis=-1)
        envelope = np.abs(hilbert(filtered, axis=-1))
        valid = envelope[..., edge:-edge]
        minimum = _minimum_samples(limits, fs)
        thresholds = 2.0 * np.median(valid, axis=-1)
        stats = np.zeros((len(x), 30, 4), dtype=float)
        band_detail = []
        for i in range(len(x)):
            row_detail = []
            for c in range(30):
                env = valid[i, c]
                threshold = thresholds[i, c]
                if threshold <= 0 or not np.isfinite(threshold):
                    events = []
                else:
                    events = _runs(env > threshold, minimum)
                durations = np.asarray([(b - a) / fs for a, b in events], dtype=float)
                rate = len(events) / (len(env) / fs)
                duty = durations.sum() / (len(env) / fs)
                mean_duration = durations.mean() if len(durations) else 0.0
                peak_ratios = [float(env[a:b].max() / max(np.median(env), 1e-30)) for a, b in events]
                stats[i, c] = (rate, duty, mean_duration, np.mean(peak_ratios) if peak_ratios else 0.0)
                row_detail.append(events)
            band_detail.append(row_detail)
        outputs.append(stats.reshape(len(x), -1))
        details.append(band_detail)
    result = np.concatenate(outputs, axis=1)
    return (result, details) if return_details else result


def feature_names(config):
    from emo_rec.features import CHANNEL_NAMES
    return [f'burst_{band}_{ch}_{stat}' for band in config['burst']['bands'] for ch in CHANNEL_NAMES for stat in ('event_rate_s', 'duty_cycle', 'mean_duration_s', 'mean_peak_over_median')]


def build_cache(config, path):
    from emo_rec.data import load_public_test_segments, load_train_segments
    files = sorted((ROOT / '数据/训练集').glob('*/*.mat'))
    public_files = sorted((ROOT / '数据/公开测试集').glob('*.mat'), key=lambda p: int(p.stem.replace('P_test', '')))
    assert len(files) == 70 and len(public_files) == 10
    train, ys, subjects, groups, trials = [], [], [], [], []
    for i, f in enumerate(files, 1):
        seg, y, trial = load_train_segments(f)
        train.append(burst_features(seg, config)); ys.append(y); trials.append(trial)
        subjects.extend([f.stem] * len(seg)); groups.extend(['DEP' if f.parent.name == '抑郁症患者' else 'HC'] * len(seg))
        print(f'BURST {i}/{len(files)} {f.stem}', flush=True)
    public, public_users, public_trials = [], [], []
    for f in public_files:
        v = burst_features(load_public_test_segments(f), config); public.append(v)
        public_users.extend([f.stem] * len(v)); public_trials.extend(np.arange(1, 9))
    names = feature_names(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X_train=np.concatenate(train), y_train=np.concatenate(ys), subject_ids=np.asarray(subjects), health_groups=np.asarray(groups), original_trial_ids=np.concatenate(trials), feature_names=np.asarray(names), X_public=np.concatenate(public), public_user_ids=np.asarray(public_users), public_trial_ids=np.asarray(public_trials), config_sha256=np.asarray(base.sha256(HERE / 'config.json')))
    metadata = [{'path': str(f.relative_to(ROOT)), 'bytes': f.stat().st_size, 'sha256': base.sha256(f)} for f in files + public_files]
    (HERE / 'input_metadata').mkdir(exist_ok=True)
    (HERE / 'input_metadata/raw_inputs.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')


def assert_alignment(a, b):
    for key in ALIGN_KEYS:
        if not np.array_equal(a[key], b[key]):
            raise RuntimeError(f'cache alignment mismatch: {key}')


def load_data(config):
    old_config = json.loads((SOURCE / 'config.json').read_text(encoding='utf8'))
    A, AP, ref, spectral, pe = source.load_data(old_config)
    path = HERE / config['burst_cache']
    if not path.exists():
        build_cache(config, path)
    burst = base.load_cache(path)
    assert str(burst['config_sha256']) == base.sha256(HERE / 'config.json')
    assert_alignment(spectral, burst)
    keep = spectral['subject_ids'] != config['excluded_subject']
    subs, pub = spectral['subject_ids'][keep], spectral['public_user_ids']
    from emo_rec.normalization import normalize_by_subject
    z = lambda m, ids: normalize_by_subject(m, ids, 'zscore')
    B, BP = z(burst['X_train'][keep], subs), z(burst['X_public'], pub)
    assert B.shape == (2760, 240) and BP.shape == (80, 240)
    assert np.isfinite(B).all() and np.isfinite(BP).all()
    return np.column_stack([A, B]), np.column_stack([AP, BP]), ref, spectral, burst


def column_sets():
    return {'control_pe': np.arange(720), 'plus_burst': np.arange(960)}


def fit_predict(A, y, train, valid, config, seed):
    out = {}
    with threadpool_limits(limits=1):
        for method, cols in column_sets().items():
            model = base.make_model(config, seed); model.fit(A[train][:, cols], y[train])
            out[method] = model.predict_proba(A[valid][:, cols])[:, 1]
    return out


def job(A, y, name, folds, fold, config):
    return name, fold, fit_predict(A, y, folds != fold, folds == fold, config, config['outer_model_seed_base'] + fold)


def fit_public(A, AP, y, config, seed):
    out = {}
    with threadpool_limits(limits=1):
        for method, cols in column_sets().items():
            model = base.make_model(config, seed); model.fit(A[:, cols], y)
            out[method] = model.predict_proba(AP[:, cols])[:, 1]
    return out


def scalars(result):
    return {**{k: result['overall'][k] for k in KEYS[:3]}, 'subject_auc_mean': result['subject_auc_mean'], 'hc_auc': result['health_groups']['HC']['pooled']['auc'], 'dep_auc': result['health_groups']['DEP']['pooled']['auc'], 'worst_subject_auc': min(v['auc'] for v in result['subjects'])}


def accepts(delta, fixed_auc, config):
    a = config['acceptance']
    return bool(delta['auc'] >= a['repeat_delta_auc_min'] and delta['f1'] >= a['repeat_delta_f1_min'] and delta['dep_auc'] >= a['repeat_delta_dep_auc_min'] and fixed_auc > a['fixed_delta_auc_strict_min'])


class Tee:
    def __init__(self, path): self.file = path.open('w', encoding='utf8'); self.stdout = sys.stdout
    def write(self, text): self.stdout.write(text); self.file.write(text); self.file.flush()
    def flush(self): self.stdout.flush(); self.file.flush()


def main():
    sys.stdout = Tee(HERE / 'run_output.txt'); start = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    for d in ('predictions', 'input_metadata'): (HERE / d).mkdir(exist_ok=True)
    (HERE / 'test_protocol.txt').write_text('固定协议：先构建/验证缓存，再训练。burst为窗内阈上包络事件，不是已证实生理burst；不跨窗、不用标签。4阶Butterworth SOS alpha 8-13 Hz、beta 13-30 Hz，Hilbert包络，去首尾0.5秒，阈值为该窗通道band包络中位数×2，严格阈上连续run，最短样本数为ceil(2/该band中心频率*fs)，各band独立。每band/channel输出事件率、占空比、平均持续时间、平均峰值/包络中位数，alpha与beta分别保留，共240维（control为720维，plus_burst为960维）；无事件/零信号为0。控制特征与057完全复用，逐subject/user无标签zscore。固定002及2027-2031重复，被试隔离；门槛为重复ΔAUC>=.005、ΔF1>=0、ΔDEP AUC>=0、fixed002 ΔAUC>0，不调参。', encoding='utf8')
    A, AP, ref, spectral, burst = load_data(config)
    y, subs, groups = ref.y_true.to_numpy(), ref.subject_id.to_numpy(), ref.health_group.to_numpy()
    splits = [('fixed002', ref.fold.to_numpy())] + [(f'repeat{s}', base.make_subject_folds(subs, groups, s)) for s in config['repeat_seeds']]
    for name, folds in splits:
        base.assert_subject_isolation(subs, folds); ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
    prov = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'inputs': {'source_script': base.sha256(SOURCE / 'run_experiment.py'), 'config': base.sha256(HERE / 'config.json')}, 'train_shape': list(A.shape), 'public_shape': list(AP.shape), 'alignment_checked': list(ALIGN_KEYS), 'subject_isolation': True, 'subject_zscore': True, 'public_metrics': False, 'feature_names': burst['feature_names'].tolist()}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding='utf8')
    results = Parallel(n_jobs=config['n_jobs'])(delayed(job)(A, y, name, folds, fold, config) for name, folds in splits for fold in range(5))
    fmap = dict(splits); pred = {name: {m: np.full(len(y), np.nan) for m in METHODS} for name, _ in splits}
    for name, fold, out in results:
        for method in METHODS: pred[name][method][fmap[name] == fold] = out[method]
    runs, rows, checks = {m: {} for m in METHODS}, [], {}
    for name, folds in splits:
        rr = ref.assign(fold=folds)
        for method in METHODS:
            assert np.isfinite(pred[name][method]).all(); pd.DataFrame({**{c: rr[c] for c in rr.columns}, 'prob_positive': pred[name][method]}).to_csv(HERE / 'predictions' / f'{method}_{name}_oof.csv', index=False)
            result = base.summarize(y, pred[name][method], rr, folds); runs[method][name] = result; rows.append({'method': method, 'split': name, **scalars(result)})
        old = pd.read_csv(SOURCE / 'predictions' / f'plus_pe_{name}_oof.csv'); checks[name] = float(np.max(np.abs(old.prob_positive.to_numpy() - pred[name]['control_pe']))); assert checks[name] < 1e-12
    public = fit_public(A, AP, y, config, config['outer_model_seed_base'])
    for method in METHODS:
        frame = pd.DataFrame({'user_id': spectral['public_user_ids'], 'trial_id': spectral['public_trial_ids'], 'prob. of positive emotion': public[method]}); frame.to_csv(HERE / 'predictions' / f'{method}_public.csv', index=False)
    aggregate = {}
    for method in METHODS:
        vals = [scalars(runs[method][f'repeat{s}']) for s in config['repeat_seeds']]; aggregate[method] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}; rows.append({'method': method, 'split': 'repeat_mean', **{k: aggregate[method][k]['mean'] for k in KEYS}})
    delta = {m: {'repeat_mean': {k: aggregate[m][k]['mean'] - aggregate['control_pe'][k]['mean'] for k in KEYS}, 'fixed002': {k: scalars(runs[m]['fixed002'])[k] - scalars(runs['control_pe']['fixed002'])[k] for k in KEYS}, 'repeat_paired_auc': [a - b for a, b in zip(aggregate[m]['auc']['values'], aggregate['control_pe']['auc']['values'])]} for m in METHODS[1:]}
    decision = {m: accepts(d['repeat_mean'], d['fixed002']['auc'], config) for m, d in delta.items()}; elapsed = time.perf_counter() - start
    metrics = {'config': config, 'provenance': prov, 'runs': runs, 'aggregate': aggregate, 'deltas_vs_control': delta, 'decision_accept': decision, 'control_reproduction_max_abs_diff_vs_057': checks, 'elapsed_seconds': elapsed}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf8'); pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    lines = ['# 实验059：振荡包络阈上事件形状', '', 'burst统计是窗内阈上包络事件，不是已证实的生理burst；不宣称文献支持情绪预测提升。方法动机参考 PMC6882750 与 eLife-29086-v2；阈值伪事件/方法敏感性反证参考 HAL-04801753v1。', '', '|方法|五重复AUC均值±SD|F1均值|DEP AUC均值|', '|---|---:|---:|---:|']
    for m in METHODS: lines.append(f"|{m}|{aggregate[m]['auc']['mean']:.6f}±{aggregate[m]['auc']['std']:.6f}|{aggregate[m]['f1']['mean']:.6f}|{aggregate[m]['dep_auc']['mean']:.6f}|")
    lines += ['', '|方法|ΔAUC|ΔF1|ΔDEP AUC|fixed002 ΔAUC|达线|', '|---|---:|---:|---:|---:|---|']
    for m, d in delta.items(): lines.append(f"|{m}|{d['repeat_mean']['auc']:+.6f}|{d['repeat_mean']['f1']:+.6f}|{d['repeat_mean']['dep_auc']:+.6f}|{d['fixed002']['auc']:+.6f}|{decision[m]}|")
    lines += ['', f'真实耗时：{elapsed:.2f}秒；control_pe逐概率与057 plus_pe核对最大差：{max(checks.values()):.3g}。不调参。', '完整指标、OOF、folds、输入哈希与运行日志见本目录。']; (HERE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf8'); print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__': main()
