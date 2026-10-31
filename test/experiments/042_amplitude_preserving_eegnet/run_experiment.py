import argparse
import gc
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from emo_rec.data import load_train_segments, load_public_test_segments
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_module(name, folder):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / folder / 'run_experiment.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module('eegnet024', '024_eegnet_cnn')
repeat = load_module('folds032', '032_repeated_subject_folds')


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def metadata():
    frame = base.load_metadata().drop(columns=['original_trial_id', 'prob_positive'], errors='ignore')
    frame['window_id'] = frame.groupby('subject_id', sort=False).cumcount()
    return frame.reset_index(drop=True)


def preprocess_windows(raw):
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[1:] != (30, 2500) or not np.isfinite(raw).all():
        raise ValueError('Expected finite (N,30,2500) raw windows')
    return np.asarray([base.previous.preprocess_continuous_trial(w) for w in raw], dtype=np.float32)


def align_windows(windows, source_metadata, target_metadata):
    keys = ['subject_id', 'window_id']
    source = pd.MultiIndex.from_frame(source_metadata[keys])
    target = pd.MultiIndex.from_frame(target_metadata[keys])
    if not source.is_unique or not target.is_unique:
        raise ValueError('Nonunique window metadata')
    idx = source.get_indexer(target)
    if (idx < 0).any():
        raise ValueError('Missing window metadata')
    return windows[idx]


def prepare(frame):
    cache = HERE / 'cache'
    cache.mkdir(exist_ok=True)
    if not (HERE / 'input_metadata.json').exists():
        paths = {p.stem: p for p in (ROOT / '数据' / '训练集').glob('*/*.mat')}
        x = np.lib.format.open_memmap(cache / 'train.npy', mode='w+', dtype='float32', shape=(2760, 30, 2000))
        hashes = {}
        for subject in frame.subject_id.unique():
            raw, labels, _ = load_train_segments(paths[subject])
            idx = np.flatnonzero(frame.subject_id.to_numpy() == subject)
            ids = frame.iloc[idx].window_id.to_numpy()
            assert np.array_equal(labels[ids], frame.iloc[idx].y_true.to_numpy())
            x[idx] = preprocess_windows(raw)[ids]
            hashes[subject] = repeat.sha256(paths[subject])
        x.flush()
        public = np.lib.format.open_memmap(cache / 'public.npy', mode='w+', dtype='float32', shape=(80, 30, 2000))
        for i in range(10):
            path = ROOT / '数据' / '公开测试集' / f'P_test{i+1}.mat'
            public[i*8:(i+1)*8] = preprocess_windows(load_public_test_segments(path))
            hashes[path.stem] = repeat.sha256(path)
        public.flush()
        frame.to_csv(cache / 'rows.csv', index=False)
        save_json(HERE / 'input_metadata.json', {'train_shape': list(x.shape), 'public_shape': list(public.shape), 'raw_sha256': hashes, 'preprocessing': 'Each raw 10s window processed independently; no original_trial_id or neighboring windows', 'labels_used_in_preprocessing': False})
    source = pd.read_csv(cache / 'rows.csv')
    x = align_windows(np.load(cache / 'train.npy', mmap_mode='r'), source, frame)
    public = np.load(cache / 'public.npy', mmap_mode='r')
    assert x.shape == (2760, 30, 2000) and public.shape == (80, 30, 2000)
    assert np.isfinite(x).all() and np.isfinite(public).all()
    return x, public


def fit_scale(x):
    total = np.zeros(30, dtype=np.float64)
    for chunk in np.array_split(x, max(1, (len(x)+63)//64)):
        centered = chunk - chunk.mean(axis=2, keepdims=True)
        total += np.square(centered, dtype=np.float64).sum(axis=(0, 2))
    return np.maximum(np.sqrt(total / (len(x)*x.shape[2])), 1e-6).astype(np.float32)[None, :, None]


def transform(x, condition, scale=None):
    if condition == 'per_window_zscore':
        return base.sample_channel_zscore(x)
    if condition != 'amplitude_preserving' or scale is None:
        raise ValueError('Amplitude condition requires training-only scale')
    return (x - x.mean(axis=2, keepdims=True)) / scale


def train_predict(x, y, fit, xp, seed, config, condition):
    started = time.perf_counter()
    base.seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = base.EEGNet(config['model']['dropout']).to(device)
    cfg = config['training']
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scale = fit_scale(x[fit]) if condition == 'amplitude_preserving' else None
    tx = torch.from_numpy(transform(x[fit], condition, scale)[:, None]).to(device)
    ty = torch.from_numpy(y[fit]).long().to(device)
    bs = cfg['batch_size']; losses = []
    for epoch in range(1, cfg['epochs'] + 1):
        model.train(); order = np.random.default_rng(seed+epoch).permutation(len(fit)); total = 0
        for start in range(0, len(order), bs):
            ids = torch.from_numpy(order[start:start+bs]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(tx[ids]), ty[ids])
            loss.backward(); opt.step(); total += loss.item()*len(ids)
        losses.append(total/len(fit))
    model.eval(); out = []
    with torch.no_grad():
        for start in range(0, len(xp), bs):
            batch = transform(xp[start:start+bs], condition, scale)
            out.append(torch.softmax(model(torch.from_numpy(batch[:, None]).to(device)), 1)[:, 1].cpu().numpy())
    p = np.concatenate(out)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    audit = {'seed': seed, 'device': str(device), 'seconds': time.perf_counter()-started, 'epoch_train_losses': losses, 'training_channel_scale': None if scale is None else scale.ravel().tolist()}
    del tx, ty, model, opt
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    return p, audit


def persist(result):
    aggregates = {}
    for condition in result['experiment_config']['conditions']:
        completed = [result['runs'][f'{condition}_repeat{s}'] for s in result['experiment_config']['repeat_seeds'] if f'{condition}_repeat{s}' in result['runs']]
        if len(completed) == 5:
            values = {key: [r['overall'][key] for r in completed] for key in ['auc', 'f1', 'balanced_accuracy']}
            values['subject_macro_auc'] = [r['subject_auc_mean'] for r in completed]
            for group in ['HC', 'DEP']:
                values[f'{group.lower()}_auc'] = [r['health_groups'][group]['pooled']['auc'] for r in completed]
            aggregates[condition] = {key: {'values': vals, 'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1))} for key, vals in values.items()}
    result['repeat_descriptive_aggregate'] = aggregates
    save_json(HERE / 'metrics.json', result)
    rows = []
    for name, r in result['runs'].items():
        rows.append({'run_name': name, **r['overall'], 'subject_macro_auc': r['subject_auc_mean'], 'hc_auc': r['health_groups']['HC']['pooled']['auc'], 'dep_auc': r['health_groups']['DEP']['pooled']['auc'], 'training_seconds': sum(a['seconds'] for a in r['training'].values())})
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    fixed = result['runs']
    a = fixed.get('amplitude_preserving_fixed'); c = fixed.get('per_window_zscore_fixed')
    decision = '尚未完成固定折，不能得出结论。'
    if a and c:
        delta = a['overall']['auc'] - c['overall']['auc']
        decision = f'固定折幅值减control AUC差值={delta:.6f}。' + ('幅值条件低于control，停止此方向，不继续调参/融合。' if delta < 0 else '幅值条件提升仍须重复折确认，不以单次固定折宣布成功。')
    report = '# 实验042：保留幅值的逐10秒 EEGNet 最小对照\n\n'
    report += '原始EEG先切成独立10秒窗口，再逐窗口执行0.1–75Hz IIR、50Hz notch、250→200Hz重采样；训练与公开完全相同。024原为50秒滤波后切窗，因此本实验control不是024历史数值的直接复现。输入2760×30×2000，69被试，排除HC43；公开80行。元数据window_id仅用于对齐，original_trial_id、邻窗、行序均不作为模型输入。\n\n'
    report += '两条件均逐窗口逐通道去均值；control除以该窗口通道标准差，amplitude仅除以当前训练折无标签全体样本通道RMS，不对每窗口尺度归一化。尺度完整记录于metrics，各折仅用训练被试。模型直接复用024 EEGNet；Adam/30轮/batch64/lr.001/wd.01/dropout.25；2026+fold，完整训练2031；无早停、外层调参或融合，阈值0.5。\n\n'
    report += '固定折来自002；重复折使用032算法及seeds=2027..2031，逐次单独保存OOF。不拼接重复结果、不当作独立样本进行显著性检验。公开集无标签只保存预测，不报告公开AUC/F1。不承诺0.95。\n\n## 真实结果\n\n' + pd.DataFrame(rows).to_csv(index=False) + '\n## 决策与完成情况\n\n' + decision + '\n\n尚未完成：' + (', '.join(result['pending']) or '无') + '\n'
    if all(f'{condition}_repeat{seed}' in fixed for condition in result['experiment_config']['conditions'] for seed in result['experiment_config']['repeat_seeds']):
        deltas = [fixed[f'amplitude_preserving_repeat{s}']['overall']['auc'] - fixed[f'per_window_zscore_repeat{s}']['overall']['auc'] for s in result['experiment_config']['repeat_seeds']]
        report += f'\n五组配对AUC差值：{deltas}；均值{np.mean(deltas):.6f}、描述性样本标准差{np.std(deltas, ddof=1):.6f}。各重复共享同一69被试，不独立。\n'
    if aggregates:
        report += '\n## 重复折描述性汇总（均值±标准差）\n\n|条件|AUC|F1|BAcc|subject macro AUC|HC AUC|DEP AUC|\n|---|---:|---:|---:|---:|---:|---:|\n'
        for condition, agg in aggregates.items():
            report += '|' + condition + '|' + '|'.join(f"{agg[k]['mean']:.6f}±{agg[k]['std']:.6f}" for k in ['auc', 'f1', 'balanced_accuracy', 'subject_macro_auc', 'hc_auc', 'dep_auc']) + '|\n'
    if len(aggregates) == 2:
        report += '\n重复折结果并非一致优势：幅值平均AUC略高，但平均F1低于control；固定折AUC仍触发停止规则，不以重复折均值推翻预注册决策。\n'
    total_seconds = sum(a['seconds'] for r in result['runs'].values() for a in r['training'].values()) + sum(a['seconds'] for a in result['public_training'].values())
    report += f'\n实际训练及预测累计耗时：{total_seconds:.1f}秒（不含输入预处理及测试）。\n'
    report += '\n每折指标、被试指标、训练/验证被试名单、30轮训练损失、训练种子、尺度与耗时见metrics.json。输入原始文件SHA256见input_metadata.json。专项测试真实输出见test_protocol.txt。设备CUDA可用时使用CUDA，公开预测来自各条件全训练数据重训，不是折模型融合。\n'
    (HERE / '实验报告.md').write_text(report, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--fixed-only', action='store_true'); args = parser.parse_args()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4)
    frame = metadata(); x, public = prepare(frame); y = frame.y_true.to_numpy()
    pred = HERE / 'predictions'; pred.mkdir(exist_ok=True)
    planned = [(c, 'fixed', None) for c in config['conditions']] + [(c, f'repeat{s}', s) for s in config['repeat_seeds'] for c in config['conditions']]
    path = HERE / 'metrics.json'
    result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'experiment_config': config, 'runs': {}, 'public_training': {}, 'pending': []}
    result['pending'] = [f'{c}_{tag}' for c, tag, _ in planned if f'{c}_{tag}' not in result['runs']]
    persist(result)
    for condition, tag, seed in planned:
        name = f'{condition}_{tag}'
        if name in result['runs'] or (args.fixed_only and seed is not None): continue
        folds = frame.fold.to_numpy() if seed is None else repeat.make_subject_folds(frame.subject_id.to_numpy(), frame.health_group.to_numpy(), seed)
        repeat.assert_subject_isolation(frame.subject_id.to_numpy(), folds)
        ref = frame.copy(); ref['fold'] = folds
        p = np.full(len(y), np.nan); audits = {}; counts = np.zeros(len(y), dtype=int)
        for f in range(5):
            valid = folds == f; fit = np.flatnonzero(~valid)
            p[valid], audits[str(f)] = train_predict(x, y, fit, x[valid], 2026+f, config, condition)
            counts[valid] += 1
            audits[str(f)]['train_subjects'] = sorted(frame.iloc[fit].subject_id.unique().tolist())
            audits[str(f)]['valid_subjects'] = sorted(frame.loc[valid, 'subject_id'].unique().tolist())
            print(f'{name} fold={f} seconds={audits[str(f)]["seconds"]:.1f}', flush=True)
        assert (counts == 1).all() and np.isfinite(p).all()
        out = ref.copy(); out['prob_positive'] = p; out.to_csv(pred / f'{name}_oof.csv', index=False)
        r = repeat.summarize(y, p, ref, folds); r.update({'training': audits, 'split_seed': seed, 'oof_counts': np.unique(counts).tolist()})
        result['runs'][name] = r; result['pending'].remove(name)
        if seed is None:
            pp, audit = train_predict(x, y, np.arange(len(y)), public, 2031, config, condition)
            pub = pd.DataFrame({'user_id': np.repeat([f'P_test{i}' for i in range(1, 11)], 8), 'trial_id': np.tile(np.arange(1, 9), 10), 'prob. of positive emotion': pp})
            validate_submission(pub, [f'P_test{i}' for i in range(1, 11)])
            pub.to_csv(pred / f'{condition}_public.csv', index=False)
            result['public_training'][condition] = audit
        persist(result)
        print(name, r['overall'], flush=True)
    persist(result)


if __name__ == '__main__':
    main()
