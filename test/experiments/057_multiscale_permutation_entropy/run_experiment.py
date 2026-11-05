import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('base032', HERE.parent / '032_repeated_subject_folds/run_experiment.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
METHODS = ('control_600', 'pe_only', 'plus_pe')
KEYS = ('auc', 'f1', 'balanced_accuracy', 'subject_auc_mean', 'hc_auc', 'dep_auc', 'worst_subject_auc')
ALIGN_KEYS = ('subject_ids', 'health_groups', 'original_trial_ids', 'y_train', 'public_user_ids', 'public_trial_ids')


def permutation_entropy(x, order, delay):
    x = np.asarray(x, dtype=float)
    n = x.shape[-1] - (order - 1) * delay
    if order < 2 or delay < 1 or n < 1:
        raise ValueError('invalid ordinal embedding')
    embedded = np.stack([x[..., i*delay:i*delay+n] for i in range(order)], axis=-1)
    ranks = np.argsort(embedded, axis=-1, kind='stable')
    codes = np.sum(ranks * (order ** np.arange(order)), axis=-1)
    out = np.empty(codes.shape[:-1])
    for index in np.ndindex(out.shape):
        _, counts = np.unique(codes[index], return_counts=True)
        p = counts / n
        out[index] = -np.sum(p * np.log(p)) / math.log(math.factorial(order))
    return out


def pe_features(segments, config):
    x = np.asarray(segments, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (30, config['sampling_rate'] * config['window_seconds']):
        raise ValueError('expected 10-second windows (n, 30, 2500)')
    if not np.isfinite(x).all():
        raise ValueError('nonfinite EEG')
    pc = config['pe']
    x = resample_poly(x, 1, pc['downsample_factor'], axis=-1,
                      window=tuple(pc['resample_window']), padtype=pc['padtype'])
    blocks = []
    for scale in pc['scales']:
        length = x.shape[-1] // scale
        coarse = x[..., :length*scale].reshape(*x.shape[:-1], length, scale).mean(axis=-1)
        blocks.append(permutation_entropy(coarse, pc['embedding_order'], pc['delay']))
    return np.concatenate(blocks, axis=1)


def build_cache(config, path):
    from emo_rec.data import load_train_segments, load_public_test_segments
    from emo_rec.features import CHANNEL_NAMES
    feats, ys, subs, groups, trials = [], [], [], [], []
    files = sorted((ROOT / '数据/训练集').rglob('*.mat'))
    public_files = sorted((ROOT / '数据/公开集/公开测试集').glob('*.mat'), key=lambda p: int(p.stem.replace('P_test', '')))
    assert len(files) == 70 and len(public_files) == 10
    for i, f in enumerate(files, 1):
        seg, y, t = load_train_segments(f)
        feats.append(pe_features(seg, config)); ys.append(y); trials.append(t)
        subs.extend([f.stem]*len(seg)); groups.extend(['DEP' if f.parent.name == '抑郁症患者' else 'HC']*len(seg))
        print(f'PE {i}/{len(files)} {f.stem}', flush=True)
    pub, pu, pt = [], [], []
    for f in public_files:
        v = pe_features(load_public_test_segments(f), config)
        pub.append(v); pu.extend([f.stem]*len(v)); pt.extend(np.arange(1, 9))
    names = [f'pe_scale{s}_{ch}' for s in config['pe']['scales'] for ch in CHANNEL_NAMES]
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, X_train=np.concatenate(feats), y_train=np.concatenate(ys), subject_ids=np.asarray(subs),
                        health_groups=np.asarray(groups), original_trial_ids=np.concatenate(trials), feature_names=np.asarray(names),
                        X_public=np.concatenate(pub), public_user_ids=np.asarray(pu), public_trial_ids=np.asarray(pt),
                        config_sha256=np.asarray(base.sha256(HERE / 'config.json')))
    metadata = [{'path': str(f.relative_to(ROOT)), 'bytes': f.stat().st_size, 'sha256': base.sha256(f)} for f in files + public_files]
    (HERE / 'input_metadata/raw_inputs.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')


def assert_alignment(a, b):
    for key in ALIGN_KEYS:
        if not np.array_equal(a[key], b[key]):
            raise RuntimeError(f'cache alignment mismatch: {key}')


def load_data(config):
    from emo_rec.normalization import normalize_by_subject
    s = base.load_cache((HERE / config['spectral_cache']).resolve())
    t = base.load_cache((HERE / config['temporal_cache']).resolve())
    assert_alignment(s, t)
    path = HERE / config['pe_cache']
    if not path.exists():
        build_cache(config, path)
    p = base.load_cache(path)
    assert str(p['config_sha256']) == base.sha256(HERE / 'config.json')
    assert_alignment(s, p)
    keep = s['subject_ids'] != config['excluded_subject']
    subs, pub = s['subject_ids'][keep], s['public_user_ids']
    assert not set(subs) & set(pub) and len(np.unique(subs)) == 69
    sm = np.asarray([str(n).startswith('log_bandpower_') for n in s['feature_names']])
    assert sm.sum() == 150 and t['X_train'].shape[1] == 450
    z = lambda a, ids: normalize_by_subject(a, ids, 'zscore')
    X = np.column_stack([z(s['X_train'][keep][:, sm], subs), z(t['X_train'][keep], subs)])
    XP = np.column_stack([z(s['X_public'][:, sm], pub), z(t['X_public'], pub)])
    P, PP = z(p['X_train'][keep], subs), z(p['X_public'], pub)
    ref = pd.read_csv((HERE / config['fixed_reference']).resolve())
    for col, arr in [('subject_id', subs), ('health_group', s['health_groups'][keep]), ('y_true', s['y_train'][keep]), ('original_trial_id', s['original_trial_ids'][keep])]:
        np.testing.assert_array_equal(ref[col].to_numpy(), arr)
    assert X.shape == (2760, 600) and XP.shape == (80, 600) and P.shape == (2760, 120) and PP.shape == (80, 120)
    assert all(np.isfinite(a).all() for a in (X, XP, P, PP))
    return np.column_stack([X, P]), np.column_stack([XP, PP]), ref[['subject_id', 'health_group', 'original_trial_id', 'fold', 'y_true']], s, p


def column_sets():
    return {'control_600': np.arange(600), 'pe_only': np.arange(600, 720), 'plus_pe': np.arange(720)}


def fit_predict(A, y, train, AP, config, seed):
    out = {}
    with threadpool_limits(limits=1):
        for method, cols in column_sets().items():
            model = base.make_model(config, seed)
            model.fit(A[train][:, cols], y[train])
            out[method] = model.predict_proba(AP[:, cols])[:, 1]
    return out


def job(A, y, name, folds, fold, config):
    return name, fold, fit_predict(A, y, folds != fold, A[folds == fold], config, config['outer_model_seed_base'] + fold)


def scalars(r):
    return {**{k: r['overall'][k] for k in KEYS[:3]}, 'subject_auc_mean': r['subject_auc_mean'],
            'hc_auc': r['health_groups']['HC']['pooled']['auc'], 'dep_auc': r['health_groups']['DEP']['pooled']['auc'],
            'worst_subject_auc': min(v['auc'] for v in r['subjects'])}


def accepts(delta, fixed_auc, config):
    a = config['acceptance']
    return bool(delta['auc'] >= a['repeat_delta_auc_min'] and delta['f1'] >= a['repeat_delta_f1_min']
                and delta['dep_auc'] >= a['repeat_delta_dep_auc_min'] and fixed_auc > a['fixed_delta_auc_strict_min'])


class Tee:
    def __init__(self, path):
        self.file = path.open('w', encoding='utf8'); self.stdout = sys.stdout
    def write(self, text):
        self.stdout.write(text); self.file.write(text); self.file.flush()
    def flush(self):
        self.stdout.flush(); self.file.flush()


def main():
    from emo_rec.submission import validate_submission
    sys.stdout = Tee(HERE / 'run_output.txt')
    start = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    for d in ('predictions', 'input_metadata'):
        (HERE / d).mkdir(exist_ok=True)
    A, AP, ref, s, p = load_data(config)
    y, subs, groups = ref.y_true.to_numpy(), ref.subject_id.to_numpy(), ref.health_group.to_numpy()
    feature_seconds = time.perf_counter() - start
    print(f'data ready train={A.shape} public={AP.shape} elapsed={feature_seconds:.1f}s', flush=True)
    prov = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
            'inputs': {k: {'path': str((HERE / config[k]).resolve()), 'sha256': base.sha256((HERE / config[k]).resolve())} for k in ('spectral_cache', 'temporal_cache', 'fixed_reference', 'pe_cache')},
            'config_sha256': base.sha256(HERE / 'config.json'), 'script_sha256': base.sha256(Path(__file__)),
            'train_shape': list(A.shape), 'public_shape': list(AP.shape), 'alignment_checked': list(ALIGN_KEYS),
            'subject_isolation': True, 'subject_zscore': True, 'public_metrics': False, 'feature_names': p['feature_names'].tolist()}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    splits = [('fixed002', ref.fold.to_numpy())] + [(f'repeat{seed}', base.make_subject_folds(subs, groups, seed)) for seed in config['repeat_seeds']]
    for name, folds in splits:
        assert set(folds) == set(range(5))
        base.assert_subject_isolation(subs, folds)
        ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
    sys.path.insert(0, str(HERE))
    import run_experiment as importable
    results = Parallel(n_jobs=config['n_jobs'])(delayed(importable.job)(A, y, name, folds, f, config) for name, folds in splits for f in range(5))
    pred = {name: {m: np.full(len(y), np.nan) for m in METHODS} for name, _ in splits}
    fmap = dict(splits)
    for name, fold, out in results:
        for m in METHODS:
            pred[name][m][fmap[name] == fold] = out[m]
    runs, rows, checks = {m: {} for m in METHODS}, [], {}
    for name, folds in splits:
        rr = ref.assign(fold=folds)
        for m in METHODS:
            pr = pred[name][m]
            assert np.isfinite(pr).all() and ((pr >= 0) & (pr <= 1)).all()
            rr.assign(prob_positive=pr).to_csv(HERE / 'predictions' / f'{m}_{name}_oof.csv', index=False)
            result = base.summarize(y, pr, rr, folds)
            runs[m][name] = result
            rows.append({'method': m, 'split': name, **scalars(result)})
        old = pd.read_csv(HERE.parent / '056_theta_beta_pac/predictions' / f'control_600_{name}_oof.csv')
        np.testing.assert_array_equal(old[rr.columns].to_numpy(), rr.to_numpy())
        checks[name] = float(np.max(np.abs(old.prob_positive.to_numpy() - pred[name]['control_600'])))
        assert checks[name] < 1e-12
    public = fit_predict(A, y, np.ones(len(y), bool), AP, config, config['outer_model_seed_base'])
    for m in METHODS:
        frame = pd.DataFrame({'user_id': s['public_user_ids'], 'trial_id': s['public_trial_ids'], 'prob. of positive emotion': public[m]})
        validate_submission(frame, [f'P_test{i}' for i in range(1, 11)])
        frame.to_csv(HERE / 'predictions' / f'{m}_public.csv', index=False)
    aggregate = {}
    for m in METHODS:
        vals = [scalars(runs[m][f'repeat{seed}']) for seed in config['repeat_seeds']]
        aggregate[m] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
        rows.append({'method': m, 'split': 'repeat_mean', **{k: aggregate[m][k]['mean'] for k in KEYS}})
    delta = {m: {'repeat_mean': {k: aggregate[m][k]['mean'] - aggregate['control_600'][k]['mean'] for k in KEYS},
                 'fixed002': {k: scalars(runs[m]['fixed002'])[k] - scalars(runs['control_600']['fixed002'])[k] for k in KEYS},
                 'repeat_paired_auc': [a-b for a, b in zip(aggregate[m]['auc']['values'], aggregate['control_600']['auc']['values'])]} for m in METHODS[1:]}
    decision = {m: accepts(d['repeat_mean'], d['fixed002']['auc'], config) for m, d in delta.items()}
    elapsed = time.perf_counter() - start
    metrics = {'config': config, 'provenance': prov, 'runs': runs, 'aggregate': aggregate, 'deltas_vs_control': delta,
               'decision_accept': decision, 'control_reproduction_max_abs_diff_vs_056': checks,
               'feature_loading_seconds': feature_seconds, 'elapsed_seconds': elapsed}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf8')
    table = pd.DataFrame(rows)
    table.to_csv(HERE / 'summary.csv', index=False)
    lines = ['# 实验057：单窗多尺度排列熵', '', '固定协议见test_protocol.txt；全部配置在首次训练前确定。公开集仅输出预测，未读取公开标签或计算公开指标。', '',
             '|方法|五重复AUC均值±SD|F1均值|DEP AUC均值|', '|---|---:|---:|---:|']
    for m in METHODS:
        a = aggregate[m]
        lines.append(f"|{m}|{a['auc']['mean']:.6f}±{a['auc']['std']:.6f}|{a['f1']['mean']:.6f}|{a['dep_auc']['mean']:.6f}|")
    lines += ['', '|方法|ΔAUC|ΔF1|ΔDEP AUC|fixed002 ΔAUC|达线|', '|---|---:|---:|---:|---:|---|']
    for m, d in delta.items():
        v = d['repeat_mean']
        lines.append(f"|{m}|{v['auc']:+.6f}|{v['f1']:+.6f}|{v['dep_auc']:+.6f}|{d['fixed002']['auc']:+.6f}|{decision[m]}|")
    lines += ['', '未达线方法停止，不调参。', f'真实耗时：{elapsed:.2f}秒；特征加载/提取和输入哈希阶段：{feature_seconds:.2f}秒。',
              f'与056 control OOF最大差：{max(checks.values()):.3g}。', '完整逐折/逐被试指标见metrics.json；18份OOF和3份公开预测见predictions。']
    (HERE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(table.to_string(index=False))
    print(json.dumps({'delta': delta, 'decision': decision, 'elapsed_seconds': elapsed}, indent=2), flush=True)


if __name__ == '__main__':
    main()
