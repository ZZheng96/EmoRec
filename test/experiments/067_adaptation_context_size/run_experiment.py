import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('base032', HERE.parent / '032_repeated_subject_folds/run_experiment.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from emo_rec.normalization import normalize_by_subject

OLD = HERE.parent / '057_multiscale_permutation_entropy'
KEYS = ('auc', 'f1', 'balanced_accuracy', 'subject_auc_mean', 'hc_auc', 'dep_auc', 'worst_subject_auc')
TRAIN_KEYS = ('X_train', 'subject_ids', 'health_groups', 'original_trial_ids', 'y_train', 'feature_names')


def load_train_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in TRAIN_KEYS}


def positions(reference):
    p = reference.groupby(['subject', 'y', 'trial'], sort=False).cumcount().to_numpy()
    counts = reference.groupby(['subject', 'y', 'trial']).size()
    assert (counts == 5).all() and len(counts) == 69 * 8
    for pos in range(5):
        r = reference[p == pos]
        assert (r.groupby('subject').size() == 8).all()
        assert (r.groupby(['subject', 'y']).size() == 4).all()
    assert np.array_equal(np.sort(np.concatenate([np.flatnonzero(p == i) for i in range(5)])), np.arange(len(reference)))
    return p


def load_data(config):
    s, t, pe = [load_train_cache((HERE / config[k]).resolve()) for k in ('spectral_cache', 'temporal_cache', 'pe_cache')]
    for other in (t, pe):
        for key in TRAIN_KEYS[1:5]:
            np.testing.assert_array_equal(s[key], other[key])
    keep = s['subject_ids'] != config['excluded_subject']
    mask = np.asarray([str(n).startswith('log_bandpower_') for n in s['feature_names']])
    assert mask.sum() == 150 and t['X_train'].shape[1] == 450 and pe['X_train'].shape[1] == 120
    raw = np.column_stack([s['X_train'][keep][:, mask], t['X_train'][keep], pe['X_train'][keep]])
    ref = pd.DataFrame({'row_index': np.arange(keep.sum()), 'subject': s['subject_ids'][keep], 'group': s['health_groups'][keep], 'trial': s['original_trial_ids'][keep], 'y': s['y_train'][keep]})
    fixed = pd.read_csv(HERE / config['fixed_reference'])
    for col, old in [('subject', 'subject_id'), ('group', 'health_group'), ('trial', 'original_trial_id'), ('y', 'y_true')]:
        np.testing.assert_array_equal(ref[col], fixed[old])
    ref['position'] = positions(ref)
    assert raw.shape == (2760, 720) and np.isfinite(raw).all()
    assert ref.subject.nunique() == 69 and (ref.groupby('subject').size() == 40).all()
    return raw, ref, fixed.fold.to_numpy()


def normalize(raw, subjects):
    # Preserve 057's separate-block arithmetic and memory layout exactly.
    return np.column_stack([normalize_by_subject(raw[:, a:b], subjects, 'zscore') for a, b in ((0, 150), (150, 600), (600, 720))])


def contexts(raw, subjects, pos):
    a40 = normalize(raw, subjects)
    a8 = np.empty_like(a40)
    for i in range(5):
        ix = pos == i
        a8[ix] = normalize(raw[ix], subjects[ix])
    return a40, a8


def job(a40, a8, y, name, folds, fold, config):
    start = time.perf_counter()
    train, valid = folds != fold, folds == fold
    with threadpool_limits(limits=config['threads_per_job']):
        model = base.make_model(config, config['outer_model_seed_base'] + fold)
        model.fit(a40[train], y[train])
        p40 = model.predict_proba(a40[valid])[:, 1]
        p8 = model.predict_proba(a8[valid])[:, 1]
    return name, fold, p40, p8, time.perf_counter() - start


def summarize(frame, context):
    rr = frame.rename(columns={'subject': 'subject_id', 'group': 'health_group'})
    return base.summarize(frame.y.to_numpy(), frame[f'p_context{context}'].to_numpy(), rr, frame.fold.to_numpy())


def scalars(r):
    return {**{k: r['overall'][k] for k in KEYS[:3]}, 'subject_auc_mean': r['subject_auc_mean'],
            'hc_auc': r['health_groups']['HC']['pooled']['auc'], 'dep_auc': r['health_groups']['DEP']['pooled']['auc'],
            'worst_subject_auc': min(v['auc'] for v in r['subjects'])}


def main():
    start = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    old_config = json.loads((OLD / 'config.json').read_text(encoding='utf8'))
    for key in ('hgb', 'outer_model_seed_base', 'repeat_seeds', 'threshold', 'excluded_subject'):
        assert config[key] == old_config[key]
    assert (HERE / 'test_protocol.txt').exists()
    for folder in ('predictions', 'input_metadata'):
        (HERE / folder).mkdir(exist_ok=True)
    raw, ref, fixed = load_data(config)
    subjects, y = ref.subject.to_numpy(), ref.y.to_numpy()
    a40, a8 = contexts(raw, subjects, ref.position.to_numpy())
    splits = [('fixed002', fixed)] + [(f'repeat{s}', base.make_subject_folds(subjects, ref.group.to_numpy(), s)) for s in config['repeat_seeds']]
    inputs = [(HERE / config[k]).resolve() for k in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')]
    inputs += [OLD / 'predictions' / f'plus_pe_{name}_oof.csv' for name, _ in splits]
    sources = [HERE / n for n in ('run_experiment.py', 'test_067.py', 'config.json', 'test_protocol.txt')]
    sources += [OLD / 'run_experiment.py', OLD / 'config.json', Path(base.__file__), ROOT / 'src/emo_rec/normalization.py', ROOT / 'src/emo_rec/data.py', ROOT / '数据/数据集说明文档.md']
    prov = {'inputs': {str(p): base.sha256(p) for p in inputs}, 'sources': {str(p): base.sha256(p) for p in sources},
            'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
            'raw_shape': list(raw.shape), 'public_arrays_loaded': False, 'model_fits': 30, 'cpu_only': 'sklearn HGB has no GPU implementation', 'n_jobs': config['n_jobs'], 'threads_per_job': 1}
    for name, folds in splits:
        base.assert_subject_isolation(subjects, folds)
        ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    loading_seconds = time.perf_counter() - start
    print(f'Frozen protocol; raw={raw.shape}; 30 fits CPU n_jobs=10 threads=1', flush=True)
    with threadpool_limits(limits=config['threads_per_job']):
        results = Parallel(n_jobs=config['n_jobs'], prefer='threads')(delayed(job)(a40, a8, y, name, folds, fold, config) for name, folds in splits for fold in range(5))
    runs, rows, checks = {}, [], {}
    for name, folds in splits:
        frame = ref.assign(fold=folds, p_context40=np.nan, p_context8=np.nan)
        for n, fold, p40, p8, seconds in results:
            if n == name:
                frame.loc[folds == fold, 'p_context40'] = p40
                frame.loc[folds == fold, 'p_context8'] = p8
        old = pd.read_csv(OLD / 'predictions' / f'plus_pe_{name}_oof.csv')
        for col, original in [('subject', 'subject_id'), ('group', 'health_group'), ('trial', 'original_trial_id'), ('y', 'y_true'), ('fold', 'fold')]:
            np.testing.assert_array_equal(frame[col], old[original])
        checks[name] = float(np.max(np.abs(frame.p_context40 - old.prob_positive)))
        assert checks[name] <= 1e-12, checks
        assert np.isfinite(frame[['p_context40', 'p_context8']]).all().all()
        frame = frame[['row_index', 'position', 'subject', 'group', 'trial', 'y', 'fold', 'p_context40', 'p_context8']]
        frame.to_csv(HERE / 'predictions' / f'{name}_oof.csv', index=False)
        runs[name] = {}
        for subset in ['all_rows'] + [str(i) for i in range(5)]:
            selected = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            runs[name][subset] = {}
            for context in (40, 8):
                r = summarize(selected, context)
                runs[name][subset][str(context)] = r
                rows.append({'split': name, 'position': subset, 'context': context, **scalars(r)})
    aggregate = {}
    for subset in ['all_rows'] + [str(i) for i in range(5)]:
        aggregate[subset] = {}
        for context in ('40', '8', 'delta8_minus40'):
            vals = []
            for seed in config['repeat_seeds']:
                r = runs[f'repeat{seed}'][subset]
                vals.append(scalars(r[context]) if context != 'delta8_minus40' else {k: scalars(r['8'])[k] - scalars(r['40'])[k] for k in KEYS})
            aggregate[subset][context] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
    elapsed = time.perf_counter() - start
    metrics = {'runs': runs, 'aggregate': aggregate, 'context40_max_abs_diff_vs057': checks, 'elapsed_seconds': elapsed, 'loading_seconds': loading_seconds,
               'fit_seconds': [{'split': r[0], 'fold': r[1], 'seconds': r[4]} for r in results], 'provenance': prov}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    lines = ['# 实验067：归一化上下文数量敏感性', '', '固定协议见test_protocol.txt。raw720；排HC43；30次训练；训练始终40窗；验证同一行context40对比context8。', '',
             '|位置|context40 AUC均值±SD|context8 AUC均值±SD|配对ΔAUC均值|context40 F1|context8 F1|HC ΔAUC|DEP ΔAUC|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for subset, r in aggregate.items():
        a, b, d = r['40'], r['8'], r['delta8_minus40']
        lines.append(f"|{subset}|{a['auc']['mean']:.6f}±{a['auc']['std']:.6f}|{b['auc']['mean']:.6f}±{b['auc']['std']:.6f}|{d['auc']['mean']:+.6f}|{a['f1']['mean']:.6f}|{b['f1']['mean']:.6f}|{d['hc_auc']['mean']:+.6f}|{d['dep_auc']['mean']:+.6f}|")
    lines += ['', f'运行耗时{elapsed:.2f}秒；context40六份OOF与057最大差{max(checks.values()):.3g}。逐行输出、逐折/逐被试指标、配对五重复差值及输入/源码哈希已保存。',
              'HGB没有GPU实现，CPU并行10任务，每任务限制1线程。不读取公开数据/公开数组，不生成公开预测。',
              '五位置全部遍历，不挑最佳；每位置每人8窗4/4，row_index完整覆盖一次。五位置非独立队列，all_rows仅为五位置预测拼接；不做超参搜索或显著性宣称。这不是实际测试分布模拟。', '',
              '## 文献动机与边界',
              '- [Unraveling BN](https://arxiv.org/html/2312.09486v3/)',
              '- [UnMix-TNS](https://arxiv.org/html/2401.08328v2)',
              '- [SAR](https://arxiv.org/html/2302.12400v1)',
              '上述文献的共同动机是测试统计受batch大小/组成影响，但研究深网BN/TTA，并非当前静态EEG特征zscore+HGB，不能外推效应大小或方向。035仅spectral150且验证40窗，不能替代本诊断。']
    (HERE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({'elapsed_seconds': elapsed, 'reproduction': checks, 'all_rows': aggregate['all_rows']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
