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
from threadpoolctl import threadpool_limits, threadpool_info

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / '068_train_context_match'
spec = importlib.util.spec_from_file_location('exp068_dependency', OLD / 'run_experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
base = e.base
load_data = e.e.load_data
contexts = e.e.contexts
scalars = e.e.scalars
KEYS = e.KEYS


def job(a40, a8, y, name, folds, fold, context, config):
    start = time.perf_counter()
    train, valid = folds != fold, folds == fold
    x = a40 if context == 40 else a8
    model = base.make_model(config, config['outer_model_seed_base'] + fold)
    model.fit(x[train], y[train])
    p = model.predict_proba(a8[valid])[:, 1]
    return name, fold, context, p, time.perf_counter() - start, int(train.sum())


def summarize(frame, context):
    rr = frame.rename(columns={'subject': 'subject_id', 'group': 'health_group'})
    return base.summarize(frame.y.to_numpy(), frame[f'p_train{context}'].to_numpy(), rr, frame.fold.to_numpy())


def aggregate_runs(runs, config):
    aggregate = {}
    for subset in ['all_rows'] + [str(i) for i in config['positions']]:
        aggregate[subset] = {}
        for context in ('40', '8', 'delta8_minus40'):
            vals = []
            for seed in config['repeat_seeds']:
                r = runs[f'repeat{seed}'][subset]
                vals.append(scalars(r[context]) if context != 'delta8_minus40' else {k: scalars(r['8'])[k] - scalars(r['40'])[k] for k in KEYS})
            aggregate[subset][context] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
    return aggregate


def acceptance(aggregate, config):
    d = aggregate['all_rows']['delta8_minus40']
    a = config['acceptance']
    checks = {'mean_delta_auc': d['auc']['mean'] >= a['mean_delta_auc_min'],
              'mean_delta_f1': d['f1']['mean'] >= a['mean_delta_f1_min'],
              'auc_nonnegative': sum(v >= 0 for v in d['auc']['values']) >= a['nonnegative_splits_min_each'],
              'f1_nonnegative': sum(v >= 0 for v in d['f1']['values']) >= a['nonnegative_splits_min_each'],
              'hc_no_excess_drop': d['hc_auc']['mean'] >= -a['health_group_mean_auc_drop_max'],
              'dep_no_excess_drop': d['dep_auc']['mean'] >= -a['health_group_mean_auc_drop_max']}
    return {'accepted': all(checks.values()), 'checks': checks}


def main():
    start = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    old_config = json.loads((OLD / 'config.json').read_text(encoding='utf8'))
    for key in ('hgb', 'outer_model_seed_base', 'repeat_seeds', 'threshold', 'excluded_subject', 'positions', 'n_jobs', 'threads_per_job'):
        assert config[key] == old_config[key]
    assert config['expected_model_fits'] == 60 and config['validation_context'] == 8
    for folder in ('predictions', 'input_metadata'):
        (HERE / folder).mkdir(exist_ok=True)
    raw, ref, fixed = load_data(config)
    subjects, y = ref.subject.to_numpy(), ref.y.to_numpy()
    a40, a8 = contexts(raw, subjects, ref.position.to_numpy())
    splits = [('fixed002', fixed)] + [(f'repeat{s}', base.make_subject_folds(subjects, ref.group.to_numpy(), s)) for s in config['repeat_seeds']]
    inputs = [(OLD / config[k]).resolve() for k in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')]
    inputs += [OLD / 'predictions' / f'{name}_oof.csv' for name, _ in splits]
    sources = [HERE / n for n in ('run_experiment.py', 'test_069.py', 'config.json', 'test_protocol.txt')]
    sources += [OLD / n for n in ('run_experiment.py', 'test_068.py', 'config.json', 'test_protocol.txt')]
    sources += [Path(base.__file__), e.e.ROOT / 'src/emo_rec/normalization.py', e.e.ROOT / 'src/emo_rec/data.py', e.e.ROOT / 'src/emo_rec/submission.py', e.e.ROOT / 'pyproject.toml', e.e.ROOT / '数据/数据集说明文档.md']
    prov = {'inputs': {str(p): base.sha256(p) for p in inputs}, 'sources': {str(p): base.sha256(p) for p in sources},
            'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
            'platform': platform.platform(), 'raw_shape': list(raw.shape), 'public_arrays_loaded': False, 'model_fits': 60,
            'cpu_only': 'sklearn HGB has no GPU implementation; controlled CPU exception', 'n_jobs': 10, 'threads_per_job': 1,
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    for name, folds in splits:
        base.assert_subject_isolation(subjects, folds)
        ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    loading_seconds = time.perf_counter() - start
    print('Frozen protocol; 2760 rows; 60 fits CPU n_jobs=10 thread=1', flush=True)
    # One process-global limit encloses all threaded jobs; no racing nested restorations.
    with threadpool_limits(limits=1):
        prov['active_threadpools'] = threadpool_info()
        results = Parallel(n_jobs=10, prefer='threads')(delayed(job)(a40, a8, y, name, folds, fold, context, config) for name, folds in splits for fold in range(5) for context in (40, 8))
    runs, rows, checks, deltas = {}, [], {}, []
    for name, folds in splits:
        frame = ref.assign(fold=folds, p_train40=np.nan, p_train8=np.nan)
        for n, fold, context, p, seconds, n_train in results:
            if n == name:
                frame.loc[folds == fold, f'p_train{context}'] = p
        old = pd.read_csv(OLD / 'predictions' / f'{name}_oof.csv')
        for col in ('row_index', 'position', 'subject', 'group', 'trial', 'y', 'fold'):
            np.testing.assert_array_equal(frame[col], old[col])
        checks[name] = float(np.max(np.abs(frame.p_train40 - old.p_train40)))
        assert checks[name] <= 1e-12, checks
        assert np.isfinite(frame[['p_train40', 'p_train8']]).all().all()
        frame.to_csv(HERE / 'predictions' / f'{name}_oof.csv', index=False)
        runs[name] = {}
        for subset in ['all_rows'] + [str(i) for i in config['positions']]:
            selected = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            runs[name][subset] = {}
            for context in (40, 8):
                r = summarize(selected, context)
                runs[name][subset][str(context)] = r
                rows.append({'split': name, 'position': subset, 'train_context': context, 'validation_context': 8, **scalars(r)})
            deltas.append({'split': name, 'position': subset, **{k: scalars(runs[name][subset]['8'])[k] - scalars(runs[name][subset]['40'])[k] for k in KEYS}})
    aggregate = aggregate_runs(runs, config)
    decision = acceptance(aggregate, config)
    elapsed = time.perf_counter() - start
    metrics = {'runs': runs, 'aggregate': aggregate, 'acceptance': decision, 'control_max_abs_diff_vs068': checks,
               'elapsed_seconds': elapsed, 'loading_seconds': loading_seconds,
               'fit_seconds': [{'split': r[0], 'fold': r[1], 'train_context': r[2], 'seconds': r[4], 'n_train': r[5]} for r in results], 'provenance': prov}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(deltas).to_csv(HERE / 'paired_delta.csv', index=False)
    lines = ['# 实验069：训练上下文匹配复验', '', '冻结协议见test_protocol.txt。69被试2760训练行全部保留；双路60 fits；验证统一a8。',
             '五重复汇总不包含fixed002；Δ为训练a8减训练a40。', '',
             '|位置|控制AUC|匹配AUC|ΔAUC|控制F1|匹配F1|ΔF1|HC ΔAUC|DEP ΔAUC|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for subset, r in aggregate.items():
        a, b, d = r['40'], r['8'], r['delta8_minus40']
        lines.append(f"|{subset}|{a['auc']['mean']:.6f}|{b['auc']['mean']:.6f}|{d['auc']['mean']:+.6f}|{a['f1']['mean']:.6f}|{b['f1']['mean']:.6f}|{d['f1']['mean']:+.6f}|{d['hc_auc']['mean']:+.6f}|{d['dep_auc']['mean']:+.6f}|")
    lines += ['', '## 预注册接受判断', json.dumps(decision, ensure_ascii=False),
              '接受线：五重复all_rows平均ΔAUC和ΔF1均>=0.005；两指标各至少4/5 split非负；HC/DEP平均ΔAUC均>=-0.005。',
              '结果：' + ('达到' if decision['accepted'] else '未达到') + '预注册接受线。不能据此宣称公开提升。',
              '', '## 逐重复all_rows配对差值', '|split|ΔAUC|ΔF1|HC ΔAUC|DEP ΔAUC|', '|---|---:|---:|---:|---:|']
    for r in deltas:
        if r['position'] == 'all_rows':
            lines.append(f"|{r['split']}|{r['auc']:+.6f}|{r['f1']:+.6f}|{r['hc_auc']:+.6f}|{r['dep_auc']:+.6f}|")
    lines += ['', f'耗时{elapsed:.2f}秒；控制六份OOF对068 p_context8最大差{max(checks.values()):.3g}。',
              'sklearn HGB无GPU实现，CPU例外n_jobs=10/thread=1；未访问公开数组。所有输入和复用源码hash见input_metadata/provenance.json。',
              '分组使用已知训练视频结构，仅属诊断，不代表公开刺激/时间匹配；五位置不是独立样本，五重复复用同一队列；不宣称独立队列、显著性或公开提升。',
              '', '## 方法动机及相反机制',
              '- [Batch Renormalization](https://arxiv.org/abs/1702.03275)：上游核读原文摘要；small/non-iid batch及训练/推理激活差异提供方法动机。',
              '- [Ghost Noise for Regularizing Deep Neural Networks, AAAI2024](https://arxiv.org/abs/2305.17205)：上游核读原文摘要；小batch可能带来正则收益，同时引入train-test discrepancy，并非越大越好。',
              '两篇均为方法动机，而非EEG zscore+HGB效果证据；不外推成绩。',
              '含自身的ddof=0 n窗zscore（非退化特征）满足|z_i|<=sqrt(n-1)：其余n-1个中心化值和为-z_i，由Cauchy不等式得到该界。因此8窗/40窗不仅估计方差不同，支持范围也不同（sqrt(7) vs sqrt(39)）。统计量匹配有可检验机制，但不保证分类改善。',
              '完整逐折/逐被试指标、五重复样本SD和配对values见metrics.json；每位置逐split差值见paired_delta.csv。']
    (HERE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({'elapsed_seconds': elapsed, 'reproduction': checks, 'acceptance': decision, 'all_rows': aggregate['all_rows']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
