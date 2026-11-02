import hashlib
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
ROOT = HERE.parents[2]
OLD = HERE.parent / '070_train_context_match_new_splits'
spec = importlib.util.spec_from_file_location('exp070_for085', OLD / 'run_experiment.py')
e070 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e070)
base = e070.base
load_data, contexts, scalars = e070.load_data, e070.contexts, e070.scalars
KEYS = e070.KEYS


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def validate_config(config):
    assert config['repeat_seeds'] == [2042, 2043, 2044, 2045, 2046]
    assert config['expected_model_fits'] == 150 and config['positions'] == [0, 1, 2, 3, 4]
    old = json.loads((OLD / 'config.json').read_text(encoding='utf8'))
    for key in ('hgb', 'outer_model_seed_base', 'threshold', 'excluded_subject', 'n_jobs', 'threads_per_job'):
        assert config[key] == old[key]


def splits_for(ref, fixed, config):
    return [('fixed002', fixed)] + [(f'repeat{s}', base.make_subject_folds(ref.subject.to_numpy(), ref.group.to_numpy(), s)) for s in config['repeat_seeds']]


def expert_train_mask(folds, positions, fold, expert):
    return (folds != fold) & (positions == expert)


def fit_one(a8, y, subjects, positions, name, folds, fold, expert, config):
    train = expert_train_mask(folds, positions, fold, expert)
    valid = folds == fold
    start = time.perf_counter()
    model = base.make_model(config, config['outer_model_seed_base'] + fold)
    model.fit(a8[train], y[train])
    p = model.predict_proba(a8[valid])[:, 1]
    return name, fold, expert, p, time.perf_counter() - start, int(train.sum()), int(subjects[train].size)


def summarize(frame, col):
    rr = frame.rename(columns={'subject': 'subject_id', 'group': 'health_group'})
    return base.summarize(frame.y.to_numpy(), frame[col].to_numpy(), rr, frame.fold.to_numpy())


def aggregate_runs(runs, config):
    out = {}
    for subset in ['all_rows'] + [str(i) for i in config['positions']]:
        out[subset] = {}
        for arm in ('control8', 'ensemble8', 'delta'):
            vals = []
            for seed in config['repeat_seeds']:
                r = runs[f'repeat{seed}'][subset]
                vals.append(scalars(r[arm]) if arm != 'delta' else r['delta'])
            out[subset][arm] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
    return out


def acceptance(aggregate, config):
    d = aggregate['all_rows']['delta']; a = config['acceptance']
    joint_nonnegative = [auc >= 0 and f1 >= 0 for auc, f1 in zip(d['auc']['values'], d['f1']['values'])]
    checks = {'mean_delta_auc': d['auc']['mean'] >= a['mean_delta_auc_min'], 'mean_delta_f1': d['f1']['mean'] >= a['mean_delta_f1_min'],
              'joint_nonnegative_splits': sum(joint_nonnegative) >= a['nonnegative_splits_min_each'],
              'hc_no_excess_drop': d['hc_auc']['mean'] >= -a['health_group_mean_auc_drop_max'],
              'dep_no_excess_drop': d['dep_auc']['mean'] >= -a['health_group_mean_auc_drop_max']}
    return {'accepted': all(checks.values()), 'checks': checks, 'joint_nonnegative_values': joint_nonnegative}


def main():
    start = time.perf_counter(); config = json.loads((HERE / 'config.json').read_text(encoding='utf8')); validate_config(config)
    raw, ref, fixed = load_data(config)
    y, subjects, positions = ref.y.to_numpy(), ref.subject.to_numpy(), ref.position.to_numpy()
    _, a8 = contexts(raw, subjects, positions)
    splits = splits_for(ref, fixed, config)
    for _, folds in splits:
        base.assert_subject_isolation(subjects, folds)
    inputs = [OLD / 'predictions' / f'{n}_oof.csv' for n, _ in splits]
    inputs += [(HERE / config[key]).resolve() for key in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')]
    sources = [HERE / n for n in ('run_experiment.py', 'test_085.py', 'config.json', 'test_protocol.txt')]
    sources += [OLD / n for n in ('run_experiment.py', 'test_070.py', 'config.json', 'test_protocol.txt')]
    dep_chain = [('069_train_context_match_replication', 'test_069.py'), ('068_train_context_match', 'test_068.py'), ('067_adaptation_context_size', 'test_067.py')]
    for dep_name, test_name in dep_chain:
        dep = OLD.parent / dep_name
        sources += [dep / n for n in ('run_experiment.py', test_name, 'config.json', 'test_protocol.txt')]
    sources += [Path(base.__file__), ROOT / 'src/emo_rec/normalization.py', ROOT / 'src/emo_rec/data.py', ROOT / 'src/emo_rec/submission.py', ROOT / 'pyproject.toml', ROOT / '数据' / '数据集说明文档.md']
    prov = {'inputs': {str(p): sha256(p) for p in inputs}, 'sources': {str(p): sha256(p) for p in sources}, 'raw_shape': list(raw.shape), 'public_arrays_loaded': False, 'model_fits': 150, 'n_jobs': config['n_jobs'], 'threads_per_job': config['threads_per_job'], 'training_rule': 'expert k trains outer-train subjects position k, predicts all validation rows; equal mean; no validation routing', 'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (HERE / 'input_metadata').mkdir(exist_ok=True); (HERE / 'predictions').mkdir(exist_ok=True)
    for n, folds in splits: ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{n}.csv', index=False)
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    jobs = [(n, folds, fold, expert) for n, folds in splits for fold in range(5) for expert in config['positions']]
    with threadpool_limits(limits=1):
        prov['active_threadpools'] = threadpool_info()
        results = Parallel(n_jobs=config['n_jobs'], prefer='threads')(delayed(fit_one)(a8, y, subjects, positions, n, folds, fold, expert, config) for n, folds, fold, expert in jobs)
    runs, summary_rows, delta_rows = {}, [], []
    for n, folds in splits:
        control = pd.read_csv(OLD / 'predictions' / f'{n}_oof.csv')
        for col in ('row_index', 'position', 'subject', 'group', 'trial', 'y'): np.testing.assert_array_equal(ref[col], control[col])
        np.testing.assert_array_equal(folds, control['fold'] if 'fold' in control else pd.read_csv(OLD / 'input_metadata' / f'folds_{n}.csv').fold)
        frame = ref.assign(fold=folds, p_control8=control.p_train8.to_numpy(), **{f'p_expert{k}': np.nan for k in config['positions']}, p_ensemble8=np.nan)
        for nn, fold, expert, p, *_ in results:
            if nn == n: frame.loc[folds == fold, f'p_expert{expert}'] = p
        expert_cols = [f'p_expert{k}' for k in config['positions']]
        assert np.isfinite(frame[expert_cols]).all().all(); frame['p_ensemble8'] = frame[expert_cols].mean(axis=1)
        frame.to_csv(HERE / 'predictions' / f'{n}_oof.csv', index=False)
        runs[n] = {}
        for subset in ['all_rows'] + [str(i) for i in config['positions']]:
            q = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            runs[n][subset] = {'control8': summarize(q, 'p_control8'), 'ensemble8': summarize(q, 'p_ensemble8')}
            summary_rows += [{'split': n, 'position': subset, 'arm': arm, **scalars(runs[n][subset][arm])} for arm in ('control8', 'ensemble8')]
            delta_rows.append({'split': n, 'position': subset, **{k: scalars(runs[n][subset]['ensemble8'])[k] - scalars(runs[n][subset]['control8'])[k] for k in KEYS}})
    # Store deltas as metric-shaped records for deterministic aggregation.
    for n in runs:
        for subset in runs[n]: runs[n][subset]['delta'] = {k: scalars(runs[n][subset]['ensemble8'])[k] - scalars(runs[n][subset]['control8'])[k] for k in KEYS}
    aggregate = aggregate_runs(runs, config); decision = acceptance(aggregate, config)
    prov['elapsed_seconds'] = time.perf_counter() - start
    metrics = {'runs': runs, 'aggregate': aggregate, 'acceptance': decision, 'fit_seconds': [{'split': r[0], 'fold': r[1], 'expert': r[2], 'seconds': r[4], 'n_train': r[5]} for r in results], 'provenance': prov}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8'); (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    pd.DataFrame(summary_rows).to_csv(HERE / 'summary.csv', index=False); pd.DataFrame(delta_rows).to_csv(HERE / 'paired_delta.csv', index=False)
    lines = ['# 实验085：位置分层五专家等权集成', '', '冻结协议见 `test_protocol.txt`。复用070的720维a8特征与HGB；每个fold五专家各仅用外层训练被试的同一position 8行/人训练，随后预测全部验证行，五概率等权平均，不使用验证位置路由。', '', '|split|control AUC|control F1|ensemble AUC|ensemble F1|ΔAUC|ΔF1|HC ΔAUC|DEP ΔAUC|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in delta_rows:
        if r['position'] == 'all_rows':
            c = runs[r['split']]['all_rows']['control8']['overall']; e = runs[r['split']]['all_rows']['ensemble8']['overall']
            lines.append(f"|{r['split']}|{c['auc']:.6f}|{c['f1']:.6f}|{e['auc']:.6f}|{e['f1']:.6f}|{r['auc']:+.6f}|{r['f1']:+.6f}|{r['hc_auc']:+.6f}|{r['dep_auc']:+.6f}|")
    d = aggregate['all_rows']['delta']; lines += ['', f"五重复平均ensemble AUC={aggregate['all_rows']['ensemble8']['auc']['mean']:.6f}，F1={aggregate['all_rows']['ensemble8']['f1']['mean']:.6f}；平均control AUC={aggregate['all_rows']['control8']['auc']['mean']:.6f}，F1={aggregate['all_rows']['control8']['f1']['mean']:.6f}；平均ΔAUC={d['auc']['mean']:+.6f}，平均ΔF1={d['f1']['mean']:+.6f}；接受：{'达到' if decision['accepted'] else '未达到'}。", '', '表中同时报告控制/处理绝对AUC与F1；指标为真实训练数据OOF。fixed002为控制复现，不纳入接受汇总。五位置仅是已知训练结构的8窗上下文模拟，不宣称公开测试位置可知。', '文献动机：[Breiman, Bagging Predictors](https://www.stat.berkeley.edu/~breiman/bagging.pdf)。本实验是等权多专家集成，不等同bootstrap，也不保证改善。', 'HGB无GPU实现，使用CPU例外n_jobs=10且全局线程限制1；未读取公开数组/标签。完整输入、递归源码hash及AUC/F1复算信息见metrics.json与provenance.json。']
    (HERE / '实验报告.md').write_text('\n'.join(lines) + '\n', encoding='utf8')

if __name__ == '__main__': main()
