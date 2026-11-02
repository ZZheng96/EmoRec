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
OLD = HERE.parent / '079_regional_spectral_pooling'
spec = importlib.util.spec_from_file_location('exp079_mapping', OLD / 'run_experiment.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
e, base, DEP = old.e, old.base, old.DEP
REGIONS, CHANNELS, BANDS, FEATURE_NAMES = old.REGIONS, old.CHANNELS, old.BANDS, old.FEATURE_NAMES
KEYS, scalars = old.KEYS, old.scalars
IDENTITY = ('row_index', 'position', 'subject', 'group', 'trial', 'y', 'fold')


def validate_config(config):
    control = json.loads((DEP / 'config.json').read_text(encoding='utf8'))
    for k in ('hgb', 'outer_model_seed_base', 'repeat_seeds', 'threshold', 'excluded_subject', 'positions', 'n_jobs', 'threads_per_job'):
        assert config[k] == control[k]
    assert config['train_contexts'] == [8] and config['validation_context'] == 8
    assert config['expected_model_fits'] == 30
    assert config['fit_roles'] == ['region8']
    assert config['acceptance'] == {'mean_delta_auc_min': 0.005, 'mean_delta_f1_min': 0.005, 'nonnegative_splits_min_each': 4, 'health_group_mean_auc_drop_max': 0.005}


def load_data(config):
    raw, ref, fixed = e.load_data(config)
    with np.load((HERE / config['spectral_cache']).resolve(), allow_pickle=False) as cache:
        names = cache['feature_names']
    names = names[np.asarray([str(n).startswith('log_bandpower_') for n in names])]
    np.testing.assert_array_equal(names, [f'log_bandpower_{ch}_{b}' for ch in CHANNELS for b in BANDS])
    s = raw[:, :150].reshape(len(raw), 30, 5)
    pooled = np.column_stack([s[:, [CHANNELS.index(ch) for ch in chans], b].mean(axis=1) for chans in REGIONS.values() for b in range(5)])
    treatment = np.column_stack([pooled, raw[:, 150:]])
    assert treatment.shape == (2760, 595) and np.isfinite(treatment).all()
    return raw, treatment, ref, fixed


def region_context8(treatment, subjects, positions):
    result = np.empty_like(treatment)
    for position in range(5):
        ix = positions == position
        result[ix] = np.column_stack([old.normalize_by_subject(treatment[ix, a:b], subjects[ix], 'zscore') for a, b in ((0, 25), (25, 475), (475, 595))])
    return result


def job(x8, y, name, folds, fold, role, config):
    start = time.perf_counter()
    train, valid = folds != fold, folds == fold
    model = base.make_model(config, config['outer_model_seed_base'] + fold)
    model.fit(x8[train], y[train])
    p = model.predict_proba(x8[valid])[:, 1]
    return name, fold, role, p, time.perf_counter() - start, int(train.sum())


def summarize(frame, column):
    rr = frame.rename(columns={'subject': 'subject_id', 'group': 'health_group'})
    return base.summarize(frame.y.to_numpy(), frame[column].to_numpy(), rr, frame.fold.to_numpy())


def aggregate_runs(runs, config):
    aggregate = {}
    for subset in ['all_rows'] + [str(i) for i in config['positions']]:
        aggregate[subset] = {}
        for arm in ('control8', 'region8', 'delta_region8_minus_control8'):
            vals = []
            for seed in config['repeat_seeds']:
                r = runs[f'repeat{seed}'][subset]
                vals.append(scalars(r[arm]) if arm != 'delta_region8_minus_control8' else {k: scalars(r['region8'])[k] - scalars(r['control8'])[k] for k in KEYS})
            aggregate[subset][arm] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
    return aggregate


def acceptance(aggregate, config):
    d = aggregate['all_rows']['delta_region8_minus_control8']
    a = config['acceptance']
    checks = {'mean_delta_auc': d['auc']['mean'] >= a['mean_delta_auc_min'], 'mean_delta_f1': d['f1']['mean'] >= a['mean_delta_f1_min'], 'joint_nonnegative': sum(x >= 0 and y >= 0 for x, y in zip(d['auc']['values'], d['f1']['values'])) >= a['nonnegative_splits_min_each'], 'hc_no_excess_drop': d['hc_auc']['mean'] >= -a['health_group_mean_auc_drop_max'], 'dep_no_excess_drop': d['dep_auc']['mean'] >= -a['health_group_mean_auc_drop_max']}
    return {'accepted': all(checks.values()), 'checks': checks}


def main():
    start = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    validate_config(config)
    assert (HERE / 'pytest_before.txt').exists()
    for folder in ('predictions', 'input_metadata', 'cache'):
        (HERE / folder).mkdir(exist_ok=True)
    raw, treatment, ref, fixed = load_data(config)
    subjects, y, positions = ref.subject.to_numpy(), ref.y.to_numpy(), ref.position.to_numpy()
    region8 = region_context8(treatment, subjects, positions)
    np.savez_compressed(HERE / 'cache/regional_features.npz', X_train=treatment[:, :25], feature_names=FEATURE_NAMES)
    splits = e.splits_for(ref, fixed, config)
    controls = {}
    for name, folds in splits:
        base.assert_subject_isolation(subjects, folds)
        frame = ref.assign(fold=folds)
        controls[name] = pd.read_csv(DEP / 'predictions' / f'{name}_oof.csv')
        for col in IDENTITY:
            np.testing.assert_array_equal(frame[col], controls[name][col])
        frame.to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
    inputs = [(HERE / config[k]).resolve() for k in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')]
    inputs += [DEP / 'predictions' / f'{name}_oof.csv' for name, _ in splits]
    sources = [HERE / n for n in ('run_experiment.py', 'test_080.py', 'config.json', 'test_protocol.txt')]
    sources += [Path(module.__file__) for module in (old, e, e.e, e.e.e, e.e.e.e, base)]
    sources += [OLD / 'config.json', DEP / 'config.json', HERE.parents[2] / 'src/emo_rec/normalization.py', HERE.parents[2] / 'src/emo_rec/data.py', HERE.parents[2] / 'pyproject.toml', HERE.parents[2] / '数据/数据集说明文档.md']
    prov = {'inputs': {str(p): base.sha256(p) for p in inputs}, 'sources': {str(p): base.sha256(p) for p in sources}, 'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'platform': platform.platform(), 'raw_shape': list(raw.shape), 'treatment_shape': list(treatment.shape), 'public_arrays_loaded': False, 'regional_mapping': REGIONS, 'control_source': '070 saved p_train8; direct reuse, serialized OOF audited', 'train_context': 8, 'validation_context': 8, 'cpu_only': 'Frozen sklearn HGB has no GPU implementation', 'n_jobs': config['n_jobs'], 'threads_per_job': 1, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'pytest_before_sha256': base.sha256(HERE / 'pytest_before.txt')}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    print('080: 30 regional a8 fits only; official control reused from 070', flush=True)
    with threadpool_limits(limits=1):
        prov['active_threadpools'] = threadpool_info()
        results = Parallel(n_jobs=config['n_jobs'], prefer='threads')(delayed(job)(region8, y, name, folds, fold, 'region8', config) for name, folds in splits for fold in range(5))
    assert len(results) == config['expected_model_fits']
    runs, rows, deltas, checks = {}, [], [], {}
    for name, folds in splits:
        frame = ref.assign(fold=folds, p_train8=controls[name].p_train8.to_numpy(), p_region8=np.nan)
        for n, fold, role, p, seconds, n_train in results:
            if n == name:
                assert role == 'region8'
                frame.loc[folds == fold, 'p_region8'] = p
        assert np.isfinite(frame[['p_train8', 'p_region8']]).all().all()
        frame.to_csv(HERE / 'predictions' / f'{name}_oof.csv', index=False)
        saved = pd.read_csv(HERE / 'predictions' / f'{name}_oof.csv')
        checks[name] = float(np.max(np.abs(saved.p_train8 - controls[name].p_train8)))
        assert checks[name] <= 1e-12, checks
        runs[name] = {}
        for subset in ['all_rows'] + [str(i) for i in config['positions']]:
            q = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            runs[name][subset] = {}
            for arm, column in (('control8', 'p_train8'), ('region8', 'p_region8')):
                r = summarize(q, column)
                runs[name][subset][arm] = r
                rows.append({'split': name, 'position': subset, 'arm': arm, 'train_context': 8, 'validation_context': 8, **scalars(r)})
            deltas.append({'split': name, 'position': subset, **{k: scalars(runs[name][subset]['region8'])[k] - scalars(runs[name][subset]['control8'])[k] for k in KEYS}})
    aggregate = aggregate_runs(runs, config)
    decision = acceptance(aggregate, config)
    prov.update(model_fits=len(results), fit_roles={'region8': 30}, elapsed_seconds=time.perf_counter() - start, cache_sha256=base.sha256(HERE / 'cache/regional_features.npz'))
    metrics = {'runs': runs, 'aggregate': aggregate, 'acceptance': decision, 'control_070_max_abs_diff': checks, 'fit_seconds': [{'split': r[0], 'fold': r[1], 'role': r[2], 'train_context': 8, 'validation_context': 8, 'seconds': r[4], 'n_train': r[5]} for r in results], 'provenance': prov}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(deltas).to_csv(HERE / 'paired_delta.csv', index=False)
    lines = ['# 实验080：区域频谱池化严格纠错对照', '', '## 079比较错误', '079报告的约+0.009是区域模型a8训练相对其自身a40训练的差异，不是相对070逐通道a8控制的提升，因此不能作为区域池化有效的证据。本实验不修改079历史目录。', '', '## 冻结方法', '仅比较070保存p_train8与新拟合p_region8；两者均a8训练预测a8验证。固定复用079区域映射，150维log-bandpower区域内均值替换为25维，temporal450+PE120不变；无调参、挑位置或公开数组读取。', 'HC43排除，69被试2760行；fixed002及2042..2046、subject-level 5-fold、同070 HGB与种子。30 fits仅训练区域模型，控制直接取070原始p_train8；不计算40窗上下文。', '', '## 五个新split均值（fixed002不参与接受）', '|位置|控制AUC|区域AUC|ΔAUC|控制F1|区域F1|ΔF1|HC ΔAUC|DEP ΔAUC|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for subset, r in aggregate.items():
        a, b, d = r['control8'], r['region8'], r['delta_region8_minus_control8']
        lines.append(f"|{subset}|{a['auc']['mean']:.9f}|{b['auc']['mean']:.9f}|{d['auc']['mean']:+.9f}|{a['f1']['mean']:.9f}|{b['f1']['mean']:.9f}|{d['f1']['mean']:+.9f}|{d['hc_auc']['mean']:+.9f}|{d['dep_auc']['mean']:+.9f}|")
    lines += ['', '## 逐split all_rows', '|split|ΔAUC|ΔF1|HC ΔAUC|DEP ΔAUC|', '|---|---:|---:|---:|---:|']
    for r in deltas:
        if r['position'] == 'all_rows':
            lines.append(f"|{r['split']}|{r['auc']:+.9f}|{r['f1']:+.9f}|{r['hc_auc']:+.9f}|{r['dep_auc']:+.9f}|")
    d = aggregate['all_rows']['delta_region8_minus_control8']
    lines += ['', '## 接受判断', '平均ΔAUC及ΔF1各>=0.005；至少4/5 split两项同时非负；HC/DEP平均ΔAUC各>=-0.005，全部满足才接受。', json.dumps(decision, ensure_ascii=False), f"AUC非负{sum(v >= 0 for v in d['auc']['values'])}/5；F1非负{sum(v >= 0 for v in d['f1']['values'])}/5。", '结论：' + ('通过' if decision['accepted'] else '失败，未达到') + '预注册接受标准。', '', '## 可追溯性与局限', f"30 fits；控制保存后相对070最大差{max(checks.values()):.3g}；运行耗时{prov['elapsed_seconds']:.2f}秒。", '源码、缓存、六份070 OOF及测试前日志hash见input_metadata/provenance.json；完整逐折/逐被试指标见metrics.json，OOF与fold逐行保存。', 'sklearn HGB无GPU实现，为同模型冻结对照使用CPU例外，10 jobs、每job 1线程。五位置和五重复并非独立样本，已知训练视频分组仅作诊断，不外推公开测试提升。']
    (HERE / '实验报告.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    text = f"080 accepted={decision['accepted']}; mean delta AUC={d['auc']['mean']:.9f}; mean delta F1={d['f1']['mean']:.9f}; 30 fits"
    (HERE / 'summary.txt').write_text(text, encoding='utf8')
    print(text, flush=True)


if __name__ == '__main__':
    main()
