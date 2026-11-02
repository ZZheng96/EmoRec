import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import scipy
from scipy.stats import rankdata
from scipy.special import ndtri
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits, threadpool_info

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DEP = HERE.parent / '070_train_context_match_new_splits'
spec = importlib.util.spec_from_file_location('exp070_dependency', DEP / 'run_experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
base, KEYS, scalars = e.base, e.KEYS, e.scalars
N_CONTEXT = 8
FEATURE_BLOCKS = (150, 450, 120)
POSITIONS = (0, 1, 2, 3, 4)
WINDOWS_PER_CONTEXT = 8
# Training-cohort OOF evidence only: no public-test inference or 0.95 claim.
IDENTITY = ('row_index', 'position', 'subject', 'group', 'trial', 'y', 'fold')


def validate_config(config):
    control = json.loads((DEP / 'config.json').read_text(encoding='utf8'))
    for k in ('hgb', 'outer_model_seed_base', 'repeat_seeds', 'threshold', 'excluded_subject', 'positions', 'n_jobs', 'threads_per_job'):
        assert config[k] == control[k]
    assert config['train_contexts'] == [8] and config['validation_context'] == 8
    assert config['feature_blocks'] == list(FEATURE_BLOCKS)
    assert config['n_context'] == N_CONTEXT == 8 and config['ties'] == 'average'
    assert config['expected_train_rows'] == 2760
    assert config['expected_model_fits'] == 30
    assert config['fit_roles'] == ['rank8']
    assert config['acceptance'] == {'mean_delta_auc_min': 0.005, 'mean_delta_f1_min': 0.005, 'nonnegative_splits_min_each': 4, 'health_group_mean_auc_drop_max': 0.005}


def load_data(config):
    raw, ref, fixed = e.load_data(config)
    assert raw.shape == (2760, sum(FEATURE_BLOCKS)) and np.isfinite(raw).all()
    assert ref.subject.nunique() == 69 and 'HC43' not in set(ref.subject)
    assert set(ref.position) == set(POSITIONS)
    assert (ref.groupby(['subject', 'position']).size() == WINDOWS_PER_CONTEXT).all()
    return raw, ref, fixed


def rank_context8(raw, subjects, positions):
    # Unlabelled adaptation uses exactly eight windows from one subject/position.
    assert raw.ndim == 2 and raw.shape == (len(subjects), sum(FEATURE_BLOCKS))
    assert len(positions) == len(subjects) and set(positions) == set(POSITIONS)
    result = np.empty_like(raw, dtype=np.float64)
    for subject in np.unique(subjects):
        for position in POSITIONS:
            ix = (subjects == subject) & (positions == position)
            assert ix.sum() == WINDOWS_PER_CONTEXT
            values = raw[ix].astype(np.float64)
            ranks = rankdata(values, axis=0, method='average')
            result[ix] = ndtri((ranks - 0.5) / N_CONTEXT)
    assert np.isfinite(result).all()
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
        for arm in ('control8', 'rank8', 'delta_rank8_minus_control8'):
            vals = []
            for seed in config['repeat_seeds']:
                r = runs[f'repeat{seed}'][subset]
                vals.append(scalars(r[arm]) if arm != 'delta_rank8_minus_control8' else {k: scalars(r['rank8'])[k] - scalars(r['control8'])[k] for k in KEYS})
            aggregate[subset][arm] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
    return aggregate


def acceptance(aggregate, config):
    d = aggregate['all_rows']['delta_rank8_minus_control8']
    a = config['acceptance']
    checks = {'mean_delta_auc': d['auc']['mean'] >= a['mean_delta_auc_min'], 'mean_delta_f1': d['f1']['mean'] >= a['mean_delta_f1_min'], 'joint_nonnegative': sum(x >= 0 and y >= 0 for x, y in zip(d['auc']['values'], d['f1']['values'])) >= a['nonnegative_splits_min_each'], 'hc_no_excess_drop': d['hc_auc']['mean'] >= -a['health_group_mean_auc_drop_max'], 'dep_no_excess_drop': d['dep_auc']['mean'] >= -a['health_group_mean_auc_drop_max']}
    return {'accepted': all(checks.values()), 'checks': checks}


def main():
    start = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    validate_config(config)
    assert (HERE / 'pytest_before.txt').exists()
    for folder in ('predictions', 'input_metadata'):
        (HERE / folder).mkdir(exist_ok=True)
    raw, ref, fixed = load_data(config)
    subjects, y, positions = ref.subject.to_numpy(), ref.y.to_numpy(), ref.position.to_numpy()
    rank8 = rank_context8(raw, subjects, positions)
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
    sources = [HERE / n for n in ('run_experiment.py', 'test_082.py', 'config.json', 'test_protocol.txt')]
    sources += [Path(module.__file__) for module in (e, e.e, e.e.e, e.e.e.e, base)]
    sources += [DEP / 'config.json', HERE.parents[2] / 'src/emo_rec/normalization.py', HERE.parents[2] / 'src/emo_rec/data.py', HERE.parents[2] / 'pyproject.toml', HERE.parents[2] / '数据/数据集说明文档.md']
    prov = {'inputs': {str(p): base.sha256(p) for p in inputs}, 'sources': {str(p): base.sha256(p) for p in sources}, 'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'platform': platform.platform(), 'raw_shape': list(raw.shape), 'treatment_shape': list(rank8.shape), 'public_arrays_loaded': False, 'scipy': scipy.__version__, 'normalization': {'method': 'rank-Gaussian', 'ties': 'average', 'n_context': N_CONTEXT, 'formula': 'ndtri((rank - 0.5) / 8)', 'feature_blocks': FEATURE_BLOCKS}, 'control_source': '070 saved p_train8; direct reuse, serialized OOF audited', 'train_context': 8, 'validation_context': 8, 'cpu_only': 'Frozen sklearn HGB has no GPU implementation', 'n_jobs': config['n_jobs'], 'threads_per_job': 1, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'pytest_before_sha256': base.sha256(HERE / 'pytest_before.txt')}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    print('082: 30 rank a8 fits only; official control reused from 070', flush=True)
    with threadpool_limits(limits=1):
        prov['active_threadpools'] = threadpool_info()
        results = Parallel(n_jobs=config['n_jobs'], prefer='threads')(delayed(job)(rank8, y, name, folds, fold, 'rank8', config) for name, folds in splits for fold in range(5))
    assert len(results) == config['expected_model_fits']
    runs, rows, deltas, checks = {}, [], [], {}
    for name, folds in splits:
        frame = ref.assign(fold=folds, p_train8=controls[name].p_train8.to_numpy(), p_rank8=np.nan)
        for n, fold, role, p, seconds, n_train in results:
            if n == name:
                assert role == 'rank8'
                frame.loc[folds == fold, 'p_rank8'] = p
        assert np.isfinite(frame[['p_train8', 'p_rank8']]).all().all()
        frame.to_csv(HERE / 'predictions' / f'{name}_oof.csv', index=False)
        saved = pd.read_csv(HERE / 'predictions' / f'{name}_oof.csv')
        checks[name] = float(np.max(np.abs(saved.p_train8 - controls[name].p_train8)))
        assert checks[name] <= 1e-12, checks
        runs[name] = {}
        for subset in ['all_rows'] + [str(i) for i in config['positions']]:
            q = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            runs[name][subset] = {}
            for arm, column in (('control8', 'p_train8'), ('rank8', 'p_rank8')):
                r = summarize(q, column)
                runs[name][subset][arm] = r
                rows.append({'split': name, 'position': subset, 'arm': arm, 'train_context': 8, 'validation_context': 8, **scalars(r)})
            deltas.append({'split': name, 'position': subset, **{k: scalars(runs[name][subset]['rank8'])[k] - scalars(runs[name][subset]['control8'])[k] for k in KEYS}})
    aggregate = aggregate_runs(runs, config)
    decision = acceptance(aggregate, config)
    prov.update(model_fits=len(results), fit_roles={'rank8': 30}, elapsed_seconds=time.perf_counter() - start)
    metrics = {'runs': runs, 'aggregate': aggregate, 'acceptance': decision, 'control_070_max_abs_diff': checks, 'fit_seconds': [{'split': r[0], 'fold': r[1], 'role': r[2], 'train_context': 8, 'validation_context': 8, 'seconds': r[4], 'n_train': r[5]} for r in results], 'provenance': prov}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(deltas).to_csv(HERE / 'paired_delta.csv', index=False)
    lines = ['# 实验082：被试内、位置内 rank-Gaussian 归一化', '', '## 唯一假设与冻结方法', '仅检验rank-Gaussian是否优于070均值/std z-score。控制直接复用070保存p_train8；处理为p_rank8；两者均a8训练、a8验证。150 spectral + 450 temporal + 120 PE共720维完全不变。', '每个subject每个position恰8窗，逐特征以average rank处理ties，固定n_context=8，u=(rank-0.5)/8，z=scipy.special.ndtri(u)。变换不依赖trial_id，不读取标签，不跨被试或位置，无调参、挑位置、公开测试数组读取。', '依据数据集说明：训练70人、30通道、250Hz；排除HC43后69人2760个10秒窗。fixed002及repeat2042..2046六套subject-level 5-fold，HGB、seed、阈值完全冻结。仅处理30 fits，不重训控制，不计算40窗上下文。', '', '## 五个新split均值（fixed002不参与接受）', '|位置|控制AUC|rank AUC|ΔAUC|控制F1|rank F1|ΔF1|HC ΔAUC|DEP ΔAUC|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for subset, r in aggregate.items():
        a, b, d = r['control8'], r['rank8'], r['delta_rank8_minus_control8']
        lines.append(f"|{subset}|{a['auc']['mean']:.9f}|{b['auc']['mean']:.9f}|{d['auc']['mean']:+.9f}|{a['f1']['mean']:.9f}|{b['f1']['mean']:.9f}|{d['f1']['mean']:+.9f}|{d['hc_auc']['mean']:+.9f}|{d['dep_auc']['mean']:+.9f}|")
    lines += ['', '## 逐split all_rows', '|split|ΔAUC|ΔF1|HC ΔAUC|DEP ΔAUC|', '|---|---:|---:|---:|---:|']
    for r in deltas:
        if r['position'] == 'all_rows':
            lines.append(f"|{r['split']}|{r['auc']:+.9f}|{r['f1']:+.9f}|{r['hc_auc']:+.9f}|{r['dep_auc']:+.9f}|")
    d = aggregate['all_rows']['delta_rank8_minus_control8']
    lines += ['', '## 接受判断', '平均ΔAUC及ΔF1各>=0.005；至少4/5 split两项同时非负；HC/DEP平均ΔAUC各>=-0.005，全部满足才接受。', json.dumps(decision, ensure_ascii=False), f"AUC非负{sum(v >= 0 for v in d['auc']['values'])}/5；F1非负{sum(v >= 0 for v in d['f1']['values'])}/5。", '结论：' + ('通过' if decision['accepted'] else '失败，未达到') + '预注册接受标准；失败即停止，不追加调参或位置选择。', '', '## 可追溯性与局限', f"30 fits；控制保存后相对070最大差{max(checks.values()):.3g}；运行耗时{prov['elapsed_seconds']:.2f}秒。", '源码、缓存、六份070 OOF及测试前日志hash见input_metadata/provenance.json；完整逐折/逐被试指标见metrics.json，OOF与fold逐行保存。', 'sklearn HGB无GPU实现，为同模型冻结对照使用CPU例外，10 jobs、每job 1线程。复用现有NumPy特征缓存与指标实现，不重新计算EEG特征。五位置和五重复并非独立样本；这只是训练队列OOF证据，不能外推公开测试集，也不能声称达到0.95。']
    (HERE / '实验报告.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    text = f"082 accepted={decision['accepted']}; mean delta AUC={d['auc']['mean']:.9f}; mean delta F1={d['f1']['mean']:.9f}; 30 fits"
    (HERE / 'summary.txt').write_text(text, encoding='utf8')
    print(text, flush=True)


if __name__ == '__main__':
    main()
