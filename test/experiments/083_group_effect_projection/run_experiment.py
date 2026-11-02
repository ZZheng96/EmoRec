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
DEP = HERE.parent / '070_train_context_match_new_splits'
spec = importlib.util.spec_from_file_location('exp070_dependency_083', DEP / 'run_experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
base, KEYS, scalars = e.base, e.KEYS, e.scalars
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from emo_rec.normalization import normalize_by_subject

FEATURE_BLOCKS = (150, 450, 120)
POSITIONS = (0, 1, 2, 3, 4)
IDENTITY = ('row_index', 'position', 'subject', 'group', 'trial', 'y', 'fold')


def validate_config(config):
    control = json.loads((DEP / 'config.json').read_text(encoding='utf8'))
    for k in ('hgb', 'outer_model_seed_base', 'repeat_seeds', 'threshold', 'excluded_subject', 'positions', 'n_jobs', 'threads_per_job'):
        assert config[k] == control[k]
    assert config['train_contexts'] == [8] and config['validation_context'] == 8
    assert config['feature_blocks'] == list(FEATURE_BLOCKS)
    assert config['expected_train_rows'] == 2760 and config['expected_model_fits'] == 30
    assert config['fit_roles'] == ['projection8']


def load_data(config):
    raw, ref, fixed = e.load_data(config)
    assert raw.shape == (2760, 720) and np.isfinite(raw).all()
    assert ref.subject.nunique() == 69 and 'HC43' not in set(ref.subject)
    assert set(ref.position) == set(POSITIONS)
    return raw, ref, fixed


def estimate_projection(x, ref, train_mask, threshold=1e-12):
    # Both effect directions use both emotion labels, from training rows only.
    vectors = []
    for y_value in (0, 1):
        hc = x[train_mask & (ref.group.to_numpy() == 'HC') & (ref.y.to_numpy() == y_value)]
        dep = x[train_mask & (ref.group.to_numpy() == 'DEP') & (ref.y.to_numpy() == y_value)]
        assert len(hc) and len(dep)
        vectors.append(hc.mean(axis=0) - dep.mean(axis=0))
    directions = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(directions, axis=1)
    active = directions[norms > threshold]
    if len(active):
        q, r = np.linalg.qr(active.T, mode='reduced')
        diag = np.abs(np.diag(r))
        rank = int(np.sum(diag > threshold))
        q = q[:, :rank]
    else:
        q = np.empty((x.shape[1], 0), dtype=np.float64)
        rank = 0
    projection = np.eye(x.shape[1]) - q @ q.T
    return projection, {'rank': rank, 'direction_norms': norms.tolist(), 'q_norm': float(np.linalg.norm(q)), 'projection_fro_norm': float(np.linalg.norm(projection)), 'q_shape': list(q.shape), 'threshold': threshold}


def a8_zscore(raw, subjects, positions):
    result = np.empty_like(np.asarray(raw, dtype=np.float64))
    for position in POSITIONS:
        ix = positions == position
        result[ix] = normalize_by_subject(raw[ix], subjects[ix], 'zscore')
    return result


def fit_one(raw, ref, name, folds, fold, config):
    train = folds != fold
    valid = folds == fold
    subjects = ref.subject.to_numpy()
    positions = ref.position.to_numpy()
    z = a8_zscore(raw, subjects, positions)
    projection, info = estimate_projection(z, ref, train, config['projection_norm_threshold'])
    x_train = np.asarray(z[train] @ projection, dtype=np.float64)
    x_valid = np.asarray(z[valid] @ projection, dtype=np.float64)
    start = time.perf_counter()
    with threadpool_limits(limits=config['threads_per_job']):
        model = base.make_model(config, config['outer_model_seed_base'] + fold)
        model.fit(x_train, ref.y.to_numpy()[train])
        p = model.predict_proba(x_valid)[:, 1]
    info.update({'split': name, 'fold': fold, 'n_train': int(train.sum()), 'n_valid': int(valid.sum()), 'train_projected_norm': float(np.linalg.norm(x_train)), 'valid_projected_norm': float(np.linalg.norm(x_valid))})
    return name, fold, p, time.perf_counter() - start, int(train.sum()), info


def summarize(frame, column):
    rr = frame.rename(columns={'subject': 'subject_id', 'group': 'health_group'})
    return base.summarize(frame.y.to_numpy(), frame[column].to_numpy(), rr, frame.fold.to_numpy())


def aggregate_runs(runs, config):
    out = {}
    for subset in ['all_rows'] + [str(i) for i in config['positions']]:
        out[subset] = {}
        for arm in ('control8', 'projection8', 'delta_projection8_minus_control8'):
            vals = []
            for seed in config['repeat_seeds']:
                r = runs[f'repeat{seed}'][subset]
                vals.append(scalars(r[arm]) if arm != 'delta_projection8_minus_control8' else {k: scalars(r['projection8'])[k] - scalars(r['control8'])[k] for k in KEYS})
            out[subset][arm] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
    return out


def acceptance(aggregate, config):
    d = aggregate['all_rows']['delta_projection8_minus_control8']; a = config['acceptance']
    checks = {'mean_delta_auc': d['auc']['mean'] >= a['mean_delta_auc_min'], 'mean_delta_f1': d['f1']['mean'] >= a['mean_delta_f1_min'], 'joint_nonnegative': sum(x >= 0 and y >= 0 for x, y in zip(d['auc']['values'], d['f1']['values'])) >= a['nonnegative_splits_min_each'], 'hc_no_excess_drop': d['hc_auc']['mean'] >= -a['health_group_mean_auc_drop_max'], 'dep_no_excess_drop': d['dep_auc']['mean'] >= -a['health_group_mean_auc_drop_max']}
    return {'accepted': all(checks.values()), 'checks': checks}


def main():
    start = time.perf_counter(); config = json.loads((HERE / 'config.json').read_text(encoding='utf8')); validate_config(config)
    assert (HERE / 'pytest_before.txt').exists()
    raw, ref, fixed = load_data(config); subjects = ref.subject.to_numpy(); splits = e.splits_for(ref, fixed, config)
    controls = {}; sources = [HERE / n for n in ('run_experiment.py', 'test_083.py', 'config.json', 'test_protocol.txt')]
    inputs = [(HERE / config[k]).resolve() for k in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')]
    for name, folds in splits:
        base.assert_subject_isolation(subjects, folds); ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
        controls[name] = pd.read_csv(DEP / 'predictions' / f'{name}_oof.csv')
        for col in IDENTITY: np.testing.assert_array_equal(ref.assign(fold=folds)[col], controls[name][col])
        inputs.append(DEP / 'predictions' / f'{name}_oof.csv')
    sources += [DEP / 'config.json', Path(e.__file__), Path(base.__file__), ROOT / 'src/emo_rec/normalization.py', ROOT / '数据/数据集说明文档.md']
    prov = {'inputs': {str(p): base.sha256(p) for p in inputs}, 'sources': {str(p): base.sha256(p) for p in sources}, 'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'raw_shape': list(raw.shape), 'treatment_shape': list(raw.shape), 'public_arrays_loaded': False, 'control_source': '070 saved p_train8; direct reuse', 'projection_formula': 'z=a8_zscore(raw) first; P=I-QQ.T; X_proj=z@P; Q from QR([d0,d1].T)', 'projection_y_values': [0, 1], 'normalization_order': 'subject-position a8 zscore before train-only projection; no post-projection zscore', 'projection_train_only': True, 'threshold': config['projection_norm_threshold'], 'model_fits': 30, 'cpu_only': 'sklearn HGB has no GPU implementation', 'n_jobs': config['n_jobs'], 'threads_per_job': config['threads_per_job'], 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    print('083: 30 projection8 fits; official control reused from 070; no public arrays', flush=True)
    with threadpool_limits(limits=1):
        results = Parallel(n_jobs=config['n_jobs'], prefer='threads')(delayed(fit_one)(raw, ref, name, folds, fold, config) for name, folds in splits for fold in range(5))
    assert len(results) == 30
    pd.DataFrame([r[5] for r in results]).to_json(HERE / 'input_metadata/projection_fits.json', orient='records', indent=2)
    runs = {}; rows = []; deltas = []; checks = {}; fit_seconds = []
    for name, folds in splits:
        frame = ref.assign(fold=folds, p_train8=controls[name].p_train8.to_numpy(), p_projection8=np.nan)
        for n, fold, p, seconds, n_train, info in results:
            if n == name: frame.loc[folds == fold, 'p_projection8'] = p; fit_seconds.append({'split': n, 'fold': fold, 'role': 'projection8', 'train_context': 8, 'validation_context': 8, 'seconds': seconds, 'n_train': n_train})
        assert np.isfinite(frame[['p_train8', 'p_projection8']]).all().all(); output_path = HERE / 'predictions' / f'{name}_oof.csv'
        frame.to_csv(output_path, index=False)
        saved = pd.read_csv(output_path)
        for col in IDENTITY:
            np.testing.assert_array_equal(frame[col].to_numpy(), saved[col].to_numpy())
        for col in ('p_train8', 'p_projection8'):
            np.testing.assert_allclose(frame[col].to_numpy(), saved[col].to_numpy(), rtol=0, atol=1e-12)
        checks[name] = float(np.max(np.abs(saved.p_train8 - controls[name].p_train8))); assert checks[name] <= 1e-12
        runs[name] = {}
        for subset in ['all_rows'] + [str(i) for i in config['positions']]:
            q = frame if subset == 'all_rows' else frame[frame.position == int(subset)]; runs[name][subset] = {}
            for arm, col in (('control8', 'p_train8'), ('projection8', 'p_projection8')):
                r = summarize(q, col); runs[name][subset][arm] = r; rows.append({'split': name, 'position': subset, 'arm': arm, 'train_context': 8, 'validation_context': 8, **scalars(r)})
            deltas.append({'split': name, 'position': subset, **{k: scalars(runs[name][subset]['projection8'])[k] - scalars(runs[name][subset]['control8'])[k] for k in KEYS}})
    aggregate = aggregate_runs(runs, config); decision = acceptance(aggregate, config); prov.update(active_threadpools=threadpool_info(), elapsed_seconds=time.perf_counter()-start)
    metrics = {'runs': runs, 'aggregate': aggregate, 'acceptance': decision, 'control_070_max_abs_diff': checks, 'fit_seconds': fit_seconds, 'provenance': prov}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8'); (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False); pd.DataFrame(deltas).to_csv(HERE / 'paired_delta.csv', index=False)
    lines = ['# 实验083：训练折内HC/DEP效应方向正交投影', '', '控制直接读取070保存p_train8；处理为训练折估计d0、d1并QR正交投影后的720维a8 HGB。仅训练队列OOF，不外推公开测试集，不声称0.95。', '', '|位置|控制AUC|处理AUC|ΔAUC|控制F1|处理F1|ΔF1|HC ΔAUC|DEP ΔAUC|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for subset, r in aggregate.items():
        a,b,d=r['control8'],r['projection8'],r['delta_projection8_minus_control8']; lines.append(f"|{subset}|{a['auc']['mean']:.9f}|{b['auc']['mean']:.9f}|{d['auc']['mean']:+.9f}|{a['f1']['mean']:.9f}|{b['f1']['mean']:.9f}|{d['f1']['mean']:+.9f}|{d['hc_auc']['mean']:+.9f}|{d['dep_auc']['mean']:+.9f}|")
    lines += ['', '## 接受判断', '五个新split all_rows平均ΔAUC/F1均>=0.005；至少4/5 split两项同时非负；HC/DEP平均ΔAUC各>=-0.005。', json.dumps(decision, ensure_ascii=False), '结论：' + ('通过' if decision['accepted'] else '失败，停止，不调参、不挑位置。'), '', '## 限制', 'Q仅由每个训练折HC/DEP两类、y=0/1的均值差d0、d1估计，验证标签未参与；退化方向按1e-12阈值丢弃。先按070 a8 subject-position zscore得到z，再将固定投影直接应用于训练/验证z@P，不再zscore。rank按实际拟合记录；a8平衡可能使d1≈-d0而rank=1，被去除方向可能包含情绪判别方向，不能称纯噪声。sklearn HGB无GPU实现，CPU例外；五位置及重复非独立样本。']
    (HERE / '实验报告.md').write_text('\n'.join(lines)+'\n', encoding='utf8'); (HERE / 'summary.txt').write_text(f"083 accepted={decision['accepted']}; mean delta AUC={aggregate['all_rows']['delta_projection8_minus_control8']['auc']['mean']:.9f}; mean delta F1={aggregate['all_rows']['delta_projection8_minus_control8']['f1']['mean']:.9f}; 30 fits", encoding='utf8')
    print(json.dumps({'acceptance': decision, 'all_rows': aggregate['all_rows']}, indent=2), flush=True)

if __name__ == '__main__': main()
