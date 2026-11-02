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
import h5py
from contextlib import contextmanager
from unittest.mock import patch
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits, threadpool_info

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / '070_train_context_match_new_splits'
spec = importlib.util.spec_from_file_location('exp070_dependency', OLD / 'run_experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
base = e.base
load_data, contexts, scalars = e.load_data, e.contexts, e.scalars
KEYS = e.KEYS


META = ('row_index', 'subject', 'group', 'trial', 'y', 'position', 'fold')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from emo_rec import data
TRAIN_KEYS = ('X_train', 'subject_ids', 'health_groups', 'original_trial_ids', 'y_train', 'feature_names')


@contextmanager
def public_guard():
    # Same allowlist and public-loader/path protection as experiment 086.
    original_get = np.lib.npyio.NpzFile.__getitem__
    original_load, original_h5 = np.load, h5py.File
    evidence = {'active': True, 'loaded_paths': [], 'accessed_keys': []}
    def get(cache, key):
        assert key in TRAIN_KEYS, f'Forbidden array: {key}'
        evidence['accessed_keys'].append(key)
        return original_get(cache, key)
    def check(path):
        text = str(path).lower()
        assert not any(token in text for token in ('public', '测试集', 'test_data')), f'Forbidden public path: {path}'
        evidence['loaded_paths'].append(str(Path(path).resolve()))
    def load(path, *args, **kwargs):
        check(path)
        return original_load(path, *args, **kwargs)
    def h5(path, *args, **kwargs):
        check(path)
        assert (ROOT / '数据/训练集').resolve() in Path(path).resolve().parents
        return original_h5(path, *args, **kwargs)
    with patch.object(np.lib.npyio.NpzFile, '__getitem__', get), patch.object(np, 'load', load), patch.object(h5py, 'File', h5), patch.object(data, 'load_public_test_segments', side_effect=AssertionError('Forbidden public loader')):
        yield evidence


def validate_metadata(frame, reference):
    for col in META:
        np.testing.assert_array_equal(frame[col].to_numpy(), reference[col].to_numpy())


def validate_config(config):
    assert config['repeat_seeds'] == [2042, 2043, 2044, 2045, 2046]
    assert config['expected_model_fits'] == 60
    assert config['expected_stage_fits'] == {'first_stage_raw': 30, 'second_stage_treatment': 30}
    assert config['prototype']['feature_mode'] == 'raw_plus_class_residual'
    assert config['prototype']['pseudo_label_source'] == 'training_only_raw_hgb'
    assert config['public_access'] is False


def splits_for(ref, fixed, config):
    return [('fixed002', fixed)] + [(f'repeat{s}', base.make_subject_folds(ref.subject.to_numpy(), ref.group.to_numpy(), s)) for s in config['repeat_seeds']]


def residual_features(x, prototypes, labels):
    return np.mean(np.abs(x - prototypes[np.asarray(labels, dtype=int)]), axis=1, keepdims=True)


def job(a8, y, name, folds, fold, config, raw_control):
    start = time.perf_counter()
    train, valid = folds != fold, folds == fold
    model = base.make_model(config, config.get('outer_model_seed_base', 2052) + fold)
    fit_start = time.perf_counter()
    model.fit(a8[train], y[train])
    first_seconds = time.perf_counter() - fit_start
    p_raw_train = model.predict_proba(a8[train])[:, 1]
    p_raw_valid = model.predict_proba(a8[valid])[:, 1]
    prototypes = np.vstack([a8[train][y[train] == c].mean(axis=0) for c in (0, 1)])
    train_res = residual_features(a8[train], prototypes, y[train])
    valid_res = residual_features(a8[valid], prototypes, (p_raw_valid >= config['threshold']).astype(int))
    x_train = np.column_stack([a8[train], train_res])
    x_valid = np.column_stack([a8[valid], valid_res])
    treatment = base.make_model(config, config.get('outer_model_seed_base', 2052) + fold)
    fit_start = time.perf_counter()
    treatment.fit(x_train, y[train])
    second_seconds = time.perf_counter() - fit_start
    p_treatment = treatment.predict_proba(x_valid)[:, 1]
    elapsed = time.perf_counter() - start
    return name, fold, p_treatment, p_raw_valid, elapsed, int(train.sum()), prototypes.shape[1], [
        {'split': name, 'fold': fold, 'stage': 'first_stage_raw', 'seconds': first_seconds, 'n_train': int(train.sum())},
        {'split': name, 'fold': fold, 'stage': 'second_stage_treatment', 'seconds': second_seconds, 'n_train': int(train.sum())},
    ]


def summarize(frame, column):
    rr = frame.rename(columns={'subject': 'subject_id', 'group': 'health_group'})
    return base.summarize(frame.y.to_numpy(), frame[column].to_numpy(), rr, frame.fold.to_numpy())


def aggregate_runs(runs, config):
    aggregate = {}
    for subset in ['all_rows'] + [str(i) for i in config['positions']]:
        aggregate[subset] = {}
        for context in ('control', 'treatment', 'delta_treatment_minus_control'):
            vals = []
            for seed in config['repeat_seeds']:
                r = runs[f'repeat{seed}'][subset]
                vals.append(scalars(r[context]) if context != 'delta_treatment_minus_control' else {k: scalars(r['treatment'])[k] - scalars(r['control'])[k] for k in KEYS})
            aggregate[subset][context] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
    return aggregate


def acceptance(aggregate, config):
    d = aggregate['all_rows']['delta_treatment_minus_control']
    a = config['acceptance']
    checks = {'mean_delta_auc': d['auc']['mean'] >= a['mean_delta_auc_min'], 'mean_delta_f1': d['f1']['mean'] >= a['mean_delta_f1_min'], 'auc_nonnegative': sum(v >= 0 for v in d['auc']['values']) >= a['nonnegative_splits_min_each'], 'f1_nonnegative': sum(v >= 0 for v in d['f1']['values']) >= a['nonnegative_splits_min_each'], 'hc_no_excess_drop': d['hc_auc']['mean'] >= -a['health_group_mean_auc_drop_max'], 'dep_no_excess_drop': d['dep_auc']['mean'] >= -a['health_group_mean_auc_drop_max']}
    return {'accepted': all(checks.values()), 'checks': checks}


def main():
    start = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    validate_config(config)
    for folder in ('predictions', 'input_metadata'):
        (HERE / folder).mkdir(exist_ok=True)
    data_config = json.loads((OLD / 'config.json').read_text(encoding='utf8'))
    data_config.update({k: config[k] for k in ('excluded_subject',)})
    with public_guard() as guard_evidence:
        raw, ref, fixed = load_data(data_config)
    subjects, y = ref.subject.to_numpy(), ref.y.to_numpy()
    _, a8 = contexts(raw, subjects, ref.position.to_numpy())
    splits = splits_for(ref, fixed, config)
    inputs = [(OLD / 'config.json'), (OLD / 'metrics.json')]
    inputs += [OLD / 'predictions' / f'{name}_oof.csv' for name, _ in splits]
    inputs += [(HERE / config[k]).resolve() for k in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')]
    assert set(guard_evidence['loaded_paths']) <= {str(p.resolve()) for p in inputs}
    sources = [HERE / n for n in ('run_experiment.py', 'test_087.py', 'config.json', 'test_protocol.txt')]
    sources += [OLD / n for n in ('run_experiment.py', 'test_070.py', 'config.json', 'test_protocol.txt')]
    sources += [Path(base.__file__), e.e.e.e.ROOT / 'src/emo_rec/normalization.py', e.e.e.e.ROOT / 'src/emo_rec/data.py', e.e.e.e.ROOT / 'src/emo_rec/submission.py', e.e.e.e.ROOT / 'pyproject.toml', e.e.e.e.ROOT / '数据/数据集说明文档.md']
    historical = json.loads((OLD / 'input_metadata/provenance.json').read_text(encoding='utf8'))
    sources += [Path(p) for p in historical['sources']]
    sources += list((ROOT / 'src/emo_rec').glob('*.py'))
    sources += [OLD / 'input_metadata/provenance.json']
    prov = {'inputs': {str(p.resolve()): sha256(p) for p in inputs}, 'sources': {str(p.resolve()): sha256(p) for p in sources}, 'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'platform': platform.platform(), 'raw_shape': list(raw.shape), 'public_arrays_loaded': False, 'public_guard': 'active around load_data', 'cpu_only': 'sklearn HGB has no GPU implementation; frozen-model CPU exception', 'n_jobs': config['n_jobs'], 'threads_per_job': config['threads_per_job'], 'model_fits': 60, 'stage_fits': config['expected_stage_fits'], 'new_seeds_have_historical_oof': False}
    prov['guard_evidence'] = guard_evidence
    for name, folds in splits:
        validate_metadata(pd.read_csv(OLD / 'predictions' / f'{name}_oof.csv'), ref.assign(fold=folds))
        base.assert_subject_isolation(subjects, folds)
        ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    print('Frozen 070 a8 control; class prototypes on training folds; 60 fits (30 raw + 30 treatment); guarded training cache access', flush=True)
    with threadpool_limits(limits=1):
        prov['active_threadpools'] = threadpool_info()
        results = Parallel(n_jobs=config['n_jobs'], prefer='threads')(delayed(job)(a8, y, name, folds, fold, config, None) for name, folds in splits for fold in range(5))
    runs, rows, deltas, control_diffs, diagnostics = {}, [], [], {}, {}
    for name, folds in splits:
        old = pd.read_csv(OLD / 'predictions' / f'{name}_oof.csv')
        validate_metadata(old, ref.assign(fold=folds))
        frame = ref.assign(fold=folds, p_control=old.p_train8.to_numpy(), p_treatment=np.nan, p_first_stage=np.nan)
        for n, fold, p, p_raw, seconds, n_train, n_features, stage_times in results:
            if n == name:
                frame.loc[folds == fold, 'p_treatment'] = p
                frame.loc[folds == fold, 'p_first_stage'] = p_raw
                control_diffs[f'{name}/fold{fold}'] = float(np.max(np.abs(p_raw - old.loc[folds == fold, 'p_train8'].to_numpy())))
                assert control_diffs[f'{name}/fold{fold}'] <= 1e-12
                diagnostics[f'{name}/fold{fold}'] = {'n_train': n_train, 'n_valid': int((folds == fold).sum()), 'raw_validation_mean': float(np.mean(p_raw)), 'pseudo_positive_rate': float(np.mean(p_raw >= config['threshold'])), 'prototype_features': 1, 'augmented_features': n_features + 1, 'seconds': seconds}
        assert np.isfinite(frame[['p_control', 'p_treatment']]).all().all()
        frame.to_csv(HERE / 'predictions' / f'{name}_oof.csv', index=False)
        runs[name] = {}
        for subset in ['all_rows'] + [str(i) for i in config['positions']]:
            selected = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            runs[name][subset] = {}
            for col, key in [('p_control', 'control'), ('p_treatment', 'treatment')]:
                r = summarize(selected, col)
                runs[name][subset][key] = r
                rows.append({'split': name, 'position': subset, 'arm': key, **scalars(r)})
            deltas.append({'split': name, 'position': subset, **{k: scalars(runs[name][subset]['treatment'])[k] - scalars(runs[name][subset]['control'])[k] for k in KEYS}})
    aggregate = aggregate_runs(runs, config)
    decision = acceptance(aggregate, config)
    prov['model_fits'] = 2 * len(results)
    assert prov['model_fits'] == config['expected_model_fits']
    prov['stage_fits'] = {stage: sum(t['stage'] == stage for r in results for t in r[7]) for stage in config['expected_stage_fits']}
    assert prov['stage_fits'] == config['expected_stage_fits']
    metrics = {'runs': runs, 'aggregate': aggregate, 'acceptance': decision, 'first_stage_control_max_abs_diff': control_diffs, 'diagnostics': diagnostics, 'elapsed_seconds': time.perf_counter() - start, 'fit_seconds': [stage for r in results for stage in r[7]], 'provenance': prov}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(deltas).to_csv(HERE / 'paired_delta.csv', index=False)
    print(json.dumps({'elapsed_seconds': metrics['elapsed_seconds'], 'acceptance': decision, 'all_rows': aggregate['all_rows']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
