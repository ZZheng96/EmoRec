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
OLD = HERE.parent / '069_train_context_match_replication'
spec = importlib.util.spec_from_file_location('exp069_dependency', OLD / 'run_experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
base = e.base
load_data, contexts, scalars = e.load_data, e.contexts, e.scalars
job, summarize, aggregate_runs, acceptance = e.job, e.summarize, e.aggregate_runs, e.acceptance
KEYS = e.KEYS


def validate_config(config):
    old = json.loads((OLD / 'config.json').read_text(encoding='utf8'))
    assert {k: v for k, v in config.items() if k not in ('experiment_id', 'repeat_seeds')} == {k: v for k, v in old.items() if k not in ('experiment_id', 'repeat_seeds')}
    assert config['repeat_seeds'] == [2042, 2043, 2044, 2045, 2046]


def splits_for(ref, fixed, config):
    return [('fixed002', fixed)] + [(f'repeat{s}', base.make_subject_folds(ref.subject.to_numpy(), ref.group.to_numpy(), s)) for s in config['repeat_seeds']]


def historical_checks():
    checks = {}
    old_config = json.loads((OLD / 'config.json').read_text(encoding='utf8'))
    for name in ['fixed002'] + [f'repeat{s}' for s in old_config['repeat_seeds']]:
        a = pd.read_csv(OLD / 'predictions' / f'{name}_oof.csv')
        b = pd.read_csv(e.OLD / 'predictions' / f'{name}_oof.csv')
        for col in ('row_index', 'position', 'subject', 'group', 'trial', 'y', 'fold'):
            np.testing.assert_array_equal(a[col], b[col])
        checks[name] = float(np.max(np.abs(a.p_train40 - b.p_train40)))
        assert checks[name] <= 1e-12
    return checks


def main():
    start = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    validate_config(config)
    for folder in ('predictions', 'input_metadata'):
        (HERE / folder).mkdir(exist_ok=True)
    history = historical_checks()
    raw, ref, fixed = load_data(config)
    subjects, y = ref.subject.to_numpy(), ref.y.to_numpy()
    a40, a8 = contexts(raw, subjects, ref.position.to_numpy())
    splits = splits_for(ref, fixed, config)
    inputs = [(HERE / config[k]).resolve() for k in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')]
    inputs += [folder / 'predictions' / f'{name}_oof.csv' for folder in (OLD, e.OLD) for name in history]
    sources = [HERE / n for n in ('run_experiment.py', 'test_070.py', 'config.json', 'test_protocol.txt')]
    for folder, test in ((OLD, 'test_069.py'), (e.OLD, 'test_068.py'), (e.e.OLD, 'test_067.py')):
        sources += [folder / n for n in ('run_experiment.py', test, 'config.json', 'test_protocol.txt')]
    root = e.e.e.ROOT
    sources += [Path(base.__file__), root / 'src/emo_rec/normalization.py', root / 'src/emo_rec/data.py', root / 'src/emo_rec/submission.py', root / 'pyproject.toml', root / '数据/数据集说明文档.md']
    sources += [HERE.parent / '058_pe_split_replication' / n for n in ('run_experiment.py', 'config.json', 'test_protocol.txt')]
    prov = {'inputs': {str(p): base.sha256(p) for p in inputs}, 'sources': {str(p): base.sha256(p) for p in sources},
            'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
            'platform': platform.platform(), 'raw_shape': list(raw.shape), 'public_arrays_loaded': False,
            'cpu_only': 'sklearn HGB has no GPU implementation; frozen-model CPU exception',
            'n_jobs': config['n_jobs'], 'threads_per_job': config['threads_per_job'],
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'new_seeds_have_historical_oof': False}
    for name, folds in splits:
        base.assert_subject_isolation(subjects, folds)
        ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    print('Frozen 069 method; new 058 seeds; 2760 rows; 60 fits; no public arrays', flush=True)
    with threadpool_limits(limits=1):
        prov['active_threadpools'] = threadpool_info()
        results = Parallel(n_jobs=config['n_jobs'], prefer='threads')(delayed(job)(a40, a8, y, name, folds, fold, context, config) for name, folds in splits for fold in range(5) for context in (40, 8))
    assert len(results) == config['expected_model_fits']
    prov['model_fits'] = len(results)
    runs, rows, deltas, checks = {}, [], [], {}
    for name, folds in splits:
        frame = ref.assign(fold=folds, p_train40=np.nan, p_train8=np.nan)
        for n, fold, context, p, seconds, n_train in results:
            if n == name:
                frame.loc[folds == fold, f'p_train{context}'] = p
        assert np.isfinite(frame[['p_train40', 'p_train8']]).all().all()
        if name == 'fixed002':
            for folder in (OLD, e.OLD):
                old = pd.read_csv(folder / 'predictions/fixed002_oof.csv')
                for col in ('row_index', 'position', 'subject', 'group', 'trial', 'y', 'fold'):
                    np.testing.assert_array_equal(frame[col], old[col])
                for col in ('p_train40', 'p_train8'):
                    diff = float(np.max(np.abs(frame[col] - old[col])))
                    checks[f'{folder.name}/{col}'] = diff
                    assert diff <= 1e-12
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
    metrics = {'runs': runs, 'aggregate': aggregate, 'acceptance': decision,
               'fixed002_max_abs_diff': checks, 'historical_069_vs068_control_max_abs_diff': history,
               'elapsed_seconds': time.perf_counter() - start,
               'fit_seconds': [{'split': r[0], 'fold': r[1], 'train_context': r[2], 'seconds': r[4], 'n_train': r[5]} for r in results], 'provenance': prov}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(deltas).to_csv(HERE / 'paired_delta.csv', index=False)
    print(json.dumps({'elapsed_seconds': metrics['elapsed_seconds'], 'fixed002_reproduction': checks, 'historical_controls': history, 'acceptance': decision, 'all_rows': aggregate['all_rows']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
