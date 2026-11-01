import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp070', HERE / 'run_experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
spec = importlib.util.spec_from_file_location('tests069', e.OLD / 'test_069.py')
t069 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t069)


@pytest.fixture(scope='module')
def data():
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    return config, *e.load_data(config)


def test_real_mapping(data):
    t069.test_real_mapping(data)


def test_normalization_reference(data):
    t069.test_normalization_reference(data)


def test_no_public_array_access(data, monkeypatch):
    t069.test_no_public_array_access(data, monkeypatch)


def test_subject_and_context_isolation(data):
    t069.test_training_context_isolation(data)


def test_all_training_rows(data, monkeypatch):
    t069.test_job_retains_all_training_rows(data, monkeypatch)


def test_frozen_protocol_and_60_fits(data):
    config, _, ref, fixed = data
    e.validate_config(config)
    splits = e.splits_for(ref, fixed, config)
    jobs = [(name, fold, context) for name, _ in splits for fold in range(5) for context in config['train_contexts']]
    assert len(set(jobs)) == config['expected_model_fits'] == 60
    for name, folds in splits:
        e.base.assert_subject_isolation(ref.subject.to_numpy(), folds)
        counts = ref.assign(fold=folds).drop_duplicates('subject').groupby(['fold', 'group']).size().unstack()
        assert (counts.max() - counts.min() <= 1).all()


def test_historical_controls():
    assert len(e.historical_checks()) == 6


@pytest.fixture(scope='module')
def artifacts():
    if not (HERE / 'metrics.json').exists():
        pytest.skip('Pre-run: output artifacts not yet generated')
    return json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))


def test_fixed002_rowwise_reproduction(artifacts):
    frame = pd.read_csv(HERE / 'predictions/fixed002_oof.csv')
    for folder in (e.OLD, e.e.OLD):
        old = pd.read_csv(folder / 'predictions/fixed002_oof.csv')
        pd.testing.assert_frame_equal(frame, old, atol=1e-12, rtol=0)
    assert max(artifacts['fixed002_max_abs_diff'].values()) <= 1e-12


def test_oof_recompute(data, artifacts):
    config, _, ref, fixed = data
    metrics = artifacts
    fits = metrics['fit_seconds']
    assert len(fits) == len({(v['split'], v['fold'], v['train_context']) for v in fits}) == 60
    recomputed, delta_rows, summary_rows = {}, [], []
    for name, expected_fold in e.splits_for(ref, fixed, config):
        frame = pd.read_csv(HERE / 'predictions' / f'{name}_oof.csv')
        for col in ref.columns:
            np.testing.assert_array_equal(frame[col], ref[col])
        np.testing.assert_array_equal(frame.fold, expected_fold)
        pd.testing.assert_frame_equal(pd.read_csv(HERE / 'input_metadata' / f'folds_{name}.csv'), ref.assign(fold=expected_fold))
        assert not frame.row_index.duplicated().any()
        assert (frame[['p_train40', 'p_train8']].ge(0) & frame[['p_train40', 'p_train8']].le(1)).all().all()
        for fit in fits:
            if fit['split'] == name:
                assert fit['n_train'] == int((expected_fold != fit['fold']).sum())
        recomputed[name] = {}
        for subset in ['all_rows'] + [str(i) for i in config['positions']]:
            selected = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            recomputed[name][subset] = {}
            for context in ('40', '8'):
                actual = e.summarize(selected, context)
                assert actual == metrics['runs'][name][subset][context]
                recomputed[name][subset][context] = actual
                summary_rows.append({'split': name, 'position': subset, 'train_context': int(context), 'validation_context': 8, **e.scalars(actual)})
            delta_rows.append({'split': name, 'position': subset, **{k: e.scalars(recomputed[name][subset]['8'])[k] - e.scalars(recomputed[name][subset]['40'])[k] for k in e.KEYS}})
    aggregate = e.aggregate_runs(recomputed, config)
    assert aggregate == metrics['aggregate']
    assert e.acceptance(aggregate, config) == metrics['acceptance']
    for filename, rows in [('paired_delta.csv', delta_rows), ('summary.csv', summary_rows)]:
        pd.testing.assert_frame_equal(pd.read_csv(HERE / filename, dtype={'position': str}), pd.DataFrame(rows), atol=1e-14, rtol=0)


def test_input_source_hashes(artifacts):
    prov = artifacts['provenance']
    assert prov == json.loads((HERE / 'input_metadata/provenance.json').read_text(encoding='utf8'))
    for section in ('inputs', 'sources'):
        for path, digest in prov[section].items():
            assert e.base.sha256(Path(path)) == digest
    assert str(HERE / 'test_070.py') in prov['sources']
    assert prov['model_fits'] == 60 and prov['public_arrays_loaded'] is False
    assert prov['new_seeds_have_historical_oof'] is False
    assert not list((HERE / 'predictions').glob('*public*'))
