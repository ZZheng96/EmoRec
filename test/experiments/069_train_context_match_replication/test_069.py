import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp068', HERE / 'run_experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
spec068 = importlib.util.spec_from_file_location('tests068', e.OLD / 'test_068.py')
t067 = importlib.util.module_from_spec(spec068)
spec068.loader.exec_module(t067)


@pytest.fixture(scope='module')
def data():
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    return config, *e.load_data(config)


def test_real_mapping(data):
    t067.test_real_mapping(data)


def test_normalization_reference(data):
    t067.test_normalization_reference(data)


def test_no_public_array_access(data, monkeypatch):
    t067.test_no_public_array_access(data, monkeypatch)


def test_training_context_isolation(data):
    config, raw, ref, fixed = data
    subjects, pos = ref.subject.to_numpy(), ref.position.to_numpy()
    a40, a8 = e.contexts(raw, subjects, pos)
    for folds in [fixed] + [e.base.make_subject_folds(subjects, ref.group.to_numpy(), s) for s in config['repeat_seeds']]:
        e.base.assert_subject_isolation(subjects, folds)
        for fold in range(5):
            train = folds != fold
            train40, train8 = e.contexts(raw[train], subjects[train], pos[train])
            np.testing.assert_array_equal(train40, a40[train])
            np.testing.assert_array_equal(train8, a8[train])
    changed = raw.copy()
    changed[pos != 0] += 1000
    _, b8 = e.contexts(changed, subjects, pos)
    np.testing.assert_array_equal(a8[pos == 0], b8[pos == 0])


def test_job_retains_all_training_rows(data, monkeypatch):
    config, raw, ref, folds = data
    a40, a8 = e.contexts(raw, ref.subject.to_numpy(), ref.position.to_numpy())
    seen = []
    class Model:
        def fit(self, x, y):
            seen.append((x.copy(), y.copy()))
        def predict_proba(self, x):
            np.testing.assert_array_equal(x, a8[folds == 0])
            return np.tile([0.5, 0.5], (len(x), 1))
    monkeypatch.setattr(e.base, 'make_model', lambda config, seed: Model())
    for context, expected in ((40, a40), (8, a8)):
        result = e.job(a40, a8, ref.y.to_numpy(), 'fixed002', folds, 0, context, config)
        np.testing.assert_array_equal(seen[-1][0], expected[folds != 0])
        np.testing.assert_array_equal(seen[-1][1], ref.y.to_numpy()[folds != 0])
        assert result[5] == int((folds != 0).sum())
        assert (ref[folds != 0].groupby('subject').size() == 40).all()


def test_oof_recompute(data):
    config, _, ref, fixed = data
    metrics = json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))
    assert len(metrics['fit_seconds']) == 60
    assert len({(v['split'], v['fold'], v['train_context']) for v in metrics['fit_seconds']}) == 60
    recomputed = {}
    delta_rows = []
    for name in ['fixed002'] + [f'repeat{s}' for s in config['repeat_seeds']]:
        frame = pd.read_csv(HERE / 'predictions' / f'{name}_oof.csv')
        old = pd.read_csv(e.OLD / 'predictions' / f'{name}_oof.csv')
        np.testing.assert_array_equal(frame.row_index, np.arange(2760))
        for col in ('row_index', 'subject', 'group', 'trial', 'y', 'position'):
            np.testing.assert_array_equal(frame[col], ref[col])
        assert not frame.row_index.duplicated().any()
        assert (frame[['p_train40', 'p_train8']].ge(0) & frame[['p_train40', 'p_train8']].le(1)).all().all()
        np.testing.assert_allclose(frame.p_train40, old.p_train40, atol=1e-12, rtol=0)
        expected_fold = fixed if name == 'fixed002' else e.base.make_subject_folds(ref.subject.to_numpy(), ref.group.to_numpy(), int(name[6:]))
        np.testing.assert_array_equal(frame.fold, expected_fold)
        for fit in metrics['fit_seconds']:
            if fit['split'] == name:
                assert fit['n_train'] == int((expected_fold != fit['fold']).sum())
        recomputed[name] = {}
        for subset in ['all_rows'] + [str(i) for i in range(5)]:
            selected = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            recomputed[name][subset] = {}
            for context in ('40', '8'):
                actual = e.summarize(selected, context)
                assert actual == metrics['runs'][name][subset][context]
                recomputed[name][subset][context] = actual
            delta_rows.append({'split': name, 'position': subset, **{k: e.scalars(recomputed[name][subset]['8'])[k] - e.scalars(recomputed[name][subset]['40'])[k] for k in e.KEYS}})
    aggregate = e.aggregate_runs(recomputed, config)
    assert aggregate == metrics['aggregate']
    assert e.acceptance(aggregate, config) == metrics['acceptance']
    pd.testing.assert_frame_equal(pd.read_csv(HERE / 'paired_delta.csv', dtype={'position': str}), pd.DataFrame(delta_rows), atol=1e-14, rtol=0)
    for section in ('inputs', 'sources'):
        for path, digest in metrics['provenance'][section].items():
            assert e.base.sha256(Path(path)) == digest
    assert not list((HERE / 'predictions').glob('*public*'))
