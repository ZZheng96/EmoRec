import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp067', HERE / 'run_experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


@pytest.fixture(scope='module')
def data():
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    return config, *e.load_data(config)


def test_real_mapping(data):
    _, raw, ref, _ = data
    from emo_rec.data import load_train_segments
    files = sorted((e.ROOT / '数据/训练集').glob('*/*.mat'))
    assert len(files) == 70
    for path in files:
        if path.stem == 'HC43':
            continue
        _, y, trial = load_train_segments(path)
        rows = ref[ref.subject == path.stem]
        np.testing.assert_array_equal(rows.y, y)
        np.testing.assert_array_equal(rows.trial, trial)
        np.testing.assert_array_equal(rows.position, np.tile(np.arange(5), 8))
    assert raw.shape == (2760, 720)
    np.testing.assert_array_equal(np.sort(ref.row_index), np.arange(2760))
    assert not ref.row_index.duplicated().any()
    for pos in range(5):
        rows = ref[ref.position == pos]
        assert len(rows) == 552
        assert (rows.groupby('subject').size() == 8).all()
        assert (rows.groupby(['subject', 'y']).size() == 4).all()


def test_normalization_reference(data):
    _, raw, ref, _ = data
    subjects, pos = ref.subject.to_numpy(), ref.position.to_numpy()
    a40, a8 = e.contexts(raw, subjects, pos)
    for s in np.unique(subjects):
        for p in (None, 0, 1, 2, 3, 4):
            mask = (subjects == s) if p is None else (subjects == s) & (pos == p)
            x = raw[mask]
            std = x.std(axis=0, ddof=0)
            expected = (x - x.mean(axis=0)) / np.where(std > 1e-12, std, 1.0)
            np.testing.assert_allclose((a40 if p is None else a8)[mask], expected, atol=1e-13, rtol=1e-13)
    constant = np.ones((8, 720))
    np.testing.assert_array_equal(e.normalize(constant, np.repeat('a', 8)), 0)
    assert tuple(inspect.signature(e.normalize).parameters) == ('raw', 'subjects')


def test_isolation(data):
    config, raw, ref, fixed = data
    subjects, pos = ref.subject.to_numpy(), ref.position.to_numpy()
    splits = [fixed] + [e.base.make_subject_folds(subjects, ref.group.to_numpy(), s) for s in config['repeat_seeds']]
    a40, a8 = e.contexts(raw, subjects, pos)
    for folds in splits:
        e.base.assert_subject_isolation(subjects, folds)
    folds = fixed
    changed = raw.copy()
    changed[folds == 0] = changed[folds == 0] * 2 + np.arange(720)
    b40, b8 = e.contexts(changed, subjects, pos)
    np.testing.assert_array_equal(a40[folds != 0], b40[folds != 0])
    np.testing.assert_array_equal(a8[folds != 0], b8[folds != 0])
    changed = raw.copy()
    changed[pos != 0] += 1000
    _, b8 = e.contexts(changed, subjects, pos)
    np.testing.assert_array_equal(a8[pos == 0], b8[pos == 0])


def test_no_public_array_access(data, monkeypatch):
    original = np.lib.npyio.NpzFile.__getitem__
    accessed = []
    def checked(self, key):
        accessed.append(key)
        assert 'public' not in key
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', checked)
    e.load_data(data[0])
    assert accessed and set(accessed) == set(e.TRAIN_KEYS)


def test_oof_recompute(data):
    config, _, ref, fixed = data
    metrics = json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))
    assert len(metrics['fit_seconds']) == 30
    assert len({(v['split'], v['fold']) for v in metrics['fit_seconds']}) == 30
    for name in ['fixed002'] + [f'repeat{s}' for s in config['repeat_seeds']]:
        frame = pd.read_csv(HERE / 'predictions' / f'{name}_oof.csv')
        old = pd.read_csv(e.OLD / 'predictions' / f'plus_pe_{name}_oof.csv')
        np.testing.assert_array_equal(frame.row_index, np.arange(2760))
        np.testing.assert_array_equal(frame.position, ref.position)
        assert (frame[['p_context40', 'p_context8']].ge(0) & frame[['p_context40', 'p_context8']].le(1)).all().all()
        np.testing.assert_allclose(frame.p_context40, old.prob_positive, atol=1e-12, rtol=0)
        expected_fold = fixed if name == 'fixed002' else e.base.make_subject_folds(ref.subject.to_numpy(), ref.group.to_numpy(), int(name[6:]))
        np.testing.assert_array_equal(frame.fold, expected_fold)
        for subset in ['all_rows'] + [str(i) for i in range(5)]:
            selected = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            for context in ('40', '8'):
                actual = e.summarize(selected, context)
                stored = metrics['runs'][name][subset][context]
                for k, v in e.scalars(actual).items():
                    assert v == pytest.approx(e.scalars(stored)[k], abs=1e-14)
                assert actual['overall'] == stored['overall']
    for subset, contexts in metrics['aggregate'].items():
        for context, values in contexts.items():
            for key, result in values.items():
                vals = []
                for seed in config['repeat_seeds']:
                    r = metrics['runs'][f'repeat{seed}'][subset]
                    vals.append(e.scalars(r[context])[key] if context != 'delta8_minus40' else e.scalars(r['8'])[key] - e.scalars(r['40'])[key])
                np.testing.assert_allclose(result['values'], vals, atol=1e-14, rtol=0)
                assert result['mean'] == pytest.approx(np.mean(vals), abs=1e-14)
                assert result['std'] == pytest.approx(np.std(vals, ddof=1), abs=1e-14)
    for path, digest in metrics['provenance']['sources'].items():
        assert e.base.sha256(Path(path)) == digest
    assert not list((HERE / 'predictions').glob('*public*'))
