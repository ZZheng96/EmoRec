import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('experiment051', HERE / 'run_experiment.py')
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))


def synthetic():
    rng = np.random.default_rng(51)
    subjects = np.repeat([f'{g}{i}' for g in ('HC', 'DEP') for i in range(10)], 6)
    groups = np.array(['HC' if s.startswith('HC') else 'DEP' for s in subjects])
    y = np.tile([0, 1], len(subjects) // 2)
    X = rng.normal(size=(len(y), 24)); X[:, 0] += 2 * y
    config = dict(CONFIG, inner_top_k=10, max_selected=10, hgb=dict(CONFIG['hgb'], max_iter=3))
    folds = exp.base.make_subject_folds(subjects, groups, 2027)
    return X, y, subjects, groups, np.array([f'f{i}' for i in range(24)]), config, folds


def test_inner_isolation_and_training_labels_only(monkeypatch):
    X, y, subjects, groups, names, config, folds = synthetic()
    train = folds != 0
    original = exp.LogisticRegression.fit
    observed = []
    def spy(self, x, labels, *args, **kwargs):
        observed.append(labels.copy())
        return original(self, x, labels, *args, **kwargs)
    monkeypatch.setattr(exp.LogisticRegression, 'fit', spy)
    with threadpool_limits(limits=1):
        selected, evidence = exp.select_stable(X[train], y[train], subjects[train], groups[train], names, config)
    inner_folds = exp.base.make_subject_folds(subjects[train], groups[train], config['inner_seed'])
    assert len(observed) == 5 and 0 < len(selected) <= 10
    for i, item in enumerate(evidence['inner_folds']):
        assert np.array_equal(observed[i], y[train][inner_folds != i])
        assert not set(item['training_subjects']) & set(subjects[~train])
        assert not set(item['held_out_subjects']) & set(subjects[~train])
        assert not set(item['training_subjects']) & set(item['held_out_subjects'])
    assert np.all(np.array(evidence['frequencies'])[selected] >= 0.6)


def test_validation_labels_do_not_change_predictions():
    X, y, subjects, groups, names, config, folds = synthetic()
    train, valid = folds != 0, folds == 0
    changed = y.copy(); changed[valid] = 1 - changed[valid]
    with threadpool_limits(limits=1):
        first, evidence = exp.predict_outer(X, y, subjects, groups, names, train, valid, config, 2026)
        second, other = exp.predict_outer(X, changed, subjects, groups, names, train, valid, config, 2026)
    assert evidence == other
    for method in exp.METHODS:
        np.testing.assert_array_equal(first[method], second[method])


def test_real_dimensions_finite_fixed_folds_and_oof_coverage():
    X, XP, ref, names, spectral = exp.load_data(CONFIG)
    assert X.shape == (2760, 600) and XP.shape == (80, 600)
    assert len(names) == 600 and np.isfinite(X).all() and np.isfinite(XP).all()
    assert 'HC43' not in set(ref.subject_id)
    for folds in [ref.fold.to_numpy()] + [exp.base.make_subject_folds(ref.subject_id.to_numpy(), ref.health_group.to_numpy(), seed) for seed in CONFIG['repeat_seeds']]:
        exp.base.assert_subject_isolation(ref.subject_id.to_numpy(), folds)
        counts = np.zeros(2760, dtype=int)
        for fold in range(5):
            counts[folds == fold] += 1
        assert np.all(counts == 1)
    frame = exp.public_frame(spectral, np.full(80, 0.5))
    assert frame.shape == (80, 3)


def test_saved_artifacts_when_available():
    if not (HERE / 'metrics.json').exists():
        return
    for method in exp.METHODS:
        for split in ['fixed002'] + [f'repeat{seed}' for seed in CONFIG['repeat_seeds']]:
            frame = pd.read_csv(HERE / f'predictions/{method}_{split}_oof.csv')
            assert len(frame) == 2760 and np.isfinite(frame.prob_positive).all()
            assert frame.prob_positive.between(0, 1).all()
            reference = pd.read_csv(HERE / CONFIG['fixed_reference'])
            columns = ['subject_id', 'health_group', 'original_trial_id', 'y_true']
            pd.testing.assert_frame_equal(frame[columns], reference[columns])
            assert frame.groupby('subject_id').size().eq(40).all()
            exp.base.assert_subject_isolation(frame.subject_id.to_numpy(), frame.fold.to_numpy())
        frame = pd.read_csv(HERE / f'predictions/{method}_public.csv')
        exp.validate_submission(frame, [f'P_test{i}' for i in range(1, 11)])
