import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('experiment054', HERE / 'run_experiment.py'); exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
FAST = {**CONFIG, 'hgb': {**CONFIG['hgb'], 'max_iter': 20}}


def synthetic():
    rng = np.random.default_rng(54); subs = [f'{g}{i}' for g in ('HC', 'DEP') for i in range(10)]
    rows = [(s, 'HC' if s.startswith('HC') else 'DEP', tid, lab) for s in subs for lab in (0, 1) for tid in range(1, 5) for _ in range(5)]
    ref = pd.DataFrame(rows, columns=['subject_id', 'health_group', 'original_trial_id', 'y_true'])
    y = ref.y_true.to_numpy(); X = rng.normal(size=(len(y), 10)); X[:, 0] += y
    return X, y, ref


def test_real_trial_keys_have_five_rows_and_subject_isolation():
    X, XP, ref, _ = exp.load_data(CONFIG); keys = exp.trial_keys(ref)
    assert X.shape == (2760, 600) and XP.shape == (80, 600) and 'HC43' not in set(ref.subject_id)
    assert pd.Series(keys).value_counts().eq(5).all() and len(set(keys)) == 69 * 8
    folds = [ref.fold.to_numpy()] + [exp.base.make_subject_folds(ref.subject_id.to_numpy(), ref.health_group.to_numpy(), s) for s in CONFIG['repeat_seeds']]
    for f in folds: exp.base.assert_subject_isolation(ref.subject_id.to_numpy(), f)


def test_validation_labels_do_not_change_fold_predictions():
    X, y, ref = synthetic(); s = ref.subject_id.to_numpy(); g = ref.health_group.to_numpy(); keys = exp.trial_keys(ref)
    folds = exp.base.make_subject_folds(s, g, 2027); train, valid = folds != 0, folds == 0
    changed = y.copy(); changed[valid] = 1 - changed[valid]
    a, ea = exp.fit_predict_fold(X, y, s, g, keys, train, X[valid], FAST, 2052)
    b, eb = exp.fit_predict_fold(X, changed, s, g, keys, train, X[valid], FAST, 2052)
    for m in exp.METHODS: np.testing.assert_array_equal(a[m], b[m])
    assert ea == eb and not set(ea['training_subjects']) & set(s[valid])


def test_teacher_depends_on_training_labels_and_soft_targets_in_range():
    X, y, ref = synthetic(); s = ref.subject_id.to_numpy(); g = ref.health_group.to_numpy(); keys = exp.trial_keys(ref)
    t, e = exp.teacher_targets(X, y, s, g, keys, FAST)
    assert ((t >= 0) & (t <= 1)).all() and pd.Series(t).groupby(keys).nunique().eq(1).all()
    t2, e2 = exp.teacher_targets(X, 1 - y, s, g, keys, FAST); assert not np.allclose(t, t2)
    w = exp.confidence_weights(y, t, FAST); assert np.isclose(w.mean(), 1) and (w > 0).all()


def test_saved_artifacts_when_available():
    if not (HERE / 'metrics.json').exists(): return
    for m in exp.METHODS:
        for split in ['fixed002'] + [f'repeat{s}' for s in CONFIG['repeat_seeds']]:
            frame = pd.read_csv(HERE / 'predictions' / f'{m}_{split}_oof.csv')
            assert len(frame) == 2760 and frame.prob_positive.between(0, 1).all() and frame.groupby('subject_id').size().eq(40).all()
            exp.base.assert_subject_isolation(frame.subject_id.to_numpy(), frame.fold.to_numpy())
        pub = pd.read_csv(HERE / 'predictions' / f'{m}_public.csv'); assert len(pub) == 80
    ev = json.loads((HERE / 'input_metadata' / 'fold_teacher_evidence.json').read_text(encoding='utf-8'))
    assert len(ev) == 30 and all(e['subject_disjoint'] for e in ev)
