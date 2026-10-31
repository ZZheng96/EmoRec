import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('experiment044', HERE / 'run_experiment.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


@pytest.fixture
def eeg():
    return np.random.default_rng(44).normal(size=(3, 30, 2500))


def test_dimensions_finite_and_bounds(eeg):
    f = run.extract_wpli(eeg)
    assert f.shape == (3, 50)
    assert np.isfinite(f).all() and ((f >= 0) & (f <= 1)).all()
    np.testing.assert_array_equal(run.extract_wpli(np.zeros((1, 30, 2500))), 0)


def test_label_and_metadata_perturbation(eeg):
    data = {'eeg': eeg, 'label': np.array([0, 1, 0]), 'trial_id': np.arange(3)}
    before = run.extract_wpli(data['eeg'])
    data['label'] = 1-data['label']; data['trial_id'] += 99
    np.testing.assert_array_equal(before, run.extract_wpli(data['eeg']))
    import inspect
    assert list(inspect.signature(run.extract_wpli).parameters) == ['segments']


def test_window_independence_and_permutation(eeg):
    original = run.extract_wpli(eeg)
    np.testing.assert_allclose(original[0], run.extract_wpli(eeg[:1])[0], atol=1e-14)
    changed = eeg.copy(); changed[1:] *= 10
    np.testing.assert_array_equal(original[0], run.extract_wpli(changed)[0])
    np.testing.assert_allclose(original[::-1], run.extract_wpli(eeg[::-1]), atol=1e-14)


def test_known_phase_lag():
    t = np.arange(2500)/250
    x = np.zeros((1, 30, 2500))
    x[0, 0] = np.sin(2*np.pi*10*t)
    x[0, 1] = np.sin(2*np.pi*10*t + np.pi/3)
    assert run.extract_wpli(x).reshape(1, 10, 5)[0, 0, 2] > .95


def test_subject_disjoint_all_schedules():
    data = run.base.load_cache((HERE / run.CONFIG['spectral_cache']).resolve())
    keep = data['subject_ids'] != 'HC43'
    schedules = [run.base.make_subject_folds(data['subject_ids'], data['health_groups'], 2026)[keep]]
    schedules += [run.base.make_subject_folds(data['subject_ids'][keep], data['health_groups'][keep], seed) for seed in run.CONFIG['random_seeds']]
    for folds in schedules:
        run.base.assert_subject_isolation(data['subject_ids'][keep], folds)
        assert len(folds) == 2760 and len(np.unique(folds)) == 5


def test_oof_rows_and_public_format():
    data = run.base.load_cache((HERE / run.CONFIG['spectral_cache']).resolve())
    keep = data['subject_ids'] != 'HC43'
    subjects = data['subject_ids'][keep]
    folds = run.base.make_subject_folds(subjects, data['health_groups'][keep], 2027)
    frame = pd.DataFrame({'subject_id': subjects, 'fold': folds, 'prob_positive': .5})
    run.validate_oof(frame, 2760)
    with pytest.raises(AssertionError):
        run.validate_oof(frame.iloc[:-1], 2760)
    public = run.public_frame(data, np.full(80, .5))
    assert len(public) == 80 and list(public.columns) == ['user_id', 'trial_id', 'prob. of positive emotion']


def test_scaler_fit_training_only():
    model = run.base.make_model(run.CONFIG, 2026)
    x = np.arange(120).reshape(40, 3)
    model.fit(x, np.tile([0, 1], 20))
    before = model[0].mean_.copy()
    model.predict_proba(x + 1000)
    np.testing.assert_array_equal(before, x.mean(axis=0))
    np.testing.assert_array_equal(model[0].mean_, before)


def test_actual_artifacts_when_present():
    if not (HERE / 'metrics.json').exists():
        return
    for path in (HERE / 'predictions').glob('*_oof.csv'):
        run.validate_oof(pd.read_csv(path), 2760)
    assert len(list((HERE / 'predictions').glob('*_oof.csv'))) == 12
    for path in (HERE / 'predictions').glob('*_public.csv'):
        run.validate_submission(pd.read_csv(path), [f'P_test{i}' for i in range(1, 11)])
