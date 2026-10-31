import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('experiment043', HERE / 'run_experiment.py')
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))


def synthetic():
    rng = np.random.default_rng(43)
    a = rng.normal(size=(24, 5, 30, 32))
    return a @ a.transpose(0, 1, 3, 2), np.tile([0, 1], 12), np.repeat(['A', 'B', 'C'], 8)


def test_csp_training_only_and_validation_label_invariance():
    cov, y, subjects = synthetic()
    train = subjects != 'C'; valid = ~train
    original = exp.FilterBankCSP.fit
    seen = []
    def spy(self, x, labels):
        seen.append((x.copy(), labels.copy()))
        return original(self, x, labels)
    with patch.object(exp.FilterBankCSP, 'fit', spy):
        csp, model = exp.fit_csp_fold(cov, y, subjects, train, valid, CONFIG)
    np.testing.assert_array_equal(seen[0][0], cov[train])
    np.testing.assert_array_equal(seen[0][1], y[train])
    changed = y.copy(); changed[valid] = 1 - changed[valid]
    csp2, model2 = exp.fit_csp_fold(cov, changed, subjects, train, valid, CONFIG)
    np.testing.assert_array_equal(csp.filters_, csp2.filters_)
    np.testing.assert_allclose(model.predict_proba(csp.transform(cov[valid])), model2.predict_proba(csp2.transform(cov[valid])))


def test_subject_disjoint_and_fixed_fold_reproduction():
    spectral = exp.load_npz(HERE / CONFIG['spectral_cache'])
    keep = spectral['subject_ids'] != 'HC43'
    folds = exp.make_folds(spectral['subject_ids'], spectral['health_groups'], 2026)[keep]
    # Read only assignment/identity columns; never historical probabilities.
    import pandas as pd
    reference = pd.read_csv(HERE / '../002_subject_normalization/predictions/subject_zscore_exclude_hc43_oof.csv', usecols=['subject_id', 'fold'])
    np.testing.assert_array_equal(reference.subject_id, spectral['subject_ids'][keep])
    np.testing.assert_array_equal(reference.fold, folds)
    for seed in [2026, *CONFIG['repeat_seeds']]:
        f = folds if seed == 2026 else exp.make_folds(spectral['subject_ids'][keep], spectral['health_groups'][keep], seed)
        for fold in range(5):
            exp.assert_disjoint(spectral['subject_ids'][keep], f != fold, f == fold)
    with pytest.raises(ValueError):
        exp.assert_disjoint(np.array(['A', 'A']), np.array([True, False]), np.array([False, True]))


def test_feature_dimensions_finite_and_permutation():
    cov, y, _ = synthetic()
    csp = exp.FilterBankCSP().fit(cov, y)
    features = csp.transform(cov)
    assert features.shape == (24, 20) and np.isfinite(features).all()
    np.testing.assert_allclose(csp.transform(cov[::-1]), features[::-1])


def test_windows_are_filtered_independently():
    rng = np.random.default_rng(12)
    windows = rng.normal(size=(2, 30, 2500))
    together = exp.window_covariances(windows, CONFIG)
    np.testing.assert_allclose(together[:1], exp.window_covariances(windows[:1], CONFIG))
    windows[1] *= 100
    np.testing.assert_allclose(together[:1], exp.window_covariances(windows, CONFIG)[:1])


def test_public_format():
    data = {'public_user_ids': np.repeat([f'P_test{i}' for i in range(1, 11)], 8), 'public_trial_ids': np.tile(np.arange(1, 9), 10)}
    frame = exp.public_frame(data, np.full(80, .5))
    assert frame.shape == (80, 3)
    assert list(frame.columns) == ['user_id', 'trial_id', 'prob. of positive emotion']


def test_oof_row_count_and_unique_coverage():
    exp.check_oof(np.full(2760, .5), np.ones(2760))
    for p, visits in [(np.full(2759, .5), np.ones(2759)), (np.full(2760, np.nan), np.ones(2760)), (np.full(2760, .5), np.full(2760, 2))]:
        with pytest.raises(ValueError):
            exp.check_oof(p, visits)
