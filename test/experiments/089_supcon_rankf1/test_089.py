import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = __import__('importlib').util.spec_from_file_location('exp089', HERE / 'run_experiment.py')
e = __import__('importlib').util.module_from_spec(spec); spec.loader.exec_module(e)
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))


def test_trial_keys_and_data():
    X, ref, folds = e.load_data(CFG)
    assert X.shape == (2760, 720)
    keys = e.trial_keys(ref)
    assert pd.Series(keys).value_counts().eq(5).all()
    for fold in range(5): e.base067.base.assert_subject_isolation(ref.subject_id.to_numpy(), folds)
    assert not CFG['public_access']


def test_validation_labels_do_not_change_prediction(monkeypatch):
    if not e.torch.cuda.is_available(): pytest.skip('CUDA required')
    rng = np.random.default_rng(89)
    subjects = np.repeat([f'S{i}' for i in range(10)], 40)
    y = np.tile(np.repeat([0, 1], 20), 10)
    keys = np.array([f'{s}_T{(j // 5) % 8}' for s in [f'S{i}' for i in range(10)] for j in range(40)])
    X = rng.normal(size=(len(y), 720)).astype(np.float32)
    train = np.array([not s.endswith('0') for s in subjects])
    a = e.fit_predict(X, y, keys, train, X[~train], {**CFG, 'epochs': 1}, 3089)
    changed = y.copy(); changed[~train] = 1 - changed[~train]
    b = e.fit_predict(X, changed, keys, train, X[~train], {**CFG, 'epochs': 1}, 3089)
    np.testing.assert_array_equal(a, b)


def test_saved_artifacts_if_present():
    if not (HERE / 'metrics.json').exists(): pytest.skip('pre-run')
    metrics = json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))
    assert metrics['provenance']['public_arrays_loaded'] is False
    for s in ['fixed002'] + [f'repeat{x}' for x in CFG['repeat_seeds']]:
        frame = pd.read_csv(HERE / 'predictions' / f'{s}_oof.csv')
        assert len(frame) == 2760 and frame.prob_positive.between(0, 1).all()
