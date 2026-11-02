import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp083', HERE / 'run_experiment.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

@pytest.fixture(scope='module')
def data():
    c = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    return c, m.load_data(c)

def test_frozen_shape_and_protocol(data):
    c, (raw, ref, fixed) = data
    m.validate_config(c)
    assert raw.shape == (2760, 720)
    assert ref.subject.nunique() == 69 and 'HC43' not in set(ref.subject)
    assert set(ref.position) == set(range(5))
    assert len(m.e.splits_for(ref, fixed, c)) == 6

def test_projection_orthogonal_idempotent_and_rank(data):
    c, (raw, ref, _) = data
    mask = np.ones(len(ref), dtype=bool); mask[::5] = False
    p, info = m.estimate_projection(raw, ref, mask, c['projection_norm_threshold'])
    np.testing.assert_allclose(p, p.T, atol=1e-10)
    np.testing.assert_allclose(p @ p, p, atol=1e-10)
    assert p.shape == (720, 720) and 0 <= info['rank'] <= 2

def test_training_only_no_validation_label_or_group(data):
    c, (raw, ref, _) = data
    mask = np.ones(len(ref), dtype=bool); mask[::5] = False
    p1, _ = m.estimate_projection(raw, ref, mask, c['projection_norm_threshold'])
    changed = ref.copy(); changed.loc[~mask, 'y'] = 1 - changed.loc[~mask, 'y']; changed.loc[~mask, 'group'] = 'DEP'
    p2, _ = m.estimate_projection(raw, changed, mask, c['projection_norm_threshold'])
    np.testing.assert_allclose(p1, p2, atol=0, rtol=0)


def test_fit_one_zscores_before_projection_and_never_after(monkeypatch, data):
    c, (raw, ref, fixed) = data
    folds = m.e.splits_for(ref, fixed, c)[0][1]
    captured = {}

    def fake_projection(x, frame, train_mask, threshold):
        captured['input'] = x.copy()
        captured['train_mask'] = train_mask.copy()
        return np.eye(x.shape[1]), {'rank': 0}

    class FakeModel:
        def fit(self, x, y):
            captured['fit'] = x.copy()
        def predict_proba(self, x):
            captured['predict'] = x.copy()
            return np.column_stack([np.zeros(len(x)), np.ones(len(x))])

    monkeypatch.setattr(m, 'estimate_projection', fake_projection)
    monkeypatch.setattr(m.base, 'make_model', lambda *args: FakeModel())
    result = m.fit_one(raw, ref, 'fixed002', folds, 0, c)
    assert result[5]['train_projected_norm'] == float(np.linalg.norm(captured['fit']))
    assert result[5]['valid_projected_norm'] == float(np.linalg.norm(captured['predict']))
    expected = m.a8_zscore(raw, ref.subject.to_numpy(), ref.position.to_numpy())
    np.testing.assert_allclose(captured['input'], expected)
    np.testing.assert_allclose(captured['fit'], expected[folds != 0])
    np.testing.assert_allclose(captured['predict'], expected[folds == 0])

def test_projection_shape_and_split_isolation(data):
    c, (raw, ref, fixed) = data
    for _, folds in m.e.splits_for(ref, fixed, c):
        m.base.assert_subject_isolation(ref.subject.to_numpy(), folds)
        p, _ = m.estimate_projection(raw, ref, folds != 0, c['projection_norm_threshold'])
        assert (raw @ p).shape == raw.shape

def test_control_070_and_no_public_array_access(monkeypatch, data):
    c, (raw, ref, fixed) = data
    for name, _ in m.e.splits_for(ref, fixed, c):
        frame = pd.read_csv(m.DEP / 'predictions' / f'{name}_oof.csv')
        assert len(frame) == 2760
    real = np.load
    class Guard:
        def __init__(self, *args, **kwargs): self.z = real(*args, **kwargs)
        def __enter__(self): return self
        def __exit__(self, *args): self.z.close()
        def __getitem__(self, key):
            assert key in {'X_train','subject_ids','health_groups','original_trial_ids','y_train','feature_names'}
            return self.z[key]
    guard = Guard
    monkeypatch.setattr(m.np, 'load', guard)
    m.load_data(c)

def test_acceptance_contract(data):
    c, _ = data
    d = {k:{'mean':0.006,'values':[0.006]*5} for k in m.KEYS}
    assert m.acceptance({'all_rows':{'delta_projection8_minus_control8':d}}, c)['accepted']

def test_artifacts_if_present(data):
    if not (HERE/'metrics.json').exists(): pytest.skip('OOF not run yet')
    metrics=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
    assert len(metrics['fit_seconds']) == 30
    assert metrics['provenance']['model_fits'] == 30
    assert metrics['provenance']['public_arrays_loaded'] is False
    assert max(metrics['control_070_max_abs_diff'].values()) <= 1e-12
    assert all(r['train_context'] == r['validation_context'] == 8 for r in metrics['fit_seconds'])
