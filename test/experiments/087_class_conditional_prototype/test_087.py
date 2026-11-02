import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp087', HERE / 'run_experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def test_config_and_shapes():
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    e.validate_config(config)
    old_config = json.loads((e.OLD / 'config.json').read_text(encoding='utf8'))
    old_config.update({'excluded_subject': config['excluded_subject']})
    with e.public_guard():
        raw, ref, fixed = e.load_data(old_config)
    _, a8 = e.contexts(raw, ref.subject.to_numpy(), ref.position.to_numpy())
    assert raw.shape == (2760, 720)
    assert a8.shape == raw.shape
    assert config['expected_model_fits'] == 60
    assert config['expected_stage_fits'] == {'first_stage_raw': 30, 'second_stage_treatment': 30}
    assert config['public_access'] is False


def test_public_guard_and_metadata_checks():
    import pytest
    with e.public_guard():
        with pytest.raises(AssertionError):
            np.load('public_forbidden.npz')
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    old_config = json.loads((e.OLD / 'config.json').read_text(encoding='utf8'))
    old_config.update({'excluded_subject': config['excluded_subject']})
    with e.public_guard():
        _, ref, fixed = e.load_data(old_config)
    for name, folds in e.splits_for(ref, fixed, config):
        old = pd.read_csv(e.OLD / 'predictions' / f'{name}_oof.csv')
        e.validate_metadata(old, ref.assign(fold=folds))
        e.base.assert_subject_isolation(ref.subject.to_numpy(), folds)


def test_residual_feature_is_one_scalar_and_no_validation_fit():
    x = np.arange(24, dtype=float).reshape(3, 8)
    p = np.vstack([x[0], x[1]])
    out = e.residual_features(x, p, np.array([0, 1, 0]))
    assert out.shape == (3, 1)
    np.testing.assert_allclose(out[0], [0])


def test_generated_artifacts():
    metrics_path = HERE / 'metrics.json'
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding='utf8'))
    assert len(metrics['fit_seconds']) == 60
    assert metrics['provenance']['model_fits'] == 60
    assert metrics['provenance']['stage_fits'] == {'first_stage_raw': 30, 'second_stage_treatment': 30}
    assert metrics['provenance']['public_arrays_loaded'] is False
    assert metrics['provenance']['public_guard'] == 'active around load_data'
    for section in ('inputs', 'sources'):
        for path, digest in metrics['provenance'][section].items():
            assert e.sha256(path) == digest
    from sklearn.metrics import roc_auc_score, f1_score
    values = {'control': {'auc': [], 'f1': []}, 'treatment': {'auc': [], 'f1': []}}
    assert len(metrics['first_stage_control_max_abs_diff']) == 30
    assert max(metrics['first_stage_control_max_abs_diff'].values()) <= 1e-12
    assert metrics['provenance']['guard_evidence']['accessed_keys']
    for name in ['fixed002'] + [f'repeat{s}' for s in [2042, 2043, 2044, 2045, 2046]]:
        frame = pd.read_csv(HERE / 'predictions' / f'{name}_oof.csv')
        old = pd.read_csv(e.OLD / 'predictions' / f'{name}_oof.csv')
        e.validate_metadata(frame, old)
        np.testing.assert_allclose(frame.p_first_stage, old.p_train8, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(frame.p_control, old.p_train8)
        for arm in values:
            p = frame[f'p_{arm}']
            actual = {'auc': roc_auc_score(frame.y, p), 'f1': f1_score(frame.y, p >= .5)}
            expected = e.scalars(metrics['runs'][name]['all_rows'][arm])
            for key, value in actual.items():
                np.testing.assert_allclose(value, expected[key], rtol=0, atol=1e-14)
                if name != 'fixed002':
                    values[arm][key].append(value)
        assert len(frame) == 2760
        assert frame.row_index.is_unique
        assert np.isfinite(frame[['p_control', 'p_treatment']]).all().all()
        assert ((frame[['p_control', 'p_treatment']] >= 0) & (frame[['p_control', 'p_treatment']] <= 1)).all().all()
    for arm in values:
        for key, numbers in values[arm].items():
            recorded = metrics['aggregate']['all_rows'][arm][key]
            np.testing.assert_allclose(numbers, recorded['values'], atol=1e-14)
            np.testing.assert_allclose(np.mean(numbers), recorded['mean'], atol=1e-14)
            np.testing.assert_allclose(np.std(numbers, ddof=1), recorded['std'], atol=1e-14)
    for key in ('auc', 'f1'):
        delta = np.array(values['treatment'][key]) - values['control'][key]
        np.testing.assert_allclose(delta, metrics['aggregate']['all_rows']['delta_treatment_minus_control'][key]['values'], atol=1e-14)
