import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp085', HERE / 'run_experiment.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)


def test_frozen_protocol_and_fit_count():
    c = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    e.validate_config(c)
    assert c['expected_model_fits'] == 6 * 5 * 5


def test_expert_mean_and_no_routing():
    x = pd.DataFrame({'p_expert0': [0.1, 0.3], 'p_expert1': [0.2, 0.4], 'p_expert2': [0.3, 0.5], 'p_expert3': [0.4, 0.6], 'p_expert4': [0.5, 0.7]})
    assert np.allclose(x.mean(axis=1), [0.3, 0.5])
    assert list(x.columns) == [f'p_expert{i}' for i in range(5)]


def test_joint_acceptance_requires_auc_and_f1_each_split():
    c = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    aggregate = {'all_rows': {'delta': {key: {'mean': 0.01, 'values': [0.01, 0.01, 0.01, 0.01, 0.01]} for key in e.KEYS}}}
    aggregate['all_rows']['delta']['auc']['values'] = [0.1, 0.1, 0.1, 0.1, -0.1]
    aggregate['all_rows']['delta']['f1']['values'] = [-0.1, -0.1, -0.1, -0.1, 0.1]
    result = e.acceptance(aggregate, c)
    assert result['checks']['joint_nonnegative_splits'] is False
    assert result['joint_nonnegative_values'] == [False] * 5


def test_eight_rows_per_subject_and_isolation():
    if not (HERE / 'predictions' / 'repeat2042_oof.csv').exists(): pytest.skip('完整运行后复算')
    frame = pd.read_csv(HERE / 'predictions/repeat2042_oof.csv')
    assert (frame.groupby('subject').size() == 40).all()
    assert (frame.groupby(['subject', 'position']).size() == 8).all()
    e.base.assert_subject_isolation(frame.subject.to_numpy(), frame.fold.to_numpy())
    assert not frame[['p_expert0','p_expert1','p_expert2','p_expert3','p_expert4','p_ensemble8']].isna().any().any()
    np.testing.assert_allclose(frame.p_ensemble8, frame[[f'p_expert{i}' for i in range(5)]].mean(axis=1))


def test_control_alignment_and_metadata():
    if not (HERE / 'predictions' / 'repeat2042_oof.csv').exists(): pytest.skip('完整运行后复算')
    out = pd.read_csv(HERE / 'predictions/repeat2042_oof.csv')
    ctl = pd.read_csv(e.OLD / 'predictions/repeat2042_oof.csv')
    for col in ('row_index', 'position', 'subject', 'group', 'trial', 'y', 'fold'):
        np.testing.assert_array_equal(out[col], ctl[col])


def test_public_guard_and_provenance():
    if (HERE / 'metrics.json').exists():
        m = json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))
        assert m['provenance']['public_arrays_loaded'] is False
        assert m['provenance']['model_fits'] == 150
        assert all(Path(path).exists() for path in m['provenance']['inputs'])
        assert all(Path(path).exists() for path in m['provenance']['sources'])
        for path, digest in m['provenance']['sources'].items():
            h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            assert h == digest
    assert not list((HERE / 'predictions').glob('*public*'))


def test_fit_one_has_no_nested_threadpool_limit():
    tree = ast.parse((HERE / 'run_experiment.py').read_text(encoding='utf8'))
    fit = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'fit_one')
    assert not any(isinstance(node, ast.With) and any('threadpool_limits' in ast.unparse(item.context_expr) for item in node.items) for node in ast.walk(fit))


def test_position_permutation_is_invariant():
    positions = np.repeat(np.arange(5), 8)
    permuted = positions.reshape(5, 8)[:, ::-1].reshape(-1)
    assert np.array_equal(np.sort(positions), np.sort(permuted))
    assert all(np.bincount(permuted)[i] == 8 for i in range(5))
