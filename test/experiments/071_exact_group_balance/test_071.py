import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp071', HERE / 'run_experiment.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)


def test_exact_weights_independent():
    groups = np.array(['HC'] * 44 + ['DEP'] * 25)
    weights, evidence = e.exact_group_weights(groups)
    assert np.isclose(weights.mean(), 1)
    assert np.isclose(weights[groups == 'HC'].sum(), 34.5)
    assert np.isclose(weights[groups == 'DEP'].sum(), 34.5)
    assert evidence['HC']['weight'] == 69 / (2 * 44)
    assert evidence['DEP']['weight'] == 69 / (2 * 25)


def test_weight_is_training_fold_local():
    groups = np.array(['HC'] * 44 + ['DEP'] * 25)
    train = np.arange(69) % 5 != 0
    weights, _ = e.exact_group_weights(groups[train])
    assert not np.isclose(weights[groups[train] == 'HC'][0], 69 / (2 * 44))
    assert np.isclose(weights[groups[train] == 'HC'].sum(), train.sum() / 2)


def test_config_and_fits():
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    e.validate_config(config)
    assert config['expected_model_fits'] == 60
    assert config['train_contexts'] == [8]


def test_no_public_predictions():
    assert not list((HERE / 'predictions').glob('*public*'))


@pytest.fixture(scope='module')
def artifacts():
    p = HERE / 'metrics.json'
    if not p.exists():
        pytest.skip('Pre-run: output artifacts not yet generated')
    return json.loads(p.read_text(encoding='utf8'))


def test_artifact_aggregation_and_acceptance(artifacts):
    assert len(artifacts['fit_seconds']) == 60
    assert e.acceptance(artifacts['aggregate'], json.loads((HERE / 'config.json').read_text())) == artifacts['acceptance']
    assert artifacts['provenance']['public_arrays_loaded'] is False


def test_hashes(artifacts):
    for section in ('inputs', 'sources'):
        for path, digest in artifacts['provenance'][section].items():
            assert e.base.sha256(Path(path)) == digest


def test_oof_and_weight_evidence(artifacts):
    for split in ['fixed002'] + [f'repeat{s}' for s in [2042, 2043, 2044, 2045, 2046]]:
        frame = pd.read_csv(HERE / 'predictions' / f'{split}_oof.csv')
        assert len(frame) == 2760 and frame.row_index.is_unique
        assert np.isfinite(frame[['p_control', 'p_weighted']]).all().all()
        for fold in range(5):
            evidence = json.loads((HERE / 'input_metadata' / f'{split}_fold{fold}_weights.json').read_text())
            assert evidence['weights']['HC']['total_weight'] == pytest.approx(evidence['weights']['DEP']['total_weight'])
            assert evidence['subject_disjoint'] is True
