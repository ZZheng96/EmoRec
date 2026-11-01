import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '057_multiscale_permutation_entropy'
spec = importlib.util.spec_from_file_location('exp058', HERE / 'run_experiment.py')
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
SOURCE_CONFIG = json.loads((SOURCE / 'config.json').read_text(encoding='utf8'))


def test_new_seeds_and_frozen_parameters():
    assert CONFIG['split_seeds'] == [2042, 2043, 2044, 2045, 2046]
    assert CONFIG['methods'] == ['control_600', 'plus_pe']
    assert CONFIG['outer_model_seed_base'] == 2052
    assert CONFIG['threshold'] == 0.5
    assert SOURCE_CONFIG['hgb'] == json.loads((SOURCE.parent / '056_theta_beta_pac/config.json').read_text())['hgb']
    assert SOURCE_CONFIG['outer_model_seed_base'] == 2052


def test_column_sets_and_subject_isolation():
    assert {k: len(v) for k, v in exp.column_sets().items()} == {'control_600': 600, 'plus_pe': 720}
    ids = np.repeat([f's{i}' for i in range(20)], 4)
    groups = np.repeat(np.arange(20) % 2, 4)
    for seed in CONFIG['split_seeds']:
        folds = exp.base.make_subject_folds(ids, groups, seed)
        exp.base.assert_subject_isolation(ids, folds)
        assert set(folds) == set(range(5))


def test_saved_outputs_alignment_oof_recompute_and_decision():
    metrics_path = HERE / 'metrics.json'
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding='utf8'))
    assert metrics['provenance']['data_loaded_by'] == '057.load_data'
    assert metrics['provenance']['alignment_checked'] is True
    assert metrics['positive_auc_gain_count'] == sum(metrics['positive_auc_gain_by_split'].values())
    d = metrics['deltas_vs_control']['plus_pe']
    expected = d['auc'] >= .005 and d['f1'] >= 0 and d['dep_auc'] >= 0
    assert metrics['decision_accept'] == expected
    for seed in CONFIG['split_seeds']:
        frames = {}
        for method in CONFIG['methods']:
            frame = pd.read_csv(HERE / 'predictions' / f'{method}_split{seed}_oof.csv')
            assert len(frame) == 2760 and frame.subject_id.nunique() == 69
            exp.base.assert_subject_isolation(frame.subject_id.to_numpy(), frame.fold.to_numpy())
            assert frame.prob_positive.between(0, 1).all()
            frames[method] = frame
        np.testing.assert_array_equal(frames['control_600'][['subject_id', 'health_group', 'original_trial_id', 'fold', 'y_true']].to_numpy(), frames['plus_pe'][['subject_id', 'health_group', 'original_trial_id', 'fold', 'y_true']].to_numpy())
        for method, frame in frames.items():
            result = exp.base.summarize(frame.y_true.to_numpy(), frame.prob_positive.to_numpy(), frame, frame.fold.to_numpy())
            for key, value in exp.scalars(result).items():
                assert abs(value - exp.scalars(metrics['runs'][method][str(seed)])[key]) < 1e-12
    for method in CONFIG['methods']:
        for key in exp.KEYS:
            values = [exp.scalars(metrics['runs'][method][str(seed)])[key] for seed in CONFIG['split_seeds']]
            assert abs(np.mean(values) - metrics['aggregate'][method][key]['mean']) < 1e-12
            if method == 'plus_pe':
                expected_delta = np.mean(values) - metrics['aggregate']['control_600'][key]['mean']
                assert abs(expected_delta - d[key]) < 1e-12
