import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e125', HERE/'run_experiment.py'); e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)

def test_trial_isolation_and_pair_counts():
    _, y, trial, _ = e.load_hc21()
    assert len(np.unique(trial)) == 8 and np.bincount(trial).tolist() == [5]*8
    pairs = list(e.trial_pairs(np.arange(8)))
    assert len(pairs) == 16
    for tr, va in pairs:
        assert not set(tr) & set(va) and len(tr) == 6 and len(va) == 2

def test_inner_selection_training_fold_isolation():
    selections = json.loads((HERE/'input_metadata/selections.json').read_text())
    for item in selections:
        assert set(item['train_trial_indices']).isdisjoint(item['valid_trial_indices'])
        assert item['train_window_count'] == 30 and item['valid_window_count'] == 10
        assert len(item['inner_scores']) == 240
        assert all(set(r['direction'] for r in item['inner_scores']) == {'raw','reverse'} for _ in [0])

def test_predictions_and_metrics_recompute():
    out = pd.read_csv(HERE/'predictions/HC21_window_csp_pairs_oof.csv'); m = json.loads((HERE/'metrics.json').read_text())
    assert len(out) == 8 and out.prediction_count.eq(4).all()
    assert np.isclose(m['csp_window_pairs']['auc'], roc_auc_score(out.y, out.prob_positive))
    assert np.isclose(m['csp_window_pairs']['f1_quota4'], f1_score(out.y, out.prediction_quota4))
    base = pd.read_csv(HERE/'predictions/baseline_720_linearsvc_oof.csv')
    assert len(base) == 40 and base.fold.nunique() == 8 and base.prob_positive.between(0,1).all()

def test_quota_has_no_labels():
    assert e.quota_predictions(np.array([.1,.8,.2,.9,.3,.7,.4,.6])).tolist() == [0,1,0,1,0,1,0,1]
    assert e.quota_predictions(np.full(8,.5)).sum() == 4
    cfg = json.loads((HERE/'config.json').read_text())
    assert cfg['quota_search'] is False and cfg['outer_validation_labels_used_for_selection'] is False
