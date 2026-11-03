import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exp124',HERE/'run_experiment.py'); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

def test_window_order_and_dimensions():
 x=np.arange(5*2,dtype=float).reshape(5,2); d=exp.dynamic(x)
 np.testing.assert_allclose(d['mean'],[4,5]); np.testing.assert_allclose(d['full'][2:4],[2.8284271247461903]*2); np.testing.assert_allclose(d['full'][-2:],[ -6,-6])

def test_dynamic_manual_recompute():
 x=np.array([[0.],[1.],[3.],[6.],[10.]])
 d=exp.dynamic(x)['full']; assert np.isclose(d[0],4); assert np.isclose(d[1],np.std(x[:,0])); assert np.isclose(d[2],2.5)

def test_pca_training_isolation_and_candidates():
 assert len(exp.CANDIDATES)==54 and len(exp.CANDIDATES)*2==108
 assert exp.CANDIDATES[0][0]=='mean' and exp.CANDIDATES[-1][2]==3

def test_outer_prediction_recompute():
 m=json.loads((HERE/'metrics.json').read_text()); out=pd.read_csv(HERE/'predictions/DEP02_trial_window_dynamics_oof.csv'); pairs=pd.read_csv(HERE/'predictions/pair_predictions.csv')
 assert len(out)==8 and len(pairs)==32 and out.prediction_count.eq(4).all(); np.testing.assert_allclose(out.prob_positive,pairs.groupby('trial_index').prob_positive.mean())
 assert np.isclose(m['pooled_oof']['auc'],roc_auc_score(out.y,out.prob_positive)); assert np.isclose(m['pooled_oof']['f1'],f1_score(out.y,out.prediction_quota4))

def test_selection_protocol():
 s=json.loads((HERE/'input_metadata/selections.json').read_text()); assert len(s)==16 and all(len(x['inner_scores'])==108 for x in s)
 assert all(x['selection_source'].startswith('inner') for x in s)

def test_metrics_provenance():
 m=json.loads((HERE/'metrics.json').read_text()); p=json.loads((HERE/'input_metadata/provenance.json').read_text()); assert m['candidate_count']==108 and p['outer_validation_labels_used_for_selection'] is False and p['window_order'].startswith('raw 40')
