import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('e096',HERE/'run_experiment.py'); e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
def test_config_and_candidate_space():
 c=json.loads((HERE/'config.json').read_text()); assert c['target_subject']=='DEP01'; assert len(e.CANDIDATES)==32; assert c['direction']=='raw'
def test_data_and_trial_balance():
 c=json.loads((HERE/'config.json').read_text()); x,r=e.load_data(c); s=r[r.subject=='DEP01'].reset_index(drop=True); assert x.shape==(2760,720); assert len(s)==40; k=e.trial_keys(s); assert len(np.unique(k))==8; assert all((k==q).sum()==5 for q in np.unique(k)); assert set(s.y)=={0,1}
def test_aggregation_catches_overwrite():
 masks=np.ones((4,3),dtype=bool); vals=[np.array([1.,2.,3.]),np.array([3.,4.,5.]),np.array([5.,6.,7.]),np.array([7.,8.,9.])]
 out,count=e.aggregate_pair_predictions(vals,masks)
 assert np.array_equal(count,[4,4,4]) and np.allclose(out,[4.,5.,6.]) and not np.allclose(out,vals[-1])

def test_protocol_and_oof():
 p=HERE/'predictions/DEP01_balanced_pairs_oof.csv'; long=HERE/'predictions/fold_predictions.csv'
 if not p.exists(): return
 d=pd.read_csv(p); assert len(d)==40 and d.prob_positive.between(0,1).all() and d.row_index.is_unique; assert d.prediction_count.eq(4).all()
 if long.exists():
  q=pd.read_csv(long); assert len(q)==160 and set(q.columns)=={'row_index','fold','y','prob'}; assert q.groupby('row_index').size().eq(4).all(); assert q.fold.nunique()==16
  avg=q.groupby('row_index').prob.mean().sort_index().to_numpy(); yy=q.groupby('row_index').y.first().sort_index().to_numpy(); assert np.isclose(e.metric(yy,avg)['auc'],json.loads((HERE/'metrics.json').read_text())['pooled_oof']['auc'])

def test_fold_metadata():
 p=HERE/'input_metadata/folds.json'
 if not p.exists(): return
 rows=json.loads(p.read_text()); assert len(rows)==16; assert all(r['train_rows']==30 and r['valid_rows']==10 and r['train_positive']==15 and r['train_negative']==15 for r in rows)
