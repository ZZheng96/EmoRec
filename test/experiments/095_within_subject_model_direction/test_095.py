import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exp095',HERE/'run_experiment.py'); e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
def test_data_shape_and_no_public():
 c=json.loads((HERE/'config.json').read_text()); x,ref=e.load_data(c); assert x.shape==(2760,720); assert ref.subject.nunique()==69 and (ref.groupby('subject').size()==40).all(); assert not any('public' in k.lower() for k in c)
def test_trial_counts():
 c=json.loads((HERE/'config.json').read_text()); _,ref=e.load_data(c); keys=e.trial_keys(ref)
 for s in ref.subject.unique():
  k=keys[ref.subject.to_numpy()==s]; assert len(np.unique(k))==8 and set(np.unique(np.bincount(pd.factorize(k)[0])))=={5}
def test_oof_complete():
 p=HERE/'predictions/within_subject_oof.csv'
 if not p.exists(): return
 d=pd.read_csv(p)
 if 'prob_linear_svc_raw' not in d: return
 assert len(d)==2760 and d.row_index.is_unique; assert d.prob_linear_svc_raw.between(0,1).all() and d.prob_adaptive.between(0,1).all(); assert d.groupby(['subject','fold']).size().eq(5).all() and set(d.fold)==set(range(8))
def test_candidates_and_params():
 c=json.loads((HERE/'config.json').read_text()); assert tuple(c['candidates'])==e.CANDIDATES; assert c['linear_svc']=={'C':1.0,'dual':'auto','max_iter':5000}; assert c['logistic_regression']=={'C':1.0,'max_iter':2000,'solver':'lbfgs'}
