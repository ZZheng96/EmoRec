import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('e095',HERE/'run_experiment.py'); e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
def test_config_and_shape():
    c=json.loads((HERE/'config.json').read_text()); assert c['target_subject']=='DEP01'; x,r=e.load_data(c); assert x.shape==(2760,720); assert r.subject.nunique()==69; assert (r.groupby('subject').size()==40).all(); assert not (r.subject=='HC43').any()
def test_dep01_trials():
    c=json.loads((HERE/'config.json').read_text()); x,r=e.load_data(c); s=r[r.subject=='DEP01'].reset_index(drop=True); k=e.trial_keys(s); assert len(np.unique(k))==8 and set(np.unique(np.bincount(pd.factorize(k)[0])))=={5}
def test_oof():
    p=HERE/'predictions/DEP01_oof.csv'
    if not p.exists(): return
    d=pd.read_csv(p); assert len(d)==40 and d.prob_positive.between(0,1).all(); assert d.fold.nunique()==8 and d.groupby('fold').size().eq(5).all()
def test_candidate_space():
    assert len(e.CANDIDATES)==3*4*3*2*3
