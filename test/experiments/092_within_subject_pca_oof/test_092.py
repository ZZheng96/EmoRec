import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp092', HERE / 'run_experiment.py'); e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)

def test_data_shape_and_no_public():
    c = json.loads((HERE/'config.json').read_text()); x, ref = e.load_data(c)
    assert x.shape == (2760, 720); assert ref.subject.nunique() == 69 and (ref.groupby('subject').size() == 40).all(); assert not any('public' in k.lower() for k in c)

def test_trial_isolation_and_counts():
    c = json.loads((HERE/'config.json').read_text()); _, ref = e.load_data(c); keys = e.trial_keys(ref)
    for subject in ref.subject.unique():
        k = keys[ref.subject.to_numpy() == subject]
        assert len(np.unique(k)) == 8 and set(np.unique(np.bincount(pd.factorize(k)[0]))) == {5}

def test_oof_complete_and_metadata():
    path = HERE/'predictions/within_subject_oof.csv'
    if not path.exists(): return
    p = pd.read_csv(path); assert len(p) == 2760 and p.row_index.is_unique
    for arm in e.ARMS: assert p['prob_' + arm].between(0, 1).all()
    assert p.groupby(['subject','fold']).size().eq(5).all() and p.groupby('subject').size().eq(40).all() and set(p.fold) == set(range(8))

def test_registered_arms_and_pca():
    c = json.loads((HERE/'config.json').read_text()); assert tuple(c['arms']) == e.ARMS; assert c['pca_components'] == 10 and c['lr_C'] == 1.0
