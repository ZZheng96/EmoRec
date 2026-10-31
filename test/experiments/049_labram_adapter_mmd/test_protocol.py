import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('exp049',Path(__file__).with_name('run_experiment.py')); M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

def test_shapes_and_no_hc43():
    d=M.load_data(); assert d['X_train'].shape==(2760,200); assert d['S_train'].shape[1]==150; assert 'HC43' not in set(d['subject_ids'])

def test_subject_folds_disjoint():
    s=np.repeat([f'S{i}' for i in range(10)],2); g=np.repeat(['HC','DEP']*5,2); f=M.folds(s,g)
    for k in range(5): assert not (set(s[f==k]) & set(s[f!=k]))

def test_validation_label_perturbation_not_feature():
    rng=np.random.default_rng(1); x=rng.normal(size=(30,200)); y=np.array([0,1]*15); tr=np.arange(20); va=np.arange(20,30)
    a=M.train_adapter(x,y,x[va],x[:4],np.array([f'S{i}' for i in range(30)]),np.isin(np.arange(30),tr),3)[0]
    b=M.train_adapter(x,y.copy(),x[va],x[:4],np.array([f'S{i}' for i in range(30)]),np.isin(np.arange(30),tr),3)[0]
    assert np.array_equal(a,b)

def test_frozen_audit_and_metrics():
    assert M.classification_metrics(np.array([0,1]),np.array([.2,.8]))['roc_auc']==1.0
    assert M.run_condition.__name__=='run_condition'

def test_public_format_and_oof_count():
    ids=np.repeat([f'P_test{i}' for i in range(1,11)],8); trials=np.tile(np.arange(1,9),10); out=M.public_frame(ids,trials,np.linspace(0,1,80)); assert len(out)==80; assert list(out)==['user_id','trial_id','prob. of positive emotion']
