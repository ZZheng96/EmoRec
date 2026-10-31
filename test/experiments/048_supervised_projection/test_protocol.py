import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
spec=importlib.util.spec_from_file_location('exp048',Path(__file__).with_name('run_experiment.py')); M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def test_subject_folds_disjoint():
    s=np.repeat([f'S{i}' for i in range(10)],2);g=np.repeat(['HC','DEP']*5,2);f=M.folds(s,g,2026)
    for x in range(5): assert not (set(s[f==x]) & set(s[f!=x]))

def test_projection_uses_training_labels_only():
    rng=np.random.default_rng(4);X=rng.normal(size=(20,4)); y=np.array([0,1]*10); Xv=rng.normal(size=(3,4));Xp=rng.normal(size=(2,4))
    a=M.fit_project(X[:15],y[:15],Xv,Xp); b=M.fit_project(X[:15],y[:15][::-1],Xv,Xp)
    assert a[1].shape==(3,5) and a[2].shape==(2,5) and np.array_equal(a[1],b[1])

def test_validation_label_perturbation_does_not_change_projection():
    rng=np.random.default_rng(3);X=rng.normal(size=(20,4)); y=np.array([0,1]*10); tr=np.arange(15); va=np.arange(15,20)
    a=M.fit_project(X[tr],y[tr],X[va],X[va])[1]; b=M.fit_project(X[tr],y[tr],X[va],X[va])[1]; assert np.array_equal(a,b)

def test_dimensions_finite_and_oof_count():
    rng=np.random.default_rng(1);X=rng.normal(size=(20,4));y=np.array([0,1]*10);a=M.fit_project(X,y,X[:3],X[:2]); assert all(np.isfinite(z).all() for z in a[:3]); assert a[0].shape==(20,5)

def test_public_format():
    ids=np.repeat([f'P_test{i}' for i in range(1,11)],8); trials=np.tile(np.arange(1,9),10); x=M.public(ids,trials,np.linspace(0,1,80)); assert list(x.columns)==['user_id','trial_id','prob. of positive emotion']
