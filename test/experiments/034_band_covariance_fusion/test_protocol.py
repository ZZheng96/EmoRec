import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('experiment034',HERE/'run_experiment.py'); experiment=importlib.util.module_from_spec(spec); spec.loader.exec_module(experiment)

def config(): return json.loads((HERE/'config.json').read_text(encoding='utf-8'))
def test_covariance_feature_shape_finite_and_deterministic():
    x=np.random.default_rng(2026).normal(size=(2,30,2500)); c=experiment.covariance_features(x,config()); assert c.shape==(2,5,30,30); assert np.isfinite(c).all(); np.testing.assert_array_equal(c,experiment.covariance_features(x,config()))
def test_invalid_input_rejected():
    with pytest.raises(ValueError): experiment.covariance_features(np.zeros((1,29,2500)),config())
    with pytest.raises(ValueError): experiment.covariance_features(np.full((1,30,2500),np.nan),config())
def test_tangent_dimension_finite():
    c=experiment.covariance_features(np.random.default_rng(1).normal(size=(3,30,2500)),config()); centers=np.mean(c,axis=0); f=experiment.tangent_features(c,centers); assert f.shape==(3,2325); assert np.isfinite(f).all()
def test_cache_alignment_and_subject_folds():
    cfg=config(); spectral=experiment.load_cache(HERE/cfg['spectral_cache']); cov=experiment.load_cache(experiment.CACHE); experiment.assert_alignment(spectral,cov); keep,frame=experiment.load_reference(spectral,cfg); assert keep.sum()==2760; assert frame.subject_id.nunique()==69; assert frame.groupby('subject_id').fold.nunique().max()==1
@pytest.mark.parametrize('name',['spectral_control','band_covariance','fixed_05_05'])
def test_outputs(name):
    oof=pd.read_csv(HERE/'predictions'/f'{name}_oof.csv'); public=pd.read_csv(HERE/'predictions'/f'{name}_public.csv'); assert len(oof)==2760 and len(public)==80; experiment.assert_subject_isolation(oof); assert np.isfinite(oof.prob_positive).all() and ((oof.prob_positive>=0)&(oof.prob_positive<=1)).all(); p=public['prob. of positive emotion']; assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
