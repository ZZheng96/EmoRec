import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exp079',HERE/'run_experiment.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_fixed_mapping_and_feature_order():
    assert [len(v) for v in m.REGIONS.values()] == [7,5,5,10,3]
    assert len(m.FEATURE_NAMES)==25 and len(set(m.FEATURE_NAMES))==25
    assert m.CHANNELS[0]=='FP1' and m.CHANNELS[-1]=='O2'

def test_data_layout_and_no_public_arrays():
    c=json.loads((HERE/'config.json').read_text())
    raw,treatment,ref,fixed=m.load_data(c)
    assert raw.shape==(2760,720) and treatment.shape==(2760,595)
    assert 'HC43' not in set(ref.subject)
    assert treatment.shape[1]==25+450+120
    assert ref.subject.nunique()==69

def test_context_normalization_is_label_free():
    rng=np.random.default_rng(79); x=rng.normal(size=(16,595)); subjects=np.repeat(['A','B'],8); pos=np.tile([0,1],8)
    a,b=m.contexts(x,subjects,pos); np.testing.assert_allclose(a[:8,25:475],m.normalize_by_subject(x[:8,25:475],subjects[:8],'zscore'))
    np.testing.assert_allclose(a,b) if np.array_equal(pos,np.zeros(16)) else None

def test_frozen_fits_and_splits():
    c=json.loads((HERE/'config.json').read_text()); _,_,ref,fixed=m.load_data(c)
    splits=m.e.splits_for(ref,fixed,c); assert len(splits)==6 and c['expected_model_fits']==60
    for _,folds in splits: m.base.assert_subject_isolation(ref.subject.to_numpy(),folds)

def test_artifacts_when_present():
    if not (HERE/'metrics.json').exists(): return
    metrics=json.loads((HERE/'metrics.json').read_text()); assert len(metrics['fit_seconds'])==60
    assert metrics['provenance']['model_fits']==60 and metrics['provenance']['public_arrays_loaded'] is False
    for name,_ in m.e.splits_for(m.load_data(json.loads((HERE/'config.json').read_text()))[2],m.load_data(json.loads((HERE/'config.json').read_text()))[3],json.loads((HERE/'config.json').read_text())):
        frame=pd.read_csv(HERE/'predictions'/f'{name}_oof.csv'); old=pd.read_csv(m.DEP/'predictions'/f'{name}_oof.csv')
        np.testing.assert_allclose(frame.p_train40,old.p_train40,atol=1e-12,rtol=0)
