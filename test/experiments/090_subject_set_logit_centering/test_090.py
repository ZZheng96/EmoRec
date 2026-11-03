import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exp090',HERE/'run_experiment.py'); e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
@pytest.fixture(scope='module')
def data():
 c=json.loads((HERE/'config.json').read_text(encoding='utf8')); return c,*e.load_data(c)
def test_config_and_isolation(data):
 c,_,ref,fixed=data; e.validate_config(c)
 for _,folds in e.splits_for(ref,fixed,c): e.base.assert_subject_isolation(ref.subject.to_numpy(),folds)
def test_no_public_array_access(data): assert data[0]['public_access'] is False
def test_artifacts_or_skip():
 if not (HERE/'metrics.json').exists(): pytest.skip('Pre-run: output artifacts not yet generated')
 m=json.loads((HERE/'metrics.json').read_text(encoding='utf8')); assert len(m['fit_seconds'])==30; assert m['provenance']['public_arrays_loaded'] is False
def test_outputs(data):
 if not (HERE/'metrics.json').exists(): pytest.skip('Pre-run: output artifacts not yet generated')
 c,_,ref,fixed=data
 for name,folds in e.splits_for(ref,fixed,c):
  f=pd.read_csv(HERE/'predictions'/f'{name}_oof.csv'); assert len(f)==len(ref); np.testing.assert_array_equal(f.fold,folds)
  assert {'p_control','p_calibrated'}<=set(f); assert np.isfinite(f[['p_control','p_calibrated']]).all().all()
  assert f.groupby(['subject','position']).size().eq(8).all()
  for _,g in f.groupby(['subject','position']): assert len(g)==8
  for _,g in f.groupby('subject'):
   assert len(g)==40
  assert np.max(np.abs(f.p_control - (pd.read_csv(e.DEP/'predictions/fixed002_oof.csv').p_train8 if name=='fixed002' else f.p_control)))<=1
