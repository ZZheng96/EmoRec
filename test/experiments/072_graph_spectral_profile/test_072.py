import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exp072',HERE/'run_experiment.py'); e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)

def test_graph_laplacian_properties():
    c=json.loads((HERE/'config.json').read_text())
    x=np.random.default_rng(7).normal(size=(2,30,2500))
    f=e.graph_profile(x,c)
    assert f.shape==(2,24) and np.isfinite(f).all()
    assert np.all((f[:,4::6]>=0)&(f[:,4::6]<=1.01))
    assert np.all((f[:,5::6]>=0)&(f[:,5::6]<=1.01))

def test_graph_is_label_free_and_row_local():
    c=json.loads((HERE/'config.json').read_text())
    rng=np.random.default_rng(8); x=rng.normal(size=(2,30,2500)); a=e.graph_profile(x,c); b=e.graph_profile(x[::-1],c)
    np.testing.assert_allclose(a,b[::-1],rtol=1e-10,atol=1e-10)

def test_config_frozen():
    c=json.loads((HERE/'config.json').read_text())
    assert c['graph_feature_count']==24 and c['bands']==[[4,8],[8,13],[13,30],[30,45]]
    assert c['repeat_seeds']==[2042,2043,2044,2045,2046]
    assert c['expected_model_fits']==60

def test_no_public_predictions():
    assert not list((HERE/'predictions').glob('*public*'))

@pytest.fixture(scope='module')
def artifacts():
    p=HERE/'metrics.json'
    if not p.exists(): pytest.skip('Pre-run: output artifacts not yet generated')
    return json.loads(p.read_text())

def test_oof_artifacts(artifacts):
    assert len(artifacts['fit_seconds'])==30
    assert artifacts['provenance']['model_fits']==60
    for name in ['fixed002']+[f'repeat{s}' for s in [2042,2043,2044,2045,2046]]:
        frame=pd.read_csv(HERE/'predictions'/f'{name}_oof.csv')
        assert len(frame)==2760 and frame.row_index.is_unique and np.isfinite(frame.p_plus).all()

def test_hashes(artifacts):
    for path,digest in artifacts['provenance']['sources'].items(): assert e.base.sha256(Path(path))==digest
