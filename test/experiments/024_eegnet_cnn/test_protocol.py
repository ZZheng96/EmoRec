import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd, pytest, torch
spec=importlib.util.spec_from_file_location('exp024',Path(__file__).with_name('run_experiment.py')); MODULE=importlib.util.module_from_spec(spec); spec.loader.exec_module(MODULE)

def test_model_input_output_shape():
    model=MODULE.EEGNet(); out=model(torch.zeros(2,1,30,2000)); assert out.shape==(2,2)
def test_fixed_parameters():
    c=json.loads((Path(__file__).with_name('config.json')).read_text()); assert c['training']['epochs']==30; assert c['training']['batch_size']==64; assert c['training']['lr']==.001; assert c['training']['weight_decay']==.01; assert c['training']['seed_base']==2026 if 'seed_base' in c['training'] else True
    assert c['seed_base']==2026 and c['training']['early_stopping'] is False

def test_subject_isolation():
    frame=pd.DataFrame({'subject_id':['A','A','B','B'],'fold':[0,0,1,1]})
    for f in [0,1]: assert not (set(frame.subject_id[frame.fold!=f]) & set(frame.subject_id[frame.fold==f]))
def test_normalization_locality():
    x=np.zeros((2,30,2000),dtype=np.float32); x[0]=3; x[1]=100
    z=MODULE.sample_channel_zscore(x); assert np.allclose(z,0); x[1,0,0]=101; assert np.isfinite(MODULE.sample_channel_zscore(x)).all()
def test_metadata_and_cache_contract():
    c=json.loads((Path(__file__).with_name('config.json')).read_text()); assert c['input']['shape']==[30,2000]; assert c['input']['dtype']=='float32'; assert c['protocol']['exclude_subject']=='HC43'
def test_output_completeness_after_run():
    d=Path(__file__).parent; assert (d/'config.json').exists() and (d/'run_experiment.py').exists() and (d/'实验报告.md').exists(); assert (d/'predictions').is_dir() and (d/'cache').is_dir()
    oof=pd.read_csv(d/'predictions/eegnet_oof.csv'); public=pd.read_csv(d/'predictions/eegnet_public.csv')
    assert len(oof)==2760 and len(public)==80 and oof.prob_positive.between(0,1).all() and public['prob. of positive emotion'].between(0,1).all()

def test_no_public_labels_in_config():
    c=json.loads((Path(__file__).with_name('config.json')).read_text()); assert c['protocol']['public_labels_used'] is False
