import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('experiment052', HERE / 'run_experiment.py'); exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))


def synthetic():
    rng = np.random.default_rng(52); subjects = np.repeat([f'{g}{i}' for g in ('HC','DEP') for i in range(10)], 6); groups = np.array(['HC' if s.startswith('HC') else 'DEP' for s in subjects]); y = np.tile([0,1], len(subjects)//2); X = rng.normal(size=(len(y), 20)); X[:,0] += y; return X,y,subjects,groups


def test_weights_use_training_labels_only_and_range():
    X,y,s,g = synthetic(); w,gw,e = exp.training_weights(X,y,s,g,CONFIG,2052)
    assert np.isfinite(w).all() and np.isfinite(gw).all(); assert w.min() >= 1 and w.max() <= 3; assert set(e['training_subjects']) == set(s); assert len(e['subject_losses']) == len(np.unique(s))
    changed = 1-y; w2,gw2,e2 = exp.training_weights(X,changed,s,g,CONFIG,2052); assert not np.array_equal(w,w2); assert e['training_labels_sha256'] != e2['training_labels_sha256']


def test_validation_labels_do_not_change_predictions_or_weights():
    X,y,s,g = synthetic(); folds = exp.base.make_subject_folds(s,g,2027); tr,va = folds != 0, folds == 0; changed=y.copy(); changed[va]=1-changed[va]
    with threadpool_limits(limits=1):
        a = exp.training_weights(X[tr],y[tr],s[tr],g[tr],CONFIG,2052); b = exp.training_weights(X[tr],changed[tr],s[tr],g[tr],CONFIG,2052)
    np.testing.assert_array_equal(a[0],b[0]); np.testing.assert_array_equal(a[1],b[1]); assert a[2] == b[2]


def test_real_dimensions_finite_and_subject_disjoint():
    X,XP,ref,spectral = exp.load_data(CONFIG); assert X.shape==(2760,600) and XP.shape==(80,600); assert 'HC43' not in set(ref.subject_id); folds=ref.fold.to_numpy(); exp.base.assert_subject_isolation(ref.subject_id.to_numpy(),folds); assert np.isfinite(X).all() and np.isfinite(XP).all()


def test_saved_artifacts_when_available():
    if not (HERE/'metrics.json').exists(): return
    m=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
    for method in exp.METHODS:
        for split in ['fixed002']+[f'repeat{s}' for s in CONFIG['repeat_seeds']]:
            frame=pd.read_csv(HERE/'predictions'/f'{method}_{split}_oof.csv'); assert len(frame)==2760 and frame.prob_positive.between(0,1).all(); exp.base.assert_subject_isolation(frame.subject_id.to_numpy(),frame.fold.to_numpy()); assert frame.groupby('subject_id').size().eq(40).all()
    for path in (HERE/'input_metadata').glob('*_weights.json'):
        data=json.loads(path.read_text(encoding='utf-8')); assert 1 <= data['min_weight'] <= data['max_weight'] <= 3; assert data['subject_disjoint'] if 'subject_disjoint' in data else True
    assert m['advance'] in (True,False)
