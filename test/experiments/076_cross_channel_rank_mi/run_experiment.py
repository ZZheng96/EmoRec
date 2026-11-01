import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
try:
    import torch
except ImportError:
    torch = None
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from emo_rec.data import load_train_segments

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / '070_train_context_match_new_splits'
spec = importlib.util.spec_from_file_location('exp070_for_076', HERE.parent / '070_train_context_match_new_splits' / 'run_experiment.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
base = e.base
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
CHANNELS = 'FP1 FP2 F7 F3 FZ F4 F8 FT7 FC3 FCZ FC4 FT8 T3 C3 CZ C4 T4 TP7 CP3 CPZ CP4 TP8 T5 P3 PZ P4 T6 O1 OZ O2'.split()
PAIRS = [(CHANNELS.index(a), CHANNELS.index(b)) for a, b in CONFIG['channel_pairs']]

def rank_mi(segments, device='cuda'):
    x = torch.as_tensor(segments, dtype=torch.float64, device=device)
    if x.ndim != 3 or tuple(x.shape[1:]) != (30, 2500):
        raise ValueError('Expected independent (n,30,2500) windows')
    result = []
    for i, j in PAIRS:
        a, b = x[:, i], x[:, j]
        qa = torch.quantile(a, torch.arange(1, CONFIG['n_bins'], device=x.device, dtype=x.dtype) / CONFIG['n_bins'], dim=1)
        qb = torch.quantile(b, torch.arange(1, CONFIG['n_bins'], device=x.device, dtype=x.dtype) / CONFIG['n_bins'], dim=1)
        ca = torch.sum(a.unsqueeze(-1) > qa.transpose(0, 1).unsqueeze(1), dim=-1)
        cb = torch.sum(b.unsqueeze(-1) > qb.transpose(0, 1).unsqueeze(1), dim=-1)
        joint = torch.zeros((len(x), 16), dtype=x.dtype, device=x.device)
        idx = ca * 4 + cb
        joint.scatter_add_(1, idx, torch.ones_like(idx, dtype=x.dtype))
        joint = joint.reshape(len(x), 4, 4)
        pxy = joint / joint.sum(dim=(-1, -2), keepdim=True)
        px, py = pxy.sum(-1, keepdim=True), pxy.sum(-2, keepdim=True)
        ratio = torch.where(pxy > 0, pxy / (px * py).clamp_min(torch.finfo(x.dtype).tiny), torch.ones_like(pxy))
        result.append(torch.where(pxy > 0, pxy * torch.log(ratio), torch.zeros_like(pxy)).sum(dim=(-1, -2)))
    out = torch.stack(result, dim=1).cpu().numpy()
    assert np.isfinite(out).all()
    return out

def assert_alignment(ref, other):
    for key in ('subject', 'group', 'trial', 'y', 'position'):
        np.testing.assert_array_equal(ref[key], other[key])

def normalize_extra(raw, subjects, positions):
    out = np.empty_like(raw)
    for pos in range(5):
        mask = positions == pos
        for subject in np.unique(subjects[mask]):
            ix = mask & (subjects == subject)
            z = raw[ix]
            out[ix] = (z - z.mean(0)) / np.where(z.std(0, ddof=0) > 1e-12, z.std(0, ddof=0), 1.0)
    return out

def build_cache(config):
    assert torch.cuda.is_available(), 'Frozen production extraction requires CUDA'
    feats, records, hashes = [], [], {}
    for path in sorted((ROOT / '数据/训练集').glob('*/*.mat')):
        if path.stem == config['excluded_subject']:
            continue
        hashes[str(path)] = base.sha256(path)
        windows, labels, trials = load_train_segments(path)
        feats.append(rank_mi(windows, 'cuda'))
        records.extend((path.stem, 'DEP' if path.parent.name == '抑郁症患者' else 'HC', trials[i], labels[i], i % 5) for i in range(len(windows)))
    ref = pd.DataFrame(records, columns=['subject', 'group', 'trial', 'y', 'position'])
    extra = np.concatenate(feats)
    np.savez_compressed(HERE / 'cache/rank_mi.npz', X_train=extra,
        **{k: ref[k].to_numpy(dtype=str if k in ('subject','group') else np.int64) for k in ref},
        feature_names=np.asarray(config['feature_names']), config_sha256=np.asarray(base.sha256(HERE / 'config.json')),
        raw_input_hashes=np.asarray(json.dumps(hashes)))
    return extra, ref, hashes

def fit_pair(control, plus, y, name, folds, fold, config):
    start = time.perf_counter(); train, valid = folds != fold, folds == fold; pred=[]
    for x in (control, plus):
        model = base.make_model(config, config['outer_model_seed_base'] + fold)
        model.fit(x[train], y[train]); pred.append(model.predict_proba(x[valid])[:, 1])
    return name, fold, *pred, time.perf_counter() - start

def summarize(frame, column):
    rr = frame.rename(columns={'subject':'subject_id','group':'health_group'})
    return base.summarize(frame.y.to_numpy(), frame[column].to_numpy(), rr, frame.fold.to_numpy())

def aggregate(deltas, config):
    out={}
    for subset in ['all_rows'] + [str(i) for i in config['positions']]:
        q=[r for r in deltas if r['split'].startswith('repeat') and r['position']==subset]
        out[subset]={k:{'mean':float(np.mean([r[k] for r in q])),'std':float(np.std([r[k] for r in q],ddof=1)),'values':[r[k] for r in q]} for k in e.KEYS}
    return out

def acceptance(agg, config):
    d,a=agg['all_rows'],config['acceptance']; joint=sum(x>=0 and y>=0 for x,y in zip(d['auc']['values'],d['f1']['values']))
    checks={'mean_delta_auc':d['auc']['mean']>=a['mean_delta_auc_min'],'mean_delta_f1':d['f1']['mean']>=a['mean_delta_f1_min'],'joint_nonnegative':joint>=a['nonnegative_splits_min_each'],'hc_no_excess_drop':d['hc_auc']['mean']>=-a['health_group_mean_auc_drop_max'],'dep_no_excess_drop':d['dep_auc']['mean']>=-a['health_group_mean_auc_drop_max']}
    return {'accepted':all(checks.values()),'checks':checks,'joint_nonnegative_count':joint}

def main():
    start=time.perf_counter(); config=CONFIG
    raw, ref, fixed=e.load_data(config); extra, extra_ref, hashes=build_cache(config); assert_alignment(ref, extra_ref)
    _, control=e.contexts(raw, ref.subject.to_numpy(), ref.position.to_numpy())
    plus=np.column_stack([control, normalize_extra(extra, ref.subject.to_numpy(), ref.position.to_numpy())])
    splits=e.splits_for(ref, fixed, config)
    for name, folds in splits:
        base.assert_subject_isolation(ref.subject.to_numpy(), folds); ref.assign(fold=folds).to_csv(HERE/'input_metadata'/f'folds_{name}.csv', index=False)
    sources=[HERE/n for n in ('run_experiment.py','test_076.py','config.json','test_protocol.txt')]
    sources += [Path(e.__file__),Path(base.__file__),ROOT/'src/emo_rec/data.py',ROOT/'src/emo_rec/normalization.py',ROOT/'数据/数据集说明文档.md']
    prov={'raw_inputs':hashes,'inputs':{str((HERE/config[k]).resolve()):base.sha256((HERE/config[k]).resolve()) for k in ('spectral_cache','temporal_cache','pe_cache','fixed_reference')},'sources':{str(p):base.sha256(p) for p in sources},'public_arrays_loaded':False,'control_shape':list(control.shape),'treatment_shape':list(plus.shape),'feature_device':'cuda','feature_dtype':'float64','python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__ if torch is not None else 'unavailable','sklearn':sklearn.__version__,'cpu_exception':'Frozen sklearn HGB','started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8')
    with threadpool_limits(1): results=Parallel(n_jobs=config['n_jobs'],prefer='threads')(delayed(fit_pair)(control,plus,ref.y.to_numpy(),name,folds,fold,config) for name,folds in splits for fold in range(5))
    runs={}; rows=[]; deltas=[]; checks={}
    for name,folds in splits:
        frame=ref.assign(fold=folds,p_control=np.nan,p_plus=np.nan)
        for n,fold,pc,pp,_ in results:
            if n==name: frame.loc[folds==fold,'p_control'],frame.loc[folds==fold,'p_plus']=pc,pp
        old=pd.read_csv(OLD/'predictions'/f'{name}_oof.csv'); checks[name]=float(np.max(np.abs(frame.p_control-old.p_train8))); assert checks[name]<=1e-12
        frame.to_csv(HERE/'predictions'/f'{name}_oof.csv',index=False); runs[name]={}
        for subset in ['all_rows']+[str(i) for i in config['positions']]:
            q=frame if subset=='all_rows' else frame[frame.position==int(subset)]; runs[name][subset]={arm:summarize(q,col) for arm,col in (('control','p_control'),('plus','p_plus'))}; sc={a:e.scalars(r) for a,r in runs[name][subset].items()}; rows.extend({'split':name,'position':subset,'arm':a,**v} for a,v in sc.items()); deltas.append({'split':name,'position':subset,**{k:sc['plus'][k]-sc['control'][k] for k in e.KEYS}})
    agg=aggregate(deltas,config); decision=acceptance(agg,config); prov.update(model_fits=60,elapsed_seconds=time.perf_counter()-start,cache_sha256=base.sha256(HERE/'cache/rank_mi.npz'))
    metrics={'runs':runs,'aggregate_delta':agg,'acceptance':decision,'control_070_max_abs_diff':checks,'provenance':prov,'fit_seconds':[{'split':r[0],'fold':r[1],'pair_seconds':r[4]} for r in results]}
    (HERE/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8'); (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8'); pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False); pd.DataFrame(deltas).to_csv(HERE/'paired_delta.csv',index=False)
    d=agg['all_rows']; text=f'实验076：接受={decision["accepted"]}；联合非负{decision["joint_nonnegative_count"]}/5。\n'+ '\n'.join(f'平均Delta {k}={d[k]["mean"]:.9f}' for k in ('auc','f1','hc_auc','dep_auc'))+f'\n控制720维、处理723维；CUDA float64；60 fits；控制误差{max(checks.values()):.3g}。'
    (HERE/'summary.txt').write_text(text,encoding='utf8'); print(text,flush=True)

if __name__=='__main__': main()
