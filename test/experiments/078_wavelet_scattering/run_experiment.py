import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from emo_rec.data import load_train_segments

spec = importlib.util.spec_from_file_location('exp077_for078', HERE.parent / '077_alpha_traveling_wave_balance/run_experiment.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
e, base, OLD = helper.e, helper.base, helper.OLD
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
normalize_extra, assert_alignment = helper.normalize_extra, helper.assert_alignment
PATHS = [(f,) for f in CONFIG['first_frequencies']] + [tuple(p) for p in CONFIG['second_pairs']]
FEATURE_NAMES = [f'ch{ch+1:02d}_S{len(p)}_' + '_'.join(map(str,p)) + '_' + stat for ch in range(30) for p in PATHS for stat in ('mean','std')]


def wavelet(frequency, device='cpu'):
    sigma = 1 / (2 * math.pi * CONFIG['bandwidth_ratio'] * frequency)
    radius = math.ceil(CONFIG['support_sigma'] * sigma * 250)
    t = torch.arange(-radius, radius+1, dtype=torch.float64, device=device) / 250
    envelope = torch.exp(-t.square() / (2*sigma*sigma))
    carrier = torch.exp(2j*math.pi*frequency*t)
    kernel = envelope * (carrier - (envelope*carrier).sum()/envelope.sum())
    return kernel / kernel.abs().sum()


def convolve_reflect(x, kernel):
    radius = (len(kernel)-1)//2
    padded = F.pad(x.reshape(-1,1,x.shape[-1]), (radius,radius), mode='reflect').reshape(*x.shape[:-1],-1)
    size = 1 << (padded.shape[-1]+len(kernel)-2).bit_length()
    result = torch.fft.ifft(torch.fft.fft(padded,n=size) * torch.fft.fft(kernel,n=size))
    return result[...,len(kernel)-1:len(kernel)-1+x.shape[-1]]


def scattering(segments, device='cuda'):
    x = torch.as_tensor(segments, dtype=torch.float64, device=device)
    if x.ndim != 3 or tuple(x.shape[1:]) != (30,2500):
        raise ValueError('Expected independent (n,30,2500) windows')
    x = x-x.mean(-1,keepdim=True)
    lowpass = torch.ones(125, dtype=x.dtype, device=x.device)/125
    first = {f:convolve_reflect(x,wavelet(f,device)).abs() for f in CONFIG['first_frequencies']}
    stats = []
    for path in PATHS:
        response = first[path[0]] if len(path)==1 else convolve_reflect(first[path[0]],wavelet(path[1],device)).abs()
        smooth = convolve_reflect(response,lowpass).real
        stats.append(torch.stack((smooth.mean(-1),smooth.std(-1,correction=0)),dim=-1))
    out = torch.stack(stats,dim=2).reshape(len(x),-1).cpu().numpy()
    assert out.shape[1]==720 and np.isfinite(out).all()
    return out


def build_cache():
    if not torch.cuda.is_available():
        raise RuntimeError('Production extraction REQUIRES CUDA; CPU is test-only')
    features, records, hashes = [], [], {}
    for path in sorted((ROOT/'数据/训练集').glob('*/*.mat')):
        if path.stem == 'HC43': continue
        hashes[str(path)] = base.sha256(path)
        windows, labels, trials = load_train_segments(path)
        features.append(np.concatenate([scattering(windows[i:i+4]) for i in range(0,len(windows),4)]))
        records.extend((path.stem,'DEP' if path.parent.name=='抑郁症患者' else 'HC',trials[i],labels[i],i%5) for i in range(len(windows)))
        print(f'CUDA scattering {path.stem}',flush=True)
    ref = pd.DataFrame(records,columns=['subject','group','trial','y','position'])
    extra = np.concatenate(features)
    np.savez_compressed(HERE/'cache/scattering.npz',X_train=extra,feature_names=np.asarray(FEATURE_NAMES),config_sha256=np.asarray(base.sha256(HERE/'config.json')),raw_input_hashes=np.asarray(json.dumps(hashes)),**{k:ref[k].to_numpy(dtype=str if k in ('subject','group') else np.int64) for k in ref})
    return extra, ref, hashes


def main():
    start = time.perf_counter()
    for folder in ('cache','input_metadata','predictions'): (HERE/folder).mkdir(exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError('Production requires CUDA; no CPU production fallback')
    config = CONFIG
    raw,ref,fixed = e.load_data(config)
    extra,extra_ref,hashes = build_cache()
    assert_alignment(ref,extra_ref)
    _,control = e.contexts(raw,ref.subject.to_numpy(),ref.position.to_numpy())
    plus = np.column_stack([control,normalize_extra(extra,ref.subject.to_numpy(),ref.position.to_numpy())])
    splits = e.splits_for(ref,fixed,config)
    for name,folds in splits:
        base.assert_subject_isolation(ref.subject.to_numpy(),folds)
        ref.assign(fold=folds).to_csv(HERE/'input_metadata'/f'folds_{name}.csv',index=False)
    sources = [HERE/n for n in ('run_experiment.py','test_078.py','config.json','test_protocol.txt')]
    sources += [Path(helper.__file__),Path(e.__file__),Path(base.__file__),ROOT/'src/emo_rec/data.py',ROOT/'src/emo_rec/normalization.py',ROOT/'pyproject.toml',ROOT/'数据/数据集说明文档.md']
    sources += [HERE.parent/folder/'run_experiment.py' for folder in ('067_adaptation_context_size','068_train_context_match','069_train_context_match_replication')]
    inputs = [(HERE/config[k]).resolve() for k in ('spectral_cache','temporal_cache','pe_cache','fixed_reference')]
    inputs += [OLD/'predictions'/f'{name}_oof.csv' for name,_ in splits]
    prov = {'raw_inputs':hashes,'inputs':{str(p):base.sha256(p) for p in inputs},'sources':{str(p):base.sha256(p) for p in sources},'public_arrays_loaded':False,'control_shape':list(control.shape),'treatment_shape':list(plus.shape),'feature_device':'cuda','gpu':torch.cuda.get_device_name(),'feature_dtype':'float64','python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'sklearn':helper.sklearn.__version__,'cpu_exception':'Frozen sklearn HGB only','started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8')
    print('Starting frozen 60 HGB fits',flush=True)
    with threadpool_limits(1):
        results = Parallel(n_jobs=config['n_jobs'],prefer='threads')(delayed(helper.fit_pair)(control,plus,ref.y.to_numpy(),name,folds,fold,config) for name,folds in splits for fold in range(5))
    assert len(results)==30
    runs,rows,deltas,checks = {},[],[],{}
    for name,folds in splits:
        frame = ref.assign(fold=folds,p_control=np.nan,p_plus=np.nan)
        for n,fold,pc,pp,_ in results:
            if n==name: frame.loc[folds==fold,'p_control'],frame.loc[folds==fold,'p_plus']=pc,pp
        old = pd.read_csv(OLD/'predictions'/f'{name}_oof.csv')
        for key in ('row_index','subject','group','trial','y','position','fold'): np.testing.assert_array_equal(frame[key],old[key])
        checks[name] = float(np.max(np.abs(frame.p_control-old.p_train8)))
        assert checks[name]<=1e-12
        assert np.isfinite(frame[['p_control','p_plus']]).all().all()
        frame.to_csv(HERE/'predictions'/f'{name}_oof.csv',index=False)
        runs[name] = {}
        for subset in ['all_rows']+[str(i) for i in config['positions']]:
            q = frame if subset=='all_rows' else frame[frame.position==int(subset)]
            runs[name][subset] = {a:helper.summarize(q,c) for a,c in (('control','p_control'),('plus','p_plus'))}
            sc = {a:e.scalars(r) for a,r in runs[name][subset].items()}
            rows.extend({'split':name,'position':subset,'arm':a,**v} for a,v in sc.items())
            deltas.append({'split':name,'position':subset,**{k:sc['plus'][k]-sc['control'][k] for k in e.KEYS}})
    agg = helper.aggregate(deltas,config)
    decision = helper.acceptance(agg,config)
    prov.update(model_fits=60,elapsed_seconds=time.perf_counter()-start,cache_sha256=base.sha256(HERE/'cache/scattering.npz'))
    metrics = {'runs':runs,'aggregate_delta':agg,'acceptance':decision,'control_070_max_abs_diff':checks,'provenance':prov,'fit_seconds':[{'split':r[0],'fold':r[1],'pair_seconds':r[4]} for r in results]}
    (HERE/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False)
    pd.DataFrame(deltas).to_csv(HERE/'paired_delta.csv',index=False)
    d = agg['all_rows']
    text = f'实验078：接受={decision["accepted"]}；联合非负{decision["joint_nonnegative_count"]}/5。\n'+'\n'.join(f'平均Delta {k}={d[k]["mean"]:.9f}' for k in ('auc','f1','hc_auc','dep_auc'))+f'\n控制720维、处理1440维；CUDA float64；60 fits；控制误差{max(checks.values()):.3g}。\n失败即停止，不调参不挑位置；不推断公开集成绩。'
    (HERE/'summary.txt').write_text(text,encoding='utf8')
    print(text,flush=True)


if __name__=='__main__': main()
