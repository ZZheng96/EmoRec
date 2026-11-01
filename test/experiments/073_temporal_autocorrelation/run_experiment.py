import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits, threadpool_info

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / '070_train_context_match_new_splits'
spec = importlib.util.spec_from_file_location('exp070', OLD / 'run_experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
base = e.base
from emo_rec.data import load_train_segments
from emo_rec.normalization import normalize_by_subject


def autocorrelation(segments, device='cuda'):
    x = torch.as_tensor(segments, dtype=torch.float64, device=device)
    if x.ndim != 3 or tuple(x.shape[1:]) != (30, 2500):
        raise ValueError('Expected independent (n,30,2500) windows')
    z = x - x.mean(dim=-1, keepdim=True)
    features = []
    for lag in (250, 500):
        a, b = z[..., :-lag], z[..., lag:]
        den = torch.sqrt((a*a).sum(-1) * (b*b).sum(-1))
        features.append(torch.where(den > 0, (a*b).sum(-1) / den, torch.zeros_like(den)))
    result = torch.cat(features, dim=1).cpu().numpy()
    assert np.isfinite(result).all()
    return result


def build_cache(config):
    feats, records, hashes = [], [], {}
    for path in sorted((ROOT / '数据/训练集').glob('*/*.mat')):
        hashes[str(path)] = base.sha256(path)
        segments, labels, trials = load_train_segments(path)
        feats.append(autocorrelation(segments, config['feature_device']))
        for i in range(len(segments)):
            records.append((path.stem, 'DEP' if path.parent.name == '抑郁症患者' else 'HC', trials[i], labels[i], i % 5))
    ref = pd.DataFrame(records, columns=['subject', 'group', 'trial', 'y', 'position'])
    keep = ref.subject != config['excluded_subject']
    ref = ref.loc[keep].reset_index(drop=True)
    features = np.concatenate(feats)[keep]
    np.savez_compressed(HERE / 'cache/autocorrelation.npz', X_train=features,
                        **{k: ref[k].to_numpy(dtype=str if k in ('subject','group') else np.int64) for k in ref},
                        feature_names=np.asarray([f'autocorr_lag{lag}_ch{ch:02d}' for lag in (250,500) for ch in range(1,31)]),
                        config_sha256=np.asarray(base.sha256(HERE/'config.json')),
                        raw_input_hashes=np.asarray(json.dumps(hashes)))
    return features, ref, hashes


def assert_alignment(ref, other):
    for key in ('subject', 'group', 'trial', 'y', 'position'):
        np.testing.assert_array_equal(ref[key], other[key])


def normalize_extra(raw, subjects, positions):
    result = np.empty_like(raw)
    for pos in range(5):
        mask = positions == pos
        result[mask] = normalize_by_subject(raw[mask], subjects[mask], 'zscore')
    return result


def fit_pair(control, plus, y, name, folds, fold, config):
    start = time.perf_counter()
    train, valid = folds != fold, folds == fold
    predictions = []
    for x in (control, plus):
        model = base.make_model(config, config['outer_model_seed_base'] + fold)
        model.fit(x[train], y[train])
        predictions.append(model.predict_proba(x[valid])[:, 1])
    return name, fold, *predictions, time.perf_counter()-start


def summarize(frame, column):
    rr = frame.rename(columns={'subject':'subject_id', 'group':'health_group'})
    return base.summarize(frame.y.to_numpy(), frame[column].to_numpy(), rr, frame.fold.to_numpy())


def aggregate(deltas, config):
    out = {}
    for subset in ['all_rows'] + [str(i) for i in config['positions']]:
        selected = [r for r in deltas if r['split'].startswith('repeat') and r['position'] == subset]
        assert len(selected) == 5
        out[subset] = {k: {'mean': float(np.mean([r[k] for r in selected])),
                          'std': float(np.std([r[k] for r in selected], ddof=1)),
                          'values': [r[k] for r in selected]} for k in e.KEYS}
    return out


def acceptance(agg, config):
    d, a = agg['all_rows'], config['acceptance']
    joint = sum(x >= 0 and y >= 0 for x,y in zip(d['auc']['values'], d['f1']['values']))
    checks = {'mean_delta_auc': d['auc']['mean'] >= a['mean_delta_auc_min'],
              'mean_delta_f1': d['f1']['mean'] >= a['mean_delta_f1_min'],
              'joint_nonnegative': joint >= a['nonnegative_splits_min_joint'],
              'hc_no_excess_drop': d['hc_auc']['mean'] >= -a['health_group_mean_auc_drop_max'],
              'dep_no_excess_drop': d['dep_auc']['mean'] >= -a['health_group_mean_auc_drop_max']}
    return {'accepted': all(checks.values()), 'checks': checks, 'joint_nonnegative_count': joint}


def main():
    start = time.perf_counter()
    config = json.loads((HERE/'config.json').read_text())
    old_config = json.loads((OLD/'config.json').read_text())
    for key in ('hgb','outer_model_seed_base','threshold','repeat_seeds','positions','excluded_subject','n_jobs','threads_per_job'):
        assert config[key] == old_config[key]
    for folder in ('cache','predictions','input_metadata'):
        (HERE/folder).mkdir(exist_ok=True)
    raw, ref, fixed = e.load_data(config)
    extra, extra_ref, hashes = build_cache(config)
    assert_alignment(ref, extra_ref)
    _, control = e.contexts(raw, ref.subject.to_numpy(), ref.position.to_numpy())
    plus = np.column_stack([control, normalize_extra(extra, ref.subject.to_numpy(), ref.position.to_numpy())])
    assert control.shape == (2760,720) and plus.shape == (2760,780)
    splits = e.splits_for(ref, fixed, config)
    sources = [HERE/n for n in ('run_experiment.py','test_073.py','config.json','test_protocol.txt')]
    sources += [Path(e.__file__),Path(e.e.__file__),Path(e.e.e.__file__),Path(e.load_data.__code__.co_filename),Path(base.__file__),ROOT/'src/emo_rec/data.py',ROOT/'src/emo_rec/normalization.py',ROOT/'pyproject.toml',ROOT/'数据/数据集说明文档.md']
    inputs = [(HERE/config[k]).resolve() for k in ('spectral_cache','temporal_cache','pe_cache','fixed_reference')]
    inputs += [OLD/'predictions'/f'{name}_oof.csv' for name,_ in splits]
    prov = {'raw_inputs':hashes,'inputs':{str(p):base.sha256(p) for p in inputs},'sources':{str(p):base.sha256(p) for p in sources},
            'public_arrays_loaded':False,'control_shape':list(control.shape),'treatment_shape':list(plus.shape),
            'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,
            'feature_device':torch.cuda.get_device_name(0),'feature_dtype':'float64',
            'cpu_exception':'Frozen sklearn HGB has no GPU implementation; 10 jobs, one thread each',
            'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'feature_seconds':time.perf_counter()-start}
    for name,folds in splits:
        base.assert_subject_isolation(ref.subject.to_numpy(),folds)
        ref.assign(fold=folds).to_csv(HERE/'input_metadata'/f'folds_{name}.csv',index=False)
    (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8')
    print('073 frozen: 60 fits, 720 versus 780, a8 to a8; no public arrays',flush=True)
    with threadpool_limits(1):
        prov['active_threadpools'] = threadpool_info()
        results = Parallel(n_jobs=config['n_jobs'],prefer='threads')(delayed(fit_pair)(control,plus,ref.y.to_numpy(),name,folds,fold,config) for name,folds in splits for fold in range(5))
    assert len(results)*2 == config['expected_model_fits'] == 60
    runs, rows, deltas, checks = {}, [], [], {}
    for name,folds in splits:
        frame = ref.assign(fold=folds,p_control=np.nan,p_plus=np.nan)
        for n,fold,pc,pp,_ in results:
            if n == name:
                frame.loc[folds==fold,'p_control'] = pc
                frame.loc[folds==fold,'p_plus'] = pp
        assert np.isfinite(frame[['p_control','p_plus']]).all().all()
        old = pd.read_csv(OLD/'predictions'/f'{name}_oof.csv')
        for col in ('row_index','position','subject','group','trial','y','fold'):
            np.testing.assert_array_equal(frame[col],old[col])
        checks[name] = float(np.max(np.abs(frame.p_control-old.p_train8)))
        assert checks[name] <= 1e-12
        frame.to_csv(HERE/'predictions'/f'{name}_oof.csv',index=False)
        runs[name] = {}
        for subset in ['all_rows']+[str(i) for i in config['positions']]:
            q = frame if subset=='all_rows' else frame[frame.position==int(subset)]
            runs[name][subset] = {arm:summarize(q,col) for arm,col in (('control','p_control'),('plus','p_plus'))}
            scalar = {arm:e.scalars(r) for arm,r in runs[name][subset].items()}
            for arm in scalar:
                rows.append({'split':name,'position':subset,'arm':arm,**scalar[arm]})
            deltas.append({'split':name,'position':subset,**{k:scalar['plus'][k]-scalar['control'][k] for k in e.KEYS}})
    agg = aggregate(deltas,config)
    decision = acceptance(agg,config)
    prov.update(model_fits=60,elapsed_seconds=time.perf_counter()-start,cache_sha256=base.sha256(HERE/'cache/autocorrelation.npz'))
    metrics = {'runs':runs,'aggregate_delta':agg,'acceptance':decision,'control_070_max_abs_diff':checks,'provenance':prov,
               'fit_seconds':[{'split':r[0],'fold':r[1],'pair_seconds':r[4]} for r in results]}
    (HERE/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False)
    pd.DataFrame(deltas).to_csv(HERE/'paired_delta.csv',index=False)
    d = agg['all_rows']
    text = f'# 实验073：连续原始波形多滞后自相关\n\n接受：{decision["accepted"]}；联合非负 {decision["joint_nonnegative_count"]}/5。\n\n'
    text += '\n'.join(f'- 五新split平均Δ{k}: {d[k]["mean"]:.9f}' for k in ('auc','f1','hc_auc','dep_auc'))
    text += f'\n\n60 fits；总耗时 {prov["elapsed_seconds"]:.2f} 秒；特征/加载/哈希 {prov["feature_seconds"]:.2f} 秒。\nCUDA float64自相关；冻结HGB无GPU实现，CPU例外10 jobs/1 thread。\n控制720维、处理780维，均a8→a8；六套控制逐行复现070，最大误差{max(checks.values()):.3g}。\n未访问公开数组。未通过则停止，不调参、不挑位置。\n\n命令（项目根目录）：\n```powershell\n.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider test/experiments/073_temporal_autocorrelation/test_073.py -q\n.venv/Scripts/python.exe -B test/experiments/073_temporal_autocorrelation/run_experiment.py\n.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider test/experiments/073_temporal_autocorrelation/test_073.py -q\n```\n前后测试输出见pytest_before.txt和pytest_after.txt。输入SHA256及运行环境见input_metadata/provenance.json。\n'
    (HERE/'summary.md').write_text(text,encoding='utf8')
    print(json.dumps({'acceptance':decision,'all_rows_delta':d,'elapsed_seconds':prov['elapsed_seconds']},indent=2),flush=True)


if __name__ == '__main__':
    main()
