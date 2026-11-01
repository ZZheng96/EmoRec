import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits, threadpool_info

HERE = Path(__file__).resolve().parent
E070 = HERE.parent / '070_train_context_match_new_splits'
spec = importlib.util.spec_from_file_location('exp070_dependency', E070 / 'run_experiment.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
base = e.base
load_data, contexts, summarize, scalars = e.load_data, e.contexts, e.summarize, e.scalars
KEYS = e.KEYS


def validate_config(config):
    old = json.loads((E070 / 'config.json').read_text(encoding='utf8'))
    for k in ('spectral_cache','temporal_cache','pe_cache','fixed_reference','outer_model_seed_base','hgb','threshold','excluded_subject','n_jobs','threads_per_job','positions','expected_model_fits','validation_context','expected_train_rows','public_access'):
        assert config[k] == old[k]
    assert config['repeat_seeds'] == [2042, 2043, 2044, 2045, 2046]


def splits_for(ref, fixed, config):
    return [('fixed002', fixed)] + [(f'repeat{s}', base.make_subject_folds(ref.subject.to_numpy(), ref.group.to_numpy(), s)) for s in config['repeat_seeds']]


def exact_group_weights(groups):
    values, counts = np.unique(groups, return_counts=True)
    assert set(values) == {'HC', 'DEP'}
    n = len(groups)
    weights = np.asarray([n / (2 * counts[np.where(values == g)[0][0]]) for g in groups], dtype=float)
    assert np.isclose(weights.mean(), 1.0)
    for g in values:
        assert np.isclose(weights[groups == g].sum(), n / 2)
    return weights, {str(g): {'n': int(c), 'weight': float(n / (2*c)), 'total_weight': float((groups == g).sum() * n / (2*c))} for g, c in zip(values, counts)}


def job(a8, y, groups, subjects, name, folds, fold, weighted, config):
    start = time.perf_counter(); train, valid = folds != fold, folds == fold
    model = base.make_model(config, config['outer_model_seed_base'] + fold)
    fit_kwargs = {}
    evidence = {'split': name, 'outer_fold': fold, 'weighted': weighted, 'outer_train_rows': int(train.sum()), 'outer_validation_rows': int(valid.sum()), 'subject_disjoint': not bool(set(subjects[train]) & set(subjects[valid]))}
    if weighted:
        weights, counts = exact_group_weights(groups[train]); fit_kwargs['histgradientboostingclassifier__sample_weight'] = weights; evidence['weights'] = counts
    model.fit(a8[train], y[train], **fit_kwargs)
    p = model.predict_proba(a8[valid])[:, 1]
    return name, fold, weighted, p, time.perf_counter() - start, int(train.sum()), evidence


def aggregate_runs(runs, config):
    out = {}
    for subset in ['all_rows'] + [str(i) for i in config['positions']]:
        out[subset] = {}
        for condition in ('control', 'weighted', 'delta_weighted_minus_control'):
            vals = []
            for seed in config['repeat_seeds']:
                if condition == 'delta_weighted_minus_control':
                    vals.append({k: scalars(runs[f'repeat{seed}'][subset]['weighted'])[k] - scalars(runs[f'repeat{seed}'][subset]['control'])[k] for k in KEYS})
                else: vals.append(scalars(runs[f'repeat{seed}'][subset][condition]))
            out[subset][condition] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
    return out


def acceptance(aggregate, config):
    d = aggregate['all_rows']['delta_weighted_minus_control']; a = config['acceptance']
    checks = {'mean_delta_dep_auc': d['dep_auc']['mean'] >= a['mean_delta_auc_min'], 'mean_delta_overall_auc': d['auc']['mean'] >= a['mean_delta_overall_auc_min'], 'mean_delta_f1': d['f1']['mean'] >= a['mean_delta_f1_min'], 'dep_nonnegative': sum(v >= 0 for v in d['dep_auc']['values']) >= a['dep_nonnegative_splits_min'], 'hc_no_excess_drop': d['hc_auc']['mean'] >= -a['health_group_mean_auc_drop_max']}
    return {'accepted': all(checks.values()), 'checks': checks}


def main():
    started = time.perf_counter(); config = json.loads((HERE/'config.json').read_text(encoding='utf8')); validate_config(config)
    for folder in ('predictions','input_metadata'): (HERE/folder).mkdir(exist_ok=True)
    raw, ref, fixed = load_data(config); subjects, y, groups = ref.subject.to_numpy(), ref.y.to_numpy(), ref.group.to_numpy(); _, a8 = contexts(raw, subjects, ref.position.to_numpy())
    splits = splits_for(ref, fixed, config); inputs = [(HERE/config[k]).resolve() for k in ('spectral_cache','temporal_cache','pe_cache','fixed_reference')]
    sources = [HERE/n for n in ('run_experiment.py','test_071.py','config.json','test_protocol.txt')] + [E070/'run_experiment.py', E070/'test_070.py', E070/'config.json', E070/'test_protocol.txt', Path(base.__file__), HERE.parents[2]/'src/emo_rec/normalization.py', HERE.parents[2]/'src/emo_rec/data.py', HERE.parents[2]/'src/emo_rec/submission.py', HERE.parents[2]/'pyproject.toml', HERE.parents[2]/'数据/数据集说明文档.md']
    prov = {'inputs': {str(p): base.sha256(p) for p in inputs}, 'sources': {str(p): base.sha256(p) for p in sources}, 'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'platform': platform.platform(), 'raw_shape': list(raw.shape), 'public_arrays_loaded': False, 'model_fits': 60, 'cpu_only': 'sklearn HGB has no GPU implementation; controlled CPU exception', 'n_jobs': config['n_jobs'], 'threads_per_job': config['threads_per_job'], 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    for name, folds in splits:
        base.assert_subject_isolation(subjects, folds); ref.assign(fold=folds).to_csv(HERE/'input_metadata'/f'folds_{name}.csv', index=False)
    with threadpool_limits(limits=1):
        prov['active_threadpools'] = threadpool_info(); results = Parallel(n_jobs=10, prefer='threads')(delayed(job)(a8,y,groups,subjects,name,folds,fold,weighted,config) for name,folds in splits for fold in range(5) for weighted in (False,True))
    assert len(results) == 60
    runs, rows, deltas, fits = {}, [], [], []
    for name, folds in splits:
        frame = ref.assign(fold=folds, p_control=np.nan, p_weighted=np.nan); runs[name] = {}
        for n, fold, weighted, p, seconds, n_train, evidence in results:
            if n == name:
                frame.loc[folds == fold, 'p_weighted' if weighted else 'p_control'] = p; fits.append({'split':n,'fold':fold,'weighted':weighted,'seconds':seconds,'n_train':n_train});
                if weighted: (HERE/'input_metadata'/f'{name}_fold{fold}_weights.json').write_text(json.dumps(evidence, indent=2), encoding='utf8')
        assert np.isfinite(frame[['p_control','p_weighted']]).all().all()
        if name == 'fixed002':
            old = pd.read_csv(E070/'predictions/fixed002_oof.csv'); np.testing.assert_array_equal(frame[['row_index','position','subject','group','trial','y','fold']], old[['row_index','position','subject','group','trial','y','fold']]); assert np.max(np.abs(frame.p_control-old.p_train8)) <= 1e-12
        frame.to_csv(HERE/'predictions'/f'{name}_oof.csv', index=False)
        for subset in ['all_rows'] + [str(i) for i in config['positions']]:
            selected = frame if subset == 'all_rows' else frame[frame.position == int(subset)]; runs[name][subset] = {}
            for condition in ('control','weighted'):
                r = summarize(selected.rename(columns={condition if condition=='control' else 'weighted':'x'}), 'x') if False else base.summarize(selected.y.to_numpy(), selected['p_'+condition].to_numpy(), selected.rename(columns={'subject':'subject_id','group':'health_group'}), selected.fold.to_numpy())
                runs[name][subset][condition] = r; rows.append({'split':name,'position':subset,'condition':condition,**scalars(r)})
            deltas.append({'split':name,'position':subset,**{k:scalars(runs[name][subset]['weighted'])[k]-scalars(runs[name][subset]['control'])[k] for k in KEYS}})
    aggregate = aggregate_runs(runs, config); decision = acceptance(aggregate, config); elapsed = time.perf_counter()-started; prov['elapsed_seconds']=elapsed
    metrics={'runs':runs,'aggregate':aggregate,'acceptance':decision,'fixed002_control_max_abs_diff':float(np.max(np.abs(pd.read_csv(HERE/'predictions/fixed002_oof.csv').p_control-pd.read_csv(E070/'predictions/fixed002_oof.csv').p_train8))),'fit_seconds':fits,'elapsed_seconds':elapsed,'provenance':prov}
    (HERE/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8'); (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8'); pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False); pd.DataFrame(deltas).to_csv(HERE/'paired_delta.csv',index=False)
    lines=['# 实验071：严格HC/DEP等权ERM','', '冻结协议见test_protocol.txt。控制与严格等权组训练均为a8→a8；处理权重 w_i=N_train/(2*N_group_train)，组总权重各N_train/2、均值1、不裁剪。', '', '|split|ΔAUC|ΔF1|HC ΔAUC|DEP ΔAUC|','|---|---:|---:|---:|---:|']
    for d in deltas:
        if d['position']=='all_rows': lines.append(f"|{d['split']}|{d['auc']:+.6f}|{d['f1']:+.6f}|{d['hc_auc']:+.6f}|{d['dep_auc']:+.6f}|")
    d=aggregate['all_rows']['delta_weighted_minus_control']; lines += ['',f"五重复均值：ΔAUC={d['auc']['mean']:+.6f}，ΔF1={d['f1']['mean']:+.6f}，HC ΔAUC={d['hc_auc']['mean']:+.6f}，DEP ΔAUC={d['dep_auc']['mean']:+.6f}。", '接受判断：'+('通过。' if decision['accepted'] else '未通过。'), '方法背景（不作为EEG证据）：Sagawa et al. https://arxiv.org/abs/1911.08731；HGB sample_weight文档 https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html。普通平均可能掩盖弱组，naive groupDRO也可能泛化失败；本实验是固定等权ERM，不是DRO。训练44 HC/25 DEP与公开5/5仅支持动机，不能解决组内条件偏移。',f'耗时：{elapsed:.2f}秒；控制fixed002逐行复现070 p_train8最大误差≤1e-12；未访问公开数组。']
    (HERE/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf8'); print(json.dumps({'acceptance':decision,'elapsed_seconds':elapsed,'all_rows':aggregate['all_rows']},indent=2),flush=True)

if __name__ == '__main__': main()
