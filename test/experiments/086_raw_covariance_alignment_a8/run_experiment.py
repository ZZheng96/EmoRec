import hashlib
import importlib.util
import json
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import sklearn
import torch
from joblib import Parallel, delayed
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits, threadpool_info

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / '070_train_context_match_new_splits'
sys.path.insert(0, str(ROOT / 'src'))
from emo_rec import data
from emo_rec.covariance import regularized_covariances
from emo_rec.features import extract_spectral_features
spec = importlib.util.spec_from_file_location('exp070_for086', OLD / 'run_experiment.py')
e070 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e070)
base, contexts, scalars, KEYS = e070.base, e070.contexts, e070.scalars, e070.KEYS
TRAIN_KEYS = ('X_train', 'subject_ids', 'health_groups', 'original_trial_ids', 'y_train', 'feature_names')
META = ('row_index', 'subject', 'group', 'trial', 'y', 'position', 'fold')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


@contextmanager
def public_guard():
    original_get = np.lib.npyio.NpzFile.__getitem__
    original_load, original_h5 = np.load, h5py.File
    def get(cache, key):
        if key not in TRAIN_KEYS: raise AssertionError(f'Forbidden array: {key}')
        return original_get(cache, key)
    def check(path):
        text = str(path).lower()
        if any(s in text for s in ('public', '测试集', 'test_data')): raise AssertionError('Forbidden public path')
    def load(path, *args, **kwargs):
        check(path); return original_load(path, *args, **kwargs)
    def h5(path, *args, **kwargs):
        check(path)
        assert (ROOT / '数据/训练集').resolve() in Path(path).resolve().parents
        return original_h5(path, *args, **kwargs)
    with patch.object(np.lib.npyio.NpzFile, '__getitem__', get), patch.object(np, 'load', load), patch.object(h5py, 'File', h5), patch.object(data, 'load_public_test_segments', side_effect=AssertionError('Forbidden public loader')):
        yield


def validate_config(c):
    old = json.loads((OLD / 'config.json').read_text(encoding='utf8'))
    for key in ('hgb', 'outer_model_seed_base', 'threshold', 'excluded_subject', 'n_jobs', 'threads_per_job', 'positions', 'repeat_seeds', 'acceptance', 'spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference'):
        assert c[key] == old[key]
    assert c['expected_model_fits'] == 30 and c['shrinkage'] == .1 and c['sampling_rate'] == 250
    assert c['public_access'] is False


def whiten_cpu(x):
    cov = regularized_covariances(x, shrinkage=.1).mean(axis=0)
    cov = (cov + cov.T) / 2
    values, vectors = eigh(cov)
    assert values.min() > 0
    w = (vectors * values ** -.5) @ vectors.T
    return w @ x, cov, w


def whiten_gpu(x):
    t = torch.as_tensor(x, dtype=torch.float64, device='cuda')
    centered = t - t.mean(dim=-1, keepdim=True)
    cov = centered @ centered.transpose(-1, -2) / (t.shape[-1] - 1)
    scale = cov.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(torch.finfo(t.dtype).tiny)
    cov = (.9 * cov + .1 * scale[:, None, None] * torch.eye(t.shape[1], device='cuda', dtype=t.dtype)).mean(0)
    cov = (cov + cov.T) / 2
    values, vectors = torch.linalg.eigh(cov)
    assert values.min().item() > 0
    w = (vectors * values.pow(-.5)) @ vectors.T
    return (w @ t).cpu().numpy(), cov.cpu().numpy(), w.cpu().numpy()


def align_contexts(x, subjects, positions, compare=False):
    out = np.empty_like(x); errors = []
    for subject in np.unique(subjects):
        for position in np.unique(positions):
            ix = np.flatnonzero((subjects == subject) & (positions == position))
            assert len(ix) == 8
            aligned, cov, w = whiten_gpu(x[ix])
            np.testing.assert_allclose(w @ cov @ w, np.eye(x.shape[1]), atol=1e-7)
            if compare:
                expected, cc, ww = whiten_cpu(x[ix])
                for a, b in ((aligned, expected), (cov, cc), (w, ww)):
                    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)
                errors.append(float(np.max(np.abs(aligned - expected))))
            out[ix] = aligned
    return out, max(errors, default=0.)


def check_raw_metadata(y, trials, ref):
    np.testing.assert_array_equal(y, ref.y.to_numpy())
    np.testing.assert_array_equal(trials, ref.trial.to_numpy())
    assert len(ref) == 40


def load_data(c):
    raw, ref, fixed = e070.load_data(c)
    original = raw.copy(); paths = []; errors = []
    for subject in ref.subject.drop_duplicates():
        found = list((ROOT / '数据/训练集').rglob(f'{subject}.mat'))
        assert len(found) == 1
        path = found[0]; paths.append(path)
        x, y, trials = data.load_train_segments(path)
        ix = np.flatnonzero(ref.subject.to_numpy() == subject); r = ref.iloc[ix]
        check_raw_metadata(y, trials, r)
        aligned, error = align_contexts(x, r.subject.to_numpy(), r.position.to_numpy(), compare=True)
        spectral, names = extract_spectral_features(aligned, c['sampling_rate'])
        mask = np.asarray([str(n).startswith('log_bandpower_') for n in names])
        assert mask.sum() == 150
        raw[ix, :150] = spectral[:, mask]; errors.append(error)
    np.testing.assert_array_equal(raw[:, 150:], original[:, 150:])
    a8 = contexts(raw, ref.subject.to_numpy(), ref.position.to_numpy())[1]
    original_a8 = contexts(original, ref.subject.to_numpy(), ref.position.to_numpy())[1]
    np.testing.assert_array_equal(a8[:, 150:], original_a8[:, 150:])
    assert np.isfinite(a8).all() and not np.allclose(raw[:, :150], a8[:, :150])
    return a8, ref, fixed, paths, {'max_gpu_cpu_signal_abs_error': max(errors), 'contexts_compared': len(errors)*5, 'unchanged_temporal_pe_raw_and_a8': True, 'a8_applied': True, 'a8_sha256': hashlib.sha256(a8.tobytes()).hexdigest()}


def validate_control(ref, folds, control):
    expected = ref.assign(fold=folds)
    for col in META: np.testing.assert_array_equal(expected[col], control[col])
    assert np.isfinite(control.p_train8).all()


def fit_one(a8, y, name, folds, fold, c):
    start = time.perf_counter(); train, valid = folds != fold, folds == fold
    model = base.make_model(c, c['outer_model_seed_base'] + fold)
    model.fit(a8[train], y[train])
    return name, fold, model.predict_proba(a8[valid])[:, 1], time.perf_counter()-start, int(train.sum())


def summarize(frame, col):
    return base.summarize(frame.y.to_numpy(), frame[col].to_numpy(), frame.rename(columns={'subject':'subject_id','group':'health_group'}), frame.fold.to_numpy())


def aggregate_runs(runs, c):
    out = {}
    for subset in ['all_rows'] + [str(i) for i in c['positions']]:
        out[subset] = {}
        for arm in ('control8', 'alignment8', 'delta'):
            vals = [runs[f'repeat{s}'][subset][arm] for s in c['repeat_seeds']]
            if arm != 'delta': vals = [scalars(v) for v in vals]
            out[subset][arm] = {k: {'mean':float(np.mean([v[k] for v in vals])), 'std':float(np.std([v[k] for v in vals], ddof=1)), 'values':[v[k] for v in vals]} for k in KEYS}
    return out


def acceptance(aggregate, c):
    d = aggregate['all_rows']['delta']; a = c['acceptance']
    joint = [auc >= 0 and f1 >= 0 for auc, f1 in zip(d['auc']['values'], d['f1']['values'])]
    checks = {'mean_delta_auc':d['auc']['mean'] >= a['mean_delta_auc_min'], 'mean_delta_f1':d['f1']['mean'] >= a['mean_delta_f1_min'], 'joint_nonnegative_splits':sum(joint) >= a['nonnegative_splits_min_each'], 'hc_no_excess_drop':d['hc_auc']['mean'] >= -a['health_group_mean_auc_drop_max'], 'dep_no_excess_drop':d['dep_auc']['mean'] >= -a['health_group_mean_auc_drop_max']}
    return {'accepted':all(checks.values()), 'checks':checks, 'joint_nonnegative_values':joint}


def provenance(c, splits, paths):
    inputs = [(HERE / c[k]).resolve() for k in ('spectral_cache','temporal_cache','pe_cache','fixed_reference')]
    inputs += [OLD / 'predictions' / f'{n}_oof.csv' for n, _ in splits] + paths
    sources = list(HERE.glob('*.py')) + [HERE / 'config.json', HERE / 'test_protocol.txt']
    historical = json.loads((OLD / 'input_metadata/provenance.json').read_text(encoding='utf8'))
    sources += [Path(p) for p in historical['sources']]
    sources += [OLD / 'input_metadata/provenance.json', ROOT / 'src/emo_rec/features.py', ROOT / 'src/emo_rec/covariance.py', HERE.parent / '084_raw_covariance_alignment/run_experiment.py']
    sources += list((ROOT / 'src/emo_rec').glob('*.py'))
    return {'inputs':{str(p):sha256(p) for p in inputs}, 'sources':{str(p):sha256(p) for p in sources}, 'public_arrays_loaded':False, 'model_fits':30, 'n_jobs':10, 'threads_per_job':1, 'python':platform.python_version(), 'numpy':np.__version__, 'sklearn':sklearn.__version__, 'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(), 'cpu_exception':'HGB and identical scipy spectral extraction; CPU covariance only numerical verification', 'control_source':'070 p_train8', 'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}


def main():
    start = time.perf_counter(); c = json.loads((HERE / 'config.json').read_text(encoding='utf8')); validate_config(c)
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    with public_guard(), threadpool_limits(limits=1):
        a8, ref, fixed, paths, checks = load_data(c)
        splits = e070.splits_for(ref, fixed, c)
        controls = {}
        for name, folds in splits:
            base.assert_subject_isolation(ref.subject.to_numpy(), folds)
            controls[name] = pd.read_csv(OLD / 'predictions' / f'{name}_oof.csv')
            validate_control(ref, folds, controls[name])
        prov = provenance(c, splits, paths); prov['feature_checks'] = checks; prov['active_threadpools'] = threadpool_info()
        for folder in ('predictions','input_metadata'): (HERE / folder).mkdir(exist_ok=True)
        for name, folds in splits: ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
        (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
        print('GPU whitening verified; a8 applied; control aligned; starting 30 fits', flush=True)
        results = Parallel(n_jobs=10, prefer='threads')(delayed(fit_one)(a8, ref.y.to_numpy(), n, folds, fold, c) for n, folds in splits for fold in range(5))
    assert len(results) == 30
    runs, rows, deltas = {}, [], []
    for name, folds in splits:
        frame = ref.assign(fold=folds, p_control8=controls[name].p_train8.to_numpy(), p_alignment8=np.nan)
        for n, fold, p, _, _ in results:
            if n == name: frame.loc[folds == fold, 'p_alignment8'] = p
        assert np.isfinite(frame.p_alignment8).all()
        frame.to_csv(HERE / 'predictions' / f'{name}_oof.csv', index=False); runs[name] = {}
        for subset in ['all_rows'] + [str(i) for i in c['positions']]:
            q = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            r = {arm:summarize(q, col) for arm,col in (('control8','p_control8'),('alignment8','p_alignment8'))}
            r['delta'] = {k:scalars(r['alignment8'])[k]-scalars(r['control8'])[k] for k in KEYS}; runs[name][subset] = r
            rows += [{'split':name,'position':subset,'arm':arm,**scalars(r[arm])} for arm in ('control8','alignment8')]
            deltas.append({'split':name,'position':subset,**r['delta']})
    agg = aggregate_runs(runs,c); decision = acceptance(agg,c); prov['elapsed_seconds'] = time.perf_counter()-start
    metrics = {'runs':runs,'aggregate':agg,'acceptance':decision,'provenance':prov,'fit_seconds':[{'split':r[0],'fold':r[1],'seconds':r[3],'n_train':r[4]} for r in results]}
    for path,obj in ((HERE/'metrics.json',metrics),(HERE/'input_metadata/provenance.json',prov)): path.write_text(json.dumps(obj,indent=2),encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False); pd.DataFrame(deltas).to_csv(HERE/'paired_delta.csv',index=False)
    lines = ['# 实验086：原始协方差白化谱 + a8', '', '仅替换150谱列；原450 temporal与120 PE在raw及a8层均逐元素不变。GPU float64白化，全部345个真实8窗上下文对照084数学定义；随后明确使用070.contexts(...)[1]。', '', '|split|control AUC|control F1|086 AUC|086 F1|ΔAUC|ΔF1|HC ΔAUC|DEP ΔAUC|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n,_ in splits:
        r=runs[n]['all_rows']; ctrl=scalars(r['control8']); tr=scalars(r['alignment8']); d=r['delta']
        lines.append(f"|{n}|{ctrl['auc']:.6f}|{ctrl['f1']:.6f}|{tr['auc']:.6f}|{tr['f1']:.6f}|{d['auc']:+.6f}|{d['f1']:+.6f}|{d['hc_auc']:+.6f}|{d['dep_auc']:+.6f}|")
    a=agg['all_rows']; d=a['delta']
    lines += ['',f"五重复086 AUC={a['alignment8']['auc']['mean']:.6f}，F1={a['alignment8']['f1']['mean']:.6f}；控制AUC={a['control8']['auc']['mean']:.6f}，F1={a['control8']['f1']['mean']:.6f}。",f"平均ΔAUC={d['auc']['mean']:+.6f}，ΔF1={d['f1']['mean']:+.6f}；joint={sum(decision['joint_nonnegative_values'])}/5；接受={decision['accepted']}。",f"验收明细：{decision['checks']}。fixed002不纳入汇总。",'', '## 测试与可追溯性','测试前后记录见pytest_before.txt、pytest_after.txt；完整复算由test_086.py执行。全部缓存、fixed、69个训练mat、070递归依赖、协议及测试/代码hash见provenance。公共NPZ键/路径/loader运行时guard；无公开数组读取。30 fits，10 jobs，全局threadpool=1，无扫参。', '', '## 方法来源与局限', '[原论文](https://arxiv.org/html/1808.05464v2)验证MI/ERP而非本情绪集；[2025综述](https://arxiv.org/html/2502.09203v1) §2.1均值cov逆根、§2.2建议temporal filtering后spatial前。两来源同作者，不是独立交叉验证；仅为方法动机，迁移效果以本次OOF为准。', 'W C_regularized W=I，不代表变换原始窗的经验平均协方差为I。本轮仅谱EA，非全特征EA。084漏a8，不能作为白化单变量证据。position仅用于训练OOF8窗模拟，不用于公开位置路由；未证明公开集泛化。五重复相互重叠，不是独立外部验证；复用历史控制且训练集被反复探索。HGB CPU例外，谱提取复用原CPU实现以保持数值定义。']
    (HERE/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'acceptance':decision,'all_rows':a,'elapsed':prov['elapsed_seconds']},indent=2),flush=True)


if __name__ == '__main__': main()
