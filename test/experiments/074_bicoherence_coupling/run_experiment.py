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

TRIPLES = ((6, 6, 12), (6, 10, 16), (10, 10, 20), (6, 20, 26), (10, 20, 30))


def bicoherence(segments, device='cuda'):
    x = torch.as_tensor(segments, dtype=torch.float64, device=device)
    if x.ndim != 3 or tuple(x.shape[1:]) != (30, 2500):
        raise ValueError('Expected independent (n,30,2500) windows')
    nperseg, hop, nfft = 500, 250, 500
    frames = x.unfold(-1, nperseg, hop)
    window = torch.hann_window(nperseg, dtype=x.dtype, device=x.device)
    spec = torch.fft.rfft(frames * window, n=nfft, dim=-1)
    # nfft=500 at 250 Hz gives exact 0.5-Hz bins; integer-Hz frequencies use bins 2*f.
    out = []
    for f1, f2, f3 in TRIPLES:
        a, b, c = spec[..., 2 * f1], spec[..., 2 * f2], spec[..., 2 * f3]
        bis = a * b * torch.conj(c)
        numerator = torch.abs(bis.mean(dim=-1)) ** 2
        denominator = (torch.abs(a * b) ** 2).mean(dim=-1) * (torch.abs(c) ** 2).mean(dim=-1)
        out.append(torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator)))
    result = torch.cat(out, dim=1).clamp(0, 1).cpu().numpy()
    assert np.isfinite(result).all()
    return result


def build_cache(config):
    feats, records, hashes = [], [], {}
    device = config['feature_device'] if torch.cuda.is_available() else 'cpu'
    for path in sorted((ROOT / '数据/训练集').glob('*/*.mat')):
        hashes[str(path)] = base.sha256(path)
        segments, labels, trials = load_train_segments(path)
        feats.append(bicoherence(segments, device))
        for i in range(len(segments)):
            records.append((path.stem, 'DEP' if path.parent.name == '抑郁症患者' else 'HC', trials[i], labels[i], i % 5))
    ref = pd.DataFrame(records, columns=['subject', 'group', 'trial', 'y', 'position'])
    keep = ref.subject != config['excluded_subject']
    ref = ref.loc[keep].reset_index(drop=True)
    features = np.concatenate(feats)[keep]
    names = [f'bicoherence_{f1}_{f2}_{f3}_ch{ch:02d}' for f1, f2, f3 in TRIPLES for ch in range(1, 31)]
    np.savez_compressed(HERE / 'cache/bicoherence.npz', X_train=features,
                        **{k: ref[k].to_numpy(dtype=str if k in ('subject', 'group') else np.int64) for k in ref},
                        feature_names=np.asarray(names), config_sha256=np.asarray(base.sha256(HERE / 'config.json')),
                        raw_input_hashes=np.asarray(json.dumps(hashes)))
    return features, ref, hashes, device


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
    return name, fold, *predictions, time.perf_counter() - start


def summarize(frame, column):
    rr = frame.rename(columns={'subject': 'subject_id', 'group': 'health_group'})
    return base.summarize(frame.y.to_numpy(), frame[column].to_numpy(), rr, frame.fold.to_numpy())


def aggregate(deltas, config):
    out = {}
    for subset in ['all_rows'] + [str(i) for i in config['positions']]:
        selected = [r for r in deltas if r['split'].startswith('repeat') and r['position'] == subset]
        out[subset] = {k: {'mean': float(np.mean([r[k] for r in selected])), 'std': float(np.std([r[k] for r in selected], ddof=1)), 'values': [r[k] for r in selected]} for k in e.KEYS}
    return out


def acceptance(agg, config):
    d, a = agg['all_rows'], config['acceptance']
    joint = sum(x >= 0 and y >= 0 for x, y in zip(d['auc']['values'], d['f1']['values']))
    checks = {'mean_delta_auc': d['auc']['mean'] >= a['mean_delta_auc_min'], 'mean_delta_f1': d['f1']['mean'] >= a['mean_delta_f1_min'],
              'joint_nonnegative': joint >= a['nonnegative_splits_min_each'], 'hc_no_excess_drop': d['hc_auc']['mean'] >= -a['health_group_mean_auc_drop_max'],
              'dep_no_excess_drop': d['dep_auc']['mean'] >= -a['health_group_mean_auc_drop_max']}
    return {'accepted': all(checks.values()), 'checks': checks, 'joint_nonnegative_count': joint}


def main():
    start = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text())
    for folder in ('cache', 'predictions', 'input_metadata'):
        (HERE / folder).mkdir(exist_ok=True)
    raw, ref, fixed = e.load_data(config)
    extra, extra_ref, hashes, device = build_cache(config)
    assert_alignment(ref, extra_ref)
    _, control = e.contexts(raw, ref.subject.to_numpy(), ref.position.to_numpy())
    plus = np.column_stack([control, normalize_extra(extra, ref.subject.to_numpy(), ref.position.to_numpy())])
    assert control.shape == (2760, 720) and plus.shape == (2760, 870)
    splits = e.splits_for(ref, fixed, config)
    inputs = [(HERE / config[k]).resolve() for k in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')]
    sources = [HERE / n for n in ('run_experiment.py', 'test_074.py', 'config.json', 'test_protocol.txt')]
    sources += [Path(e.__file__), Path(e.e.__file__), Path(base.__file__), ROOT / 'src/emo_rec/data.py', ROOT / 'src/emo_rec/normalization.py', ROOT / 'pyproject.toml', ROOT / '数据/数据集说明文档.md']
    prov = {'raw_inputs': hashes, 'inputs': {str(p): base.sha256(p) for p in inputs}, 'sources': {str(p): base.sha256(p) for p in sources}, 'public_arrays_loaded': False, 'control_shape': list(control.shape), 'treatment_shape': list(plus.shape), 'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'torch': torch.__version__, 'feature_device': device, 'feature_dtype': 'float64', 'cpu_exception': 'Frozen sklearn HGB has no GPU implementation', 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    for name, folds in splits:
        base.assert_subject_isolation(ref.subject.to_numpy(), folds)
        ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    with threadpool_limits(1):
        prov['active_threadpools'] = threadpool_info()
        results = Parallel(n_jobs=config['n_jobs'], prefer='threads')(delayed(fit_pair)(control, plus, ref.y.to_numpy(), name, folds, fold, config) for name, folds in splits for fold in range(5))
    assert len(results) * 2 == config['expected_model_fits']
    runs, rows, deltas, checks = {}, [], [], {}
    for name, folds in splits:
        frame = ref.assign(fold=folds, p_control=np.nan, p_plus=np.nan)
        for n, fold, pc, pp, _ in results:
            if n == name:
                frame.loc[folds == fold, 'p_control'], frame.loc[folds == fold, 'p_plus'] = pc, pp
        old = pd.read_csv(OLD / 'predictions' / f'{name}_oof.csv')
        for col in ('row_index', 'position', 'subject', 'group', 'trial', 'y', 'fold'):
            np.testing.assert_array_equal(frame[col], old[col])
        checks[name] = float(np.max(np.abs(frame.p_control - old.p_train8)))
        assert checks[name] <= 1e-12
        frame.to_csv(HERE / 'predictions' / f'{name}_oof.csv', index=False)
        runs[name] = {}
        for subset in ['all_rows'] + [str(i) for i in config['positions']]:
            q = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            runs[name][subset] = {arm: summarize(q, col) for arm, col in (('control', 'p_control'), ('plus', 'p_plus'))}
            scalar = {arm: e.scalars(r) for arm, r in runs[name][subset].items()}
            for arm in scalar: rows.append({'split': name, 'position': subset, 'arm': arm, **scalar[arm]})
            deltas.append({'split': name, 'position': subset, **{k: scalar['plus'][k] - scalar['control'][k] for k in e.KEYS}})
    agg, decision = aggregate(deltas, config), None
    decision = acceptance(agg, config)
    prov.update(model_fits=60, elapsed_seconds=time.perf_counter() - start, cache_sha256=base.sha256(HERE / 'cache/bicoherence.npz'))
    metrics = {'runs': runs, 'aggregate_delta': agg, 'acceptance': decision, 'control_070_max_abs_diff': checks, 'provenance': prov, 'fit_seconds': [{'split': r[0], 'fold': r[1], 'pair_seconds': r[4]} for r in results]}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(deltas).to_csv(HERE / 'paired_delta.csv', index=False)
    d = agg['all_rows']
    text = f'# 实验074：窗口归一化双相干特征\n\n接受：{decision["accepted"]}；五新split联合非负 {decision["joint_nonnegative_count"]}/5。\n\n' + '\n'.join(f'- 五新split平均Δ{k}: {d[k]["mean"]:.9f}' for k in ('auc', 'f1', 'hc_auc', 'dep_auc')) + f'\n\n固定双相干150维，控制720维、处理870维；设备{device}；60 fits。控制六套逐行复现070，最大误差{max(checks.values()):.3g}。未访问公开数组；不调参。\n'
    (HERE / 'summary.md').write_text(text, encoding='utf8')
    print(json.dumps({'acceptance': decision, 'all_rows_delta': d, 'elapsed_seconds': prov['elapsed_seconds']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
