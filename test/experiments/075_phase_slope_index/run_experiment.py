import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('exp074_helpers', HERE.parent / '074_bicoherence_coupling/run_experiment.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
e, base, OLD = h.e, h.base, h.OLD
assert_alignment, normalize_extra = h.assert_alignment, h.normalize_extra
summarize, aggregate, acceptance = h.summarize, h.aggregate, h.acceptance
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
CHANNELS = 'FP1 FP2 F7 F3 FZ F4 F8 FT7 FC3 FCZ FC4 FT8 T3 C3 CZ C4 T4 TP7 CP3 CPZ CP4 TP8 T5 P3 PZ P4 T6 O1 OZ O2'.split()
SOURCE = [CHANNELS.index(c) for c in CONFIG['source_channels']]
TARGET = [CHANNELS.index(c) for c in CONFIG['target_channels']]


def psi_edges(segments, device='cuda'):
    x = torch.as_tensor(segments, dtype=torch.float64, device=device)
    if x.ndim != 3 or tuple(x.shape[1:]) != (30, 2500):
        raise ValueError('Expected independent (n,30,2500) windows')
    frames = x.unfold(-1, 500, 250)
    frames = frames - frames.mean(dim=-1, keepdim=True)
    window = torch.hann_window(500, periodic=False, dtype=x.dtype, device=x.device)
    spectrum = torch.fft.rfft(frames * window, n=500, dim=-1)
    out = []
    for low, high in CONFIG['bands']:
        z = spectrum[..., 2 * low:2 * high + 1]
        s = torch.einsum('ncsf,ndsf->ncdf', z, z.conj()) / z.shape[-2]
        power = (z.abs() ** 2).mean(dim=-2)
        denominator = torch.sqrt(power[:, :, None, :] * power[:, None, :, :])
        coherence = torch.where(denominator > 0, s / denominator.clamp_min(torch.finfo(x.dtype).tiny), torch.zeros_like(s))
        psi = (coherence[..., :-1].conj() * coherence[..., 1:]).imag.sum(dim=-1)
        # Enforce the exact mathematical antisymmetry, removing roundoff only.
        psi = (psi - psi.transpose(1, 2)) / 2
        out.append(psi)
    return torch.stack(out, dim=1)


def phase_slope_index(segments, device='cuda'):
    edges = psi_edges(segments, device)
    forward = edges[:, :, SOURCE, :][:, :, :, TARGET].sum(dim=(-2, -1))
    positive = edges.clamp_min(0).sum(dim=(-2, -1))
    negative = (-edges).clamp_min(0).sum(dim=(-2, -1))
    result = torch.stack((positive, negative, torch.zeros_like(positive), positive + negative, forward, -forward), dim=-1)
    result = result.reshape(len(segments), 24).cpu().numpy()
    assert np.isfinite(result).all()
    return result


def build_cache(config):
    assert torch.cuda.is_available(), 'Frozen production extraction requires CUDA'
    features, records, hashes = [], [], {}
    for path in sorted((ROOT / '数据/训练集').glob('*/*.mat')):
        if path.stem == config['excluded_subject']:
            continue
        hashes[str(path)] = base.sha256(path)
        windows, labels, trials = h.load_train_segments(path)
        features.append(phase_slope_index(windows))
        records.extend((path.stem, 'DEP' if path.parent.name == '抑郁症患者' else 'HC', trials[i], labels[i], i % 5) for i in range(len(windows)))
    ref = pd.DataFrame(records, columns=['subject', 'group', 'trial', 'y', 'position'])
    extra = np.concatenate(features)
    names = [f'{b}_{s}' for b in config['band_names'] for s in ('positive', 'negative', 'net', 'absolute', 'frontotemporal_to_centroparietal', 'centroparietal_to_frontotemporal')]
    np.savez_compressed(HERE / 'cache/psi.npz', X_train=extra, feature_names=np.asarray(names),
                        **{k: ref[k].to_numpy(dtype=str if k in ('subject', 'group') else np.int64) for k in ref},
                        config_sha256=np.asarray(base.sha256(HERE / 'config.json')), raw_input_hashes=np.asarray(json.dumps(hashes)))
    return extra, ref, hashes


def main():
    start = time.perf_counter()
    config = CONFIG
    for folder in ('cache', 'predictions', 'input_metadata'):
        (HERE / folder).mkdir(exist_ok=True)
    raw, ref, fixed = e.load_data(config)
    extra, extra_ref, hashes = build_cache(config)
    assert_alignment(ref, extra_ref)
    _, control = e.contexts(raw, ref.subject.to_numpy(), ref.position.to_numpy())
    plus = np.column_stack([control, normalize_extra(extra, ref.subject.to_numpy(), ref.position.to_numpy())])
    assert control.shape == (2760, 720) and plus.shape == (2760, 744)
    splits = e.splits_for(ref, fixed, config)
    sources = [HERE / n for n in ('run_experiment.py', 'test_075.py', 'config.json', 'test_protocol.txt')]
    sources += [Path(m.__file__) for m in (h, e, e.e, e.e.e, e.e.e.e, base)]
    sources += [ROOT / p for p in ('src/emo_rec/data.py', 'src/emo_rec/normalization.py', 'pyproject.toml', '数据/数据集说明文档.md')]
    inputs = [(HERE / config[k]).resolve() for k in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')]
    inputs += [OLD / 'predictions' / f'{name}_oof.csv' for name, _ in splits]
    prov = {'raw_inputs': hashes, 'inputs': {str(p): base.sha256(p) for p in inputs}, 'sources': {str(p): base.sha256(p) for p in sources},
            'public_arrays_loaded': False, 'control_shape': list(control.shape), 'treatment_shape': list(plus.shape),
            'feature_device': 'cuda', 'gpu': torch.cuda.get_device_name(), 'feature_dtype': 'float64',
            'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'sklearn': h.sklearn.__version__,
            'cpu_exception': 'Frozen sklearn HGB; metadata IO and subject normalization follow 070', 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    for name, folds in splits:
        base.assert_subject_isolation(ref.subject.to_numpy(), folds)
        ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    print('PSI frozen 24 dimensions; CUDA extraction complete; starting 60 HGB fits', flush=True)
    with threadpool_limits(1):
        results = Parallel(n_jobs=config['n_jobs'], prefer='threads')(delayed(h.fit_pair)(control, plus, ref.y.to_numpy(), name, folds, fold, config) for name, folds in splits for fold in range(5))
    assert len(results) * 2 == config['expected_model_fits']
    runs, rows, deltas, checks = {}, [], [], {}
    for name, folds in splits:
        frame = ref.assign(fold=folds, p_control=np.nan, p_plus=np.nan)
        for n, fold, pc, pp, _ in results:
            if n == name:
                frame.loc[folds == fold, 'p_control'], frame.loc[folds == fold, 'p_plus'] = pc, pp
        assert np.isfinite(frame[['p_control', 'p_plus']]).all().all()
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
            rows.extend({'split': name, 'position': subset, 'arm': arm, **scalar[arm]} for arm in scalar)
            deltas.append({'split': name, 'position': subset, **{k: scalar['plus'][k] - scalar['control'][k] for k in e.KEYS}})
    agg = aggregate(deltas, config)
    decision = acceptance(agg, config)
    prov.update(model_fits=60, elapsed_seconds=time.perf_counter() - start, cache_sha256=base.sha256(HERE / 'cache/psi.npz'))
    metrics = {'runs': runs, 'aggregate_delta': agg, 'acceptance': decision, 'control_070_max_abs_diff': checks, 'provenance': prov,
               'fit_seconds': [{'split': r[0], 'fold': r[1], 'pair_seconds': r[4]} for r in results]}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(deltas).to_csv(HERE / 'paired_delta.csv', index=False)
    text = f'实验075 PSI：接受={decision["accepted"]}；五新split联合非负{decision["joint_nonnegative_count"]}/5。\n'
    text += '\n'.join(f'平均Delta {k}={agg["all_rows"][k]["mean"]:.9f}' for k in ('auc', 'f1', 'hc_auc', 'dep_auc'))
    text += f'\n控制720/处理744维；CUDA float64；60 fits；控制070最大误差{max(checks.values()):.3g}。\n全局净流恒零及其他冗余按协议保留。未访问公开数组，不调参、不挑位置。'
    text += '\n未达到接受线，停止该候选。' if not decision['accepted'] else '\n达到冻结接受线。'
    (HERE / 'summary.txt').write_text(text, encoding='utf8')
    print(text, flush=True)


if __name__ == '__main__':
    main()
