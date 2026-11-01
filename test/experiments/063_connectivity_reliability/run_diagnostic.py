import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
SOURCE = (HERE / '../062_orthogonal_envelope_connectivity').resolve()
spec = importlib.util.spec_from_file_location('exp062', SOURCE / 'run_experiment.py')
exp062 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp062)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def corr(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    ac = a - a.mean(); bc = b - b.mean()
    den = np.sqrt(np.sum(ac * ac) * np.sum(bc * bc))
    return np.nan if den == 0 else float(np.sum(ac * bc) / den)


def summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {'finite_n': 0, 'median': None, 'q1': None, 'q3': None}
    return {'finite_n': int(len(finite)), 'median': float(np.median(finite)),
            'q1': float(np.quantile(finite, .25)), 'q3': float(np.quantile(finite, .75))}


def load_raw(config, cache):
    from emo_rec.data import load_train_segments
    files = sorted((ROOT / '数据/训练集').glob('*/*.mat'))
    segments = []; subjects = []; groups = []; ys = []; trials = []
    for path in files:
        x, y, trial = load_train_segments(path)
        segments.append(x); ys.append(y); trials.append(trial)
        subjects.extend([path.stem] * len(x)); groups.extend(['DEP' if path.parent.name == '抑郁症患者' else 'HC'] * len(x))
    raw_subjects = np.asarray(subjects); raw_groups = np.asarray(groups); raw_y = np.concatenate(ys); raw_trials = np.concatenate(trials)
    for name, actual, expected in [('subject_ids', raw_subjects, cache['subject_ids']), ('health_groups', raw_groups, cache['health_groups']), ('y_train', raw_y, cache['y_train']), ('original_trial_ids', raw_trials, cache['original_trial_ids'])]:
        np.testing.assert_array_equal(actual, expected, err_msg=f'raw/cache row alignment mismatch: {name}')
    return np.concatenate(segments), raw_subjects, raw_groups, raw_y, raw_trials, files


def make_features(x, config):
    fs = config['sampling_rate']; trim = fs * config['central_trim_seconds']; n = x.shape[-1]
    if x.shape != (x.shape[0], 30, fs * config['window_seconds']):
        raise ValueError(f'unexpected raw shape {x.shape}')
    conn = []; control = []; full_blocks = []
    for band in config['bands']:
        sos = butter(config['filter_order'], band, btype='bandpass', fs=fs, output='sos')
        z = hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)[..., trim:-trim]
        full_blocks.append(np.asarray([exp062.orthogonal_envelope_matrix(w).sum(axis=1) / 29 for w in z]))
        halves = np.split(z, 2, axis=-1)
        c_halves = []; p_halves = []
        for half in halves:
            c_halves.append(np.asarray([exp062.orthogonal_envelope_matrix(w).sum(axis=1) / 29 for w in half]))
            p_halves.append(np.log(np.mean(np.abs(half) ** 2, axis=-1)))
        conn.append((c_halves[0], c_halves[1])); control.append((p_halves[0], p_halves[1]))
    return conn, control, np.concatenate(full_blocks, axis=1)


def reliability_rows(subject, conn, control, bands):
    rows = []
    for bi, band in enumerate(bands):
        c0, c1 = conn[bi]; p0, p1 = control[bi]
        cr = np.asarray([corr(c0[:, j], c1[:, j]) for j in range(30)])
        pr = np.asarray([corr(p0[:, j], p1[:, j]) for j in range(30)])
        for feature, values in [('oAEC_node', cr), ('log_power_channel', pr)]:
            s = summary(values)
            rows.append({'subject_id': subject, 'band': f'{band[0]}-{band[1]}', 'feature': feature, **s})
    return rows


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def oof_deltas(config, ref):
    rows = []; split_stats = []
    for split in config['oof_splits']:
        frames = {}
        for method in config['oof_methods']:
            path = SOURCE / 'predictions' / f'{method}_{split}_oof.csv'
            frame = pd.read_csv(path)
            pd.testing.assert_frame_equal(frame[['subject_id', 'health_group', 'original_trial_id', 'y_true']], ref[['subject_id', 'health_group', 'original_trial_id', 'y_true']], check_dtype=False)
            frames[method] = frame
        control = frames['control_pe']; plus = frames['plus_oAEC']; y = control.y_true.to_numpy()
        pred_corr = corr(control.prob_positive.to_numpy(), plus.prob_positive.to_numpy())
        brier_change = brier(y, plus.prob_positive) - brier(y, control.prob_positive)
        deltas = []
        for subject, idx in control.groupby('subject_id').groups.items():
            yi = y[idx]
            if len(np.unique(yi)) < 2:
                delta = np.nan
            else:
                delta = float(roc_auc_score(yi, plus.prob_positive.to_numpy()[idx]) - roc_auc_score(yi, control.prob_positive.to_numpy()[idx]))
            deltas.append(delta)
            rows.append({'split': split, 'subject_id': subject, 'auc_delta_plus_minus_control': delta})
        finite = np.asarray(deltas)[np.isfinite(deltas)]
        split_stats.append({'split': split, 'prediction_pearson': pred_corr, 'brier_change_plus_minus_control': brier_change,
                            'subject_auc_delta_finite_n': int(len(finite)), 'subject_auc_delta_positive_n': int(np.sum(finite > 0)),
                            'subject_auc_delta_negative_n': int(np.sum(finite < 0)), 'subject_auc_delta_zero_n': int(np.sum(finite == 0)),
                            'subject_auc_delta_median': None if not len(finite) else float(np.median(finite))})
    return split_stats, rows


def main():
    started = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    source_config = json.loads((SOURCE / 'config.json').read_text(encoding='utf8'))
    assert config['sampling_rate'] == source_config['sampling_rate']
    cache_path = (HERE / config['source_cache']).resolve()
    cache = exp062.base.load_cache(cache_path)
    raw, subjects, groups, y, trials, raw_files = load_raw(config, cache)
    keep = subjects != config['excluded_subject']
    x = raw[keep]; kept_subjects = subjects[keep]
    conn, control, full_conn = make_features(x, config)
    expected = cache['X_train'][keep]
    np.testing.assert_allclose(full_conn, expected, atol=1e-12, rtol=0)
    summary_rows = []
    for subject in np.unique(kept_subjects):
        idx = kept_subjects == subject
        per_conn = [(a[idx], b[idx]) for a, b in conn]
        per_control = [(a[idx], b[idx]) for a, b in control]
        summary_rows.extend(reliability_rows(subject, per_conn, per_control, config['bands']))
    summary_path = HERE / 'summary.csv'; pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    ref = pd.DataFrame({'subject_id': subjects[keep], 'health_group': groups[keep], 'original_trial_id': trials[keep], 'y_true': y[keep]})
    split_stats, delta_rows = oof_deltas(config, ref)
    for split in config['oof_splits']:
        pd.DataFrame([r for r in delta_rows if r['split'] == split]).to_csv(HERE / f'subject_delta_{split}.csv', index=False)
    input_paths = [HERE / 'run_diagnostic.py', HERE / 'config.json', SOURCE / 'run_experiment.py', SOURCE / 'config.json', cache_path]
    oof_paths = [SOURCE / 'predictions' / f'{m}_{s}_oof.csv' for s in config['oof_splits'] for m in config['oof_methods']]
    input_paths += oof_paths + raw_files
    hashes = {str(p.resolve()): sha256(p) for p in input_paths}
    elapsed = time.perf_counter() - started
    result = {'experiment_id': config['experiment_id'], 'protocol': str(HERE / 'protocol.txt'), 'excluded_subject': config['excluded_subject'],
              'sampling_rate': source_config['sampling_rate'], 'raw_cache_alignment': True, 'full8s_cache_allclose': True,
              'n_subjects': int(len(np.unique(kept_subjects))), 'windows_per_subject': 40, 'reliability_summary': summary_rows,
              'oof_complementarity': split_stats, 'input_sha256': hashes, 'elapsed_seconds': elapsed,
              'python': platform.python_version(), 'numpy': np.__version__}
    (HERE / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'summary_csv': str(summary_path), 'subjects': result['n_subjects'], 'split_stats': split_stats, 'elapsed_seconds': elapsed}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
