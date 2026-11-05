import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
import sys
sys.path.insert(0, str(ROOT / 'src'))
BASE = HERE.parent / '036_trial_consistency_features' / 'run_experiment.py'
import importlib.util
spec = importlib.util.spec_from_file_location('base036', BASE)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def folds_for(subjects, groups, seed):
    unique = np.unique(subjects)
    ug = np.asarray([groups[np.flatnonzero(subjects == s)[0]] for s in unique])
    assignment = {}
    for fold, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(unique, ug)):
        for s in unique[valid]: assignment[s] = fold
    return np.asarray([assignment[s] for s in subjects])


def trial_features(X, subjects, trial_ids, windows_per_trial=5):
    X = np.asarray(X, dtype=np.float64)
    out = np.empty((len(X), X.shape[1] * 2 + 2), dtype=np.float64)
    groups = {}
    seen = {}
    for i, (s, t) in enumerate(zip(subjects, trial_ids)):
        key = (s, int(t)); occurrence = seen.get(key, 0); seen[key] = occurrence + 1
        groups.setdefault((s, int(t), occurrence // windows_per_trial), []).append(i)
    for (s, t, block), rows in groups.items():
        rows = np.asarray(rows)
        if len(rows) != windows_per_trial:
            raise RuntimeError(f'physical trial {(s, t)} has {len(rows)} windows, expected {windows_per_trial}')
        values = X[rows]
        mean, std = values.mean(0), values.std(0)
        centered = values - mean
        norms = np.linalg.norm(centered, axis=1)
        corr = centered @ centered.T / np.maximum(norms[:, None] * norms[None, :], 1e-12)
        consistency = (corr.sum(1) - 1) / (len(rows) - 1)
        trial_consistency = float((corr.sum() - len(rows)) / (len(rows) * (len(rows) - 1)))
        out[rows] = np.column_stack([np.broadcast_to(mean, values.shape), np.broadcast_to(std, values.shape), consistency, np.full(len(rows), trial_consistency)])
    if not np.isfinite(out).all(): raise RuntimeError('non-finite trial features')
    return np.column_stack([X, out])


def metrics(y, p):
    pred = p >= 0.5; tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'auc': float(roc_auc_score(y, p)), 'f1': float(f1_score(y, pred, zero_division=0)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def summary(y, p, ref, folds):
    subj = []
    for s in np.unique(ref.subject_id):
        m = ref.subject_id.to_numpy() == s; subj.append({'subject_id': s, 'health_group': ref.loc[m, 'health_group'].iloc[0], **metrics(y[m], p[m])})
    groups = {}
    for g in ('HC', 'DEP'):
        m = ref.health_group.to_numpy() == g; rows = [x for x in subj if x['health_group'] == g]
        groups[g] = {'n_subjects': len(rows), 'pooled': metrics(y[m], p[m]), 'subject_auc_mean': float(np.mean([x['auc'] for x in rows]))}
    return {'overall': metrics(y, p), 'folds': [{'fold': f, **metrics(y[folds == f], p[folds == f])} for f in range(5)], 'subject_auc_mean': float(np.mean([x['auc'] for x in subj])), 'health_groups': groups, 'subjects': subj}


def model(config, seed): return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config['hgb'], random_state=seed))


def main():
    start = time.perf_counter(); config = json.loads((HERE / 'config.json').read_text(encoding='utf8')); out = HERE / 'predictions'; out.mkdir(exist_ok=True)
    cache = base.load_cache((HERE / config['spectral_cache']).resolve()); keep = cache['subject_ids'] != config['excluded_subject']
    subjects, groups, y = cache['subject_ids'][keep], cache['health_groups'][keep], cache['y_train'][keep]
    if len(y) != 2760 or len(np.unique(subjects)) != 69: raise RuntimeError('unexpected filtered data')
    X = base.normalize_by_subject(cache['X_train'][keep, :150], subjects, 'zscore')
    XP = base.normalize_by_subject(cache['X_public'][:, :150], cache['public_user_ids'], 'zscore')
    trial_ids = cache['original_trial_ids'][keep]
    F = trial_features(X, subjects, trial_ids, config['train_windows_per_trial'])
    # Public labels are never loaded; deterministic 5+3 grouping is only for feature construction.
    public_trials = np.repeat(np.arange(1, 9), 1)
    FP = trial_features(XP, cache['public_user_ids'], public_trials, config['train_windows_per_trial']) if False else None
    occurrence = {}
    physical_keys = []
    for s, t in zip(subjects, trial_ids):
        key = (s, int(t)); n = occurrence.get(key, 0); physical_keys.append(f'{s}_{int(t)}_{n // config["train_windows_per_trial"]}'); occurrence[key] = n + 1
    physical_keys = np.asarray(physical_keys)
    ref = pd.DataFrame({'subject_id': subjects, 'health_group': groups, 'original_trial_id': trial_ids, 'physical_trial_key': physical_keys, 'y_true': y})
    ref.attrs['public_user_ids'] = cache['public_user_ids']; ref.attrs['public_trial_ids'] = cache['public_trial_ids']
    prov = {'inputs': {'spectral_cache': str((HERE / config['spectral_cache']).resolve()), 'spectral_cache_sha256': sha256((HERE / config['spectral_cache']).resolve()), 'plus_pe_oof_dir': str((HERE / config['plus_pe_oof_dir']).resolve())}, 'config_sha256': sha256(HERE / 'config.json'), 'script_sha256': sha256(Path(__file__)), 'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'train_shape': list(F.shape), 'physical_trial_groups': int(len(np.unique(physical_keys))), 'trial_windows': config['train_windows_per_trial'], 'public_labels_read': False, 'subject_isolation': True}
    result = {'config': config, 'provenance': prov, 'runs': {}}
    rows = []
    with threadpool_limits(limits=1):
        for seed in config['random_seeds']:
            folds = folds_for(subjects, groups, seed); base.assert_subject_isolation(subjects, folds)
            p_trial = np.full(len(y), np.nan); p_fusion = np.full(len(y), np.nan)
            plus = pd.read_csv((HERE / config['plus_pe_oof_dir']) / f'plus_pe_repeat{seed}_oof.csv')
            plus = plus[plus.subject_id != config['excluded_subject']].reset_index(drop=True)
            np.testing.assert_array_equal(plus.subject_id.to_numpy(), subjects); np.testing.assert_array_equal(plus.original_trial_id.to_numpy(), trial_ids)
            for fold in range(5):
                train, valid = folds != fold, folds == fold
                m = model(config, config['model_seed_base'] + fold); m.fit(F[train], y[train]); p_trial[valid] = m.predict_proba(F[valid])[:, 1]
            fusion = config['fusion']['trial_stats_weight'] * p_trial + config['fusion']['plus_pe_weight'] * plus.prob_positive.to_numpy()
            name = f'trial_stats_seed{seed}'; fr = ref.copy(); fr['fold'] = folds; fr['prob_positive'] = p_trial; fr.to_csv(out / f'{name}_oof.csv', index=False)
            ff = ref.copy(); ff['fold'] = folds; ff['prob_positive'] = fusion; ff.to_csv(out / f'fixed_trial_stats_plus_pe_seed{seed}_oof.csv', index=False)
            for n, p in [('trial_stats', p_trial), ('fixed_trial_stats_plus_pe', fusion)]:
                r = summary(y, p, ref, folds); result['runs'].setdefault(n, {})[str(seed)] = r; rows.append({'method': n, 'seed': seed, **r['overall'], 'subject_auc_mean': r['subject_auc_mean'], 'hc_auc': r['health_groups']['HC']['pooled']['auc'], 'dep_auc': r['health_groups']['DEP']['pooled']['auc']})
            print(f'completed seed {seed}/5 trial_AUC={rows[-2]["auc"]:.6f} fusion_AUC={rows[-1]["auc"]:.6f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(HERE / 'summary.csv', index=False)
    agg = {}
    for n in ('trial_stats', 'fixed_trial_stats_plus_pe'):
        agg[n] = {k: {'mean': float(frame.loc[frame.method == n, k].mean()), 'std': float(frame.loc[frame.method == n, k].std(ddof=1))} for k in ('auc', 'f1', 'balanced_accuracy', 'subject_auc_mean', 'hc_auc', 'dep_auc')}
    result['aggregate'] = agg; result['elapsed_seconds'] = time.perf_counter() - start; result['completed_runs'] = len(rows)
    (HERE / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    lines = ['# 实验216：顺序不变的物理trial内频谱统计与固定融合', '', '严格排除HC43后，以subject级健康分层5-fold隔离；每个训练物理trial由original_trial_id显式识别且恰为连续5个窗口，但统计特征不依赖窗口行序：trial内均值、std、每窗口对其他窗口的平均相关性及trial平均相关性。未读取公开标签。', '', '|方法|窗口AUC|窗口F1|DEP AUC|HC AUC|', '|---|---:|---:|---:|---:|']
    for n, a in agg.items(): lines.append(f'|{n}|{a["auc"]["mean"]:.6f}±{a["auc"]["std"]:.6f}|{a["f1"]["mean"]:.6f}±{a["f1"]["std"]:.6f}|{a["dep_auc"]["mean"]:.6f}±{a["dep_auc"]["std"]:.6f}|{a["hc_auc"]["mean"]:.6f}±{a["hc_auc"]["std"]:.6f}|')
    lines += ['', '## trial级指标', '训练窗口标签属于单窗口，但trial级预测按5窗口概率均值聚合；同一trial内标签一致，因此按trial聚合后计算AUC/F1。']
    for n in ('trial_stats', 'fixed_trial_stats_plus_pe'):
        vals = []
        for seed in config['random_seeds']:
            d = pd.read_csv(out / (f'trial_stats_seed{seed}_oof.csv' if n == 'trial_stats' else f'fixed_trial_stats_plus_pe_seed{seed}_oof.csv'))
            d['trial_key'] = d.physical_trial_key; g = d.groupby('trial_key').agg(y_true=('y_true', 'first'), p=('prob_positive', 'mean')); vals.append(metrics(g.y_true.to_numpy(), g.p.to_numpy()))
        lines.append(f'- {n} trial AUC={np.mean([v["auc"] for v in vals]):.6f}±{np.std([v["auc"] for v in vals], ddof=1):.6f}, trial F1={np.mean([v["f1"] for v in vals]):.6f}±{np.std([v["f1"] for v in vals], ddof=1):.6f}.')
    lines += ['', f'完成{len(rows)}个seed运行，耗时{result["elapsed_seconds"]:.1f}秒。OOF、metrics.json、summary.csv及provenance审计见本目录；057 plus_pe仅作为预声明固定0.5/0.5融合输入。']
    (HERE / '实验报告.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    (HERE / 'provenance.json').write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding='utf8')
    print(frame.groupby('method')[['auc','f1','dep_auc']].mean().to_string(), flush=True)

if __name__ == '__main__': main()
