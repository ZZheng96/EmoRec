import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt, welch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('base032', HERE.parent / '032_repeated_subject_folds' / 'run_experiment.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
METHODS = ('control_600', 'plus_ABC', 'plus_D', 'plus_all', 'new_only_ABCD')
KEYS = ('auc', 'f1', 'balanced_accuracy', 'subject_auc_mean', 'hc_auc', 'dep_auc', 'worst_subject_auc')
FAMILIES = ('A_exponent', 'A_offset', 'A_alpha_peak_freq', 'A_alpha_peak_power', 'B_spectral_entropy', 'C_hjorth_mobility', 'C_hjorth_complexity', 'D_asymmetry')
ALIGN_KEYS = ('subject_ids', 'health_groups', 'original_trial_ids', 'y_train', 'public_user_ids', 'public_trial_ids')


def new_window_features(segments, config):
    """A (aperiodic exponent/offset, alpha peak freq/residual power), B (spectral entropy), C (Hjorth) per channel."""
    from emo_rec.features import CHANNEL_NAMES
    x = np.asarray(segments, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != len(CHANNEL_NAMES):
        raise ValueError('expected segments with shape (n, 30, samples)')
    fs = config['sampling_rate']
    f, psd = welch(x, fs=fs, nperseg=config['welch']['nperseg'], noverlap=config['welch']['noverlap'], axis=-1)
    logp = np.log10(np.maximum(psd, 1e-30))
    lo, hi = config['aperiodic_fit_range_hz']; ex_lo, ex_hi = config['aperiodic_exclude_hz']
    fit = (f >= lo) & (f <= hi) & ~((f >= ex_lo) & (f <= ex_hi))
    design = np.column_stack([np.log10(f[fit]), np.ones(fit.sum())])
    coef, *_ = np.linalg.lstsq(design, logp[..., fit].reshape(-1, fit.sum()).T, rcond=None)
    slope = coef[0].reshape(x.shape[:2]); offset = coef[1].reshape(x.shape[:2])
    a_lo, a_hi = config['alpha_peak_range_hz']; am = (f >= a_lo) & (f <= a_hi)
    resid = logp[..., am] - (slope[..., None] * np.log10(f[am])[None, None, :] + offset[..., None])
    peak = resid.argmax(axis=-1)
    alpha_freq = f[am][peak]; alpha_power = resid.max(axis=-1)
    e_lo, e_hi = config['entropy_range_hz']; em = (f >= e_lo) & (f <= e_hi)
    p = psd[..., em] / np.maximum(psd[..., em].sum(axis=-1, keepdims=True), 1e-300)
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-300)), axis=-1) / np.log(em.sum())
    sos = butter(4, config['hjorth_bandpass_hz'], btype='band', fs=fs, output='sos')
    xf = sosfiltfilt(sos, x, axis=-1); dx = np.diff(xf, axis=-1); ddx = np.diff(dx, axis=-1)
    v0, v1, v2 = (np.maximum(np.var(a, axis=-1), 1e-30) for a in (xf, dx, ddx))
    mobility = np.sqrt(v1 / v0); complexity = np.sqrt(v2 / v1) / mobility
    blocks = (('aperiodic_exponent', -slope), ('aperiodic_offset', offset), ('alpha_peak_freq', alpha_freq), ('alpha_peak_power', alpha_power), ('spectral_entropy', entropy), ('hjorth_mobility', mobility), ('hjorth_complexity', complexity))
    names = [f'{prefix}_{ch}' for prefix, _ in blocks for ch in CHANNEL_NAMES]
    return np.concatenate([b for _, b in blocks], axis=1), names


def family_of(name):
    for prefix, fam in (('aperiodic_exponent', 'A_exponent'), ('aperiodic_offset', 'A_offset'), ('alpha_peak_freq', 'A_alpha_peak_freq'), ('alpha_peak_power', 'A_alpha_peak_power'), ('spectral_entropy', 'B_spectral_entropy'), ('hjorth_mobility', 'C_hjorth_mobility'), ('hjorth_complexity', 'C_hjorth_complexity'), ('asymmetry', 'D_asymmetry')):
        if name.startswith(prefix): return fam
    raise ValueError(name)


def build_feature_cache(config, path):
    from emo_rec.data import load_public_test_segments, load_train_segments
    feats, labels, subjects, groups, trials = [], [], [], [], []
    names = None
    train_files = sorted((ROOT / '数据' / '训练集').glob('*/*.mat'))
    for index, file in enumerate(train_files, start=1):
        segments, y, t = load_train_segments(file)
        f, names = new_window_features(segments, config)
        feats.append(f); labels.append(y); trials.append(t)
        subjects.extend([file.stem] * len(segments)); groups.extend(['DEP' if file.parent.name == '抑郁症患者' else 'HC'] * len(segments))
        print(f'features {index}/{len(train_files)} {file.stem}', flush=True)
    pub, pub_users, pub_trials = [], [], []
    for file in sorted((ROOT / '数据' / '公开测试集').glob('*.mat'), key=lambda p: int(p.stem.replace('P_test', ''))):
        f, _ = new_window_features(load_public_test_segments(file), config)
        pub.append(f); pub_users.extend([file.stem] * len(f)); pub_trials.extend(np.arange(1, 9))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X_train=np.concatenate(feats), y_train=np.concatenate(labels), subject_ids=np.asarray(subjects), health_groups=np.asarray(groups),
                        original_trial_ids=np.concatenate(trials), feature_names=np.asarray(names), X_public=np.concatenate(pub), public_user_ids=np.asarray(pub_users), public_trial_ids=np.asarray(pub_trials))


def assert_alignment(a, b):
    for key in ALIGN_KEYS:
        if not np.array_equal(a[key], b[key]):
            raise RuntimeError(f'cache alignment mismatch: {key}')


def load_data(config):
    from emo_rec.normalization import normalize_by_subject
    spectral = base.load_cache((HERE / config['spectral_cache']).resolve())
    temporal = base.load_cache((HERE / config['temporal_cache']).resolve())
    base.assert_cache_alignment(spectral, temporal)
    cache_path = HERE / config['new_feature_cache']
    if not cache_path.exists():
        build_feature_cache(config, cache_path)
    new = base.load_cache(cache_path)
    assert_alignment(spectral, new)
    keep = spectral['subject_ids'] != config['excluded_subject']
    subjects = spectral['subject_ids'][keep]; pub_ids = spectral['public_user_ids']
    fn = spectral['feature_names']
    sm = np.asarray([n.startswith('log_bandpower_') for n in fn]); am = np.asarray([n.startswith('asymmetry_') for n in fn])
    assert sm.sum() == 150 and am.sum() == 60 and temporal['X_train'].shape[1] == 450 and new['X_train'].shape[1] == 210
    z = lambda M, ids: normalize_by_subject(M, ids, 'zscore')
    X = np.column_stack([z(spectral['X_train'][keep][:, sm], subjects), z(temporal['X_train'][keep], subjects), z(new['X_train'][keep], subjects), z(spectral['X_train'][keep][:, am], subjects)])
    XP = np.column_stack([z(spectral['X_public'][:, sm], pub_ids), z(temporal['X_public'], pub_ids), z(new['X_public'], pub_ids), z(spectral['X_public'][:, am], pub_ids)])
    new_names = list(new['feature_names']) + [str(n) for n in fn[am]]
    ref = pd.read_csv((HERE / config['fixed_reference']).resolve())
    assert np.array_equal(ref.subject_id.to_numpy(), subjects) and np.array_equal(ref.y_true.to_numpy(), spectral['y_train'][keep]) and np.array_equal(ref.original_trial_id.to_numpy(), spectral['original_trial_ids'][keep])
    ref = ref[['subject_id', 'health_group', 'original_trial_id', 'fold', 'y_true']]
    assert X.shape == (2760, 870) and XP.shape == (80, 870) and np.isfinite(X).all() and np.isfinite(XP).all()
    return X, XP, ref, spectral, new_names, new


def column_sets():
    c, abc, d = np.arange(600), np.arange(600, 810), np.arange(810, 870)
    return {'control_600': c, 'plus_ABC': np.r_[c, abc], 'plus_D': np.r_[c, d], 'plus_all': np.r_[c, abc, d], 'new_only_ABCD': np.r_[abc, d]}


def fit_predict(X, y, train, X_eval, config, seed):
    out = {}
    with threadpool_limits(limits=1):
        for m, cols in column_sets().items():
            fitted = base.make_model(config, seed); fitted.fit(X[train][:, cols], y[train]); out[m] = fitted.predict_proba(X_eval[:, cols])[:, 1]
    return out


def job(X, y, split_name, folds, fold, config):
    train, valid = folds != fold, folds == fold
    return split_name, fold, fit_predict(X, y, train, X[valid], config, config['outer_model_seed_base'] + fold)


def summarize(y, p, ref, folds):
    result = base.summarize(y, p, ref, folds)
    result['worst_subject_auc'] = float(np.min([r['auc'] for r in result['subjects']]))
    return result


def scalars(result):
    return {**{k: result['overall'][k] for k in KEYS[:3]}, 'subject_auc_mean': result['subject_auc_mean'], 'hc_auc': result['health_groups']['HC']['pooled']['auc'], 'dep_auc': result['health_groups']['DEP']['pooled']['auc'], 'worst_subject_auc': result['worst_subject_auc']}


def within_subject_effects(Xnew, y, subjects, groups, names):
    """Per-subject Cohen d (pos minus neg / pooled sd); descriptive P2 diagnostic only."""
    subs = np.unique(subjects); d = np.zeros((len(subs), Xnew.shape[1])); g = []
    for i, s in enumerate(subs):
        m = subjects == s; a, b = Xnew[m & (y == 1)], Xnew[m & (y == 0)]
        sd = np.sqrt(((len(a) - 1) * a.var(axis=0, ddof=1) + (len(b) - 1) * b.var(axis=0, ddof=1)) / (len(a) + len(b) - 2))
        d[i] = (a.mean(axis=0) - b.mean(axis=0)) / np.maximum(sd, 1e-12); g.append(groups[m][0])
    g = np.asarray(g); sign = np.sign(d.mean(axis=0))
    table = pd.DataFrame({'feature': names, 'family': [family_of(n) for n in names], 'mean_d_all': d.mean(axis=0), 'abs_d_all': np.abs(d).mean(axis=0),
                          'abs_d_HC': np.abs(d[g == 'HC']).mean(axis=0), 'abs_d_DEP': np.abs(d[g == 'DEP']).mean(axis=0),
                          'consistency_all': (np.sign(d) == sign).mean(axis=0), 'consistency_HC': (np.sign(d[g == 'HC']) == sign).mean(axis=0), 'consistency_DEP': (np.sign(d[g == 'DEP']) == sign).mean(axis=0)})
    fam = table.groupby('family')[['abs_d_all', 'abs_d_HC', 'abs_d_DEP', 'consistency_all', 'consistency_HC', 'consistency_DEP']].mean().reindex(list(FAMILIES))
    return table, fam


class Tee:
    def __init__(self, path): self.file = open(path, 'w', encoding='utf-8'); self.stdout = sys.stdout
    def write(self, s): self.stdout.write(s); self.file.write(s); self.file.flush()
    def flush(self): self.stdout.flush(); self.file.flush()


def main():
    sys.stdout = Tee(HERE / 'run_output.txt')
    started = time.perf_counter(); config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    (HERE / 'predictions').mkdir(exist_ok=True); (HERE / 'input_metadata').mkdir(exist_ok=True)
    X, XP, ref, spectral, new_names, new = load_data(config); y = ref.y_true.to_numpy(); subjects = ref.subject_id.to_numpy(); groups = ref.health_group.to_numpy()
    print(f'data ready X={X.shape} XP={XP.shape} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    provenance = {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'inputs': {k: {'path': str((HERE / config[k]).resolve()), 'sha256': base.sha256((HERE / config[k]).resolve())} for k in ('spectral_cache', 'temporal_cache', 'fixed_reference', 'new_feature_cache')},
                  'config_sha256': base.sha256(HERE / 'config.json'), 'train_shape': list(X.shape), 'public_shape': list(XP.shape), 'excluded': config['excluded_subject'], 'unlabeled_subject_zscore': True,
                  'column_layout': {'control_600': [0, 600], 'ABC_new_210': [600, 810], 'D_asymmetry_60': [810, 870]}, 'method_dims': {m: int(len(c)) for m, c in column_sets().items()}, 'new_feature_names': new_names, 'alignment_checked': list(ALIGN_KEYS)}
    (HERE / 'input_metadata' / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    splits = [('fixed002', ref.fold.to_numpy())] + [(f'repeat{seed}', base.make_subject_folds(subjects, groups, seed)) for seed in config['repeat_seeds']]
    for _, folds in splits: base.assert_subject_isolation(subjects, folds)
    sys.path.insert(0, str(HERE)); import run_experiment as importable  # pickle job by reference for loky workers
    tasks = [delayed(importable.job)(X, y, name, folds, fold, config) for name, folds in splits for fold in range(5)]
    results = Parallel(n_jobs=config['n_jobs'])(tasks)
    print(f'outer folds done elapsed={time.perf_counter()-started:.1f}s', flush=True)
    preds = {name: {m: np.full(len(y), np.nan) for m in METHODS} for name, _ in splits}; split_folds = dict(splits)
    for name, fold, out in results:
        valid = split_folds[name] == fold
        for m in METHODS: preds[name][m][valid] = out[m]
    runs = {m: {} for m in METHODS}; rows = []; control_check = {}
    for name, folds in splits:
        ref_split = ref.copy(); ref_split['fold'] = folds
        for m in METHODS:
            p = preds[name][m]; assert np.isfinite(p).all() and len(p) == 2760
            out = ref_split.copy(); out['prob_positive'] = p; out.to_csv(HERE / 'predictions' / f'{m}_{name}_oof.csv', index=False)
            result = summarize(y, p, ref_split, folds); runs[m][name] = result; rows.append({'method': m, 'split': name, **scalars(result)})
        old = (HERE / config['control_reference_dir']).resolve() / f'spectral_temporal_hgb_control_{name}_oof.csv'
        if old.exists(): control_check[name] = float(np.max(np.abs(pd.read_csv(old).prob_positive.to_numpy() - preds[name]['control_600'])))
    public = fit_predict(X, y, np.ones(len(y), bool), XP, config, config['outer_model_seed_base'])
    for m in METHODS:
        frame = pd.DataFrame({'user_id': spectral['public_user_ids'], 'trial_id': spectral['public_trial_ids'], 'prob. of positive emotion': public[m]})
        assert len(frame) == 80 and np.isfinite(frame.iloc[:, 2]).all(); frame.to_csv(HERE / 'predictions' / f'{m}_public.csv', index=False)
    aggregate = {}
    for m in METHODS:
        vals = [scalars(runs[m][f'repeat{s}']) for s in config['repeat_seeds']]
        aggregate[m] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
        rows.append({'method': m, 'split': 'repeat_mean', **{k: aggregate[m][k]['mean'] for k in KEYS}})
    deltas, decisions = {}, {}
    for m in METHODS[1:]:
        deltas[m] = {'fixed002': {k: scalars(runs[m]['fixed002'])[k] - scalars(runs['control_600']['fixed002'])[k] for k in KEYS},
                     'repeat_mean': {k: aggregate[m][k]['mean'] - aggregate['control_600'][k]['mean'] for k in KEYS},
                     'repeat_paired_auc': [aggregate[m]['auc']['values'][i] - aggregate['control_600']['auc']['values'][i] for i in range(len(config['repeat_seeds']))]}
        d = deltas[m]['repeat_mean']; decisions[m] = bool(d['auc'] >= 0.005 and d['f1'] >= 0 and d['dep_auc'] >= 0 and deltas[m]['fixed002']['auc'] > 0)
    effects, families = within_subject_effects(X[:, 600:870], y, subjects, groups, new_names)
    effects.sort_values('abs_d_all', ascending=False).to_csv(HERE / 'p2_feature_effects.csv', index=False)
    elapsed = time.perf_counter() - started
    metrics = {'config': config, 'provenance': provenance, 'runs': runs, 'aggregate': aggregate, 'deltas_vs_control': deltas, 'decision_accept': decisions, 'control_reproduction_max_abs_diff_vs_052': control_check,
               'p2_family_effects': families.reset_index().to_dict(orient='records'), 'p2_top10_features': effects.sort_values('abs_d_all', ascending=False).head(10).to_dict(orient='records'), 'elapsed_seconds': elapsed}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    table = pd.DataFrame(rows); table.to_csv(HERE / 'summary.csv', index=False)
    write_summary(metrics, table)
    print(table.to_string(index=False)); print(families.to_string()); print(json.dumps({'deltas_repeat_mean': {m: deltas[m]['repeat_mean'] for m in deltas}, 'fixed002_auc_delta': {m: deltas[m]['fixed002']['auc'] for m in deltas}, 'decision': decisions, 'control_check': control_check, 'elapsed_seconds': elapsed}, indent=1), flush=True)


def write_summary(metrics, table):
    d = metrics['deltas_vs_control']; dec = metrics['decision_accept']; agg = metrics['aggregate']
    lines = ['# 实验055：非周期(1/f)、谱熵、Hjorth与不对称特征（新信息源）', '',
             '数据与协议同052/054：001频谱log_bandpower 150维 + 009时序450维（control_600），排除HC43，所有特征块逐被试/user无标签z-score；fixed002五折 + seeds 2027..2031重复，HGB参数同base.make_model，训练种子2052+fold，阈值0.5，joblib 10进程、每进程threadpool_limits(1)。公开集仅输出预测，不计算任何指标。', '',
             '新特征（每10秒窗、每通道，原始EEG经emo_rec.data读取，样本顺序已与001/009缓存逐项断言对齐）：',
             '- A：Welch(nperseg=500, noverlap=250) PSD；在2–40 Hz且排除7–14 Hz频点上最小二乘拟合log10(PSD)~log10(f)，exponent=−slope、offset（60维）；alpha峰频=7–14 Hz内去除非周期拟合后log10残差最大频率（30维），alpha峰残差功率=该残差（30维）。',
             '- B：1–45 Hz归一化PSD的Shannon熵，除以ln(频点数)归一化到[0,1]（30维）。',
             '- C：Hjorth mobility/complexity（60维），先1–45 Hz 4阶Butterworth零相位带通，整窗计算。',
             '- D：001缓存asymmetry_* 60维（12对×5频带，左减右log功率）。', '',
             '方法：control_600；plus_ABC（810维）；plus_D（660维）；plus_all（870维）；new_only_ABCD（270维，仅用于判断信息量）。', '',
             '## 预注册接受线（事先确定，不事后调参）', '', '相对control_600：五重复均值ΔAUC ≥ +0.005 且 ΔF1 ≥ 0 且 ΔDEP AUC ≥ 0，且fixed002 ΔAUC > 0；否则停止，不追加调参。', '',
             '## 结果', '', '|方法|分割|AUC|F1|BAcc|subject宏AUC|HC AUC|DEP AUC|最差被试AUC|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    lines += ['|' + r['method'] + '|' + r['split'] + '|' + '|'.join(f'{r[k]:.4f}' for k in KEYS) + '|' for _, r in table.iterrows()]
    lines += ['', '五重复均值±SD：', '', '|方法|AUC|F1|DEP AUC|', '|---|---:|---:|---:|']
    lines += [f'|{m}|{agg[m]["auc"]["mean"]:.4f}±{agg[m]["auc"]["std"]:.4f}|{agg[m]["f1"]["mean"]:.4f}±{agg[m]["f1"]["std"]:.4f}|{agg[m]["dep_auc"]["mean"]:.4f}±{agg[m]["dep_auc"]["std"]:.4f}|' for m in agg]
    lines += ['', '## 相对control_600', '', '|方法|ΔAUC|ΔF1|ΔDEP AUC|ΔHC AUC|Δ最差被试|fixed002 ΔAUC|逐重复ΔAUC|达线|', '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for m in d:
        r = d[m]['repeat_mean']
        lines.append(f'|{m}|{r["auc"]:+.4f}|{r["f1"]:+.4f}|{r["dep_auc"]:+.4f}|{r["hc_auc"]:+.4f}|{r["worst_subject_auc"]:+.4f}|{d[m]["fixed002"]["auc"]:+.4f}|' + ', '.join(f'{v:+.4f}' for v in d[m]['repeat_paired_auc']) + f'|{"是" if dec[m] else "否"}|')
    cc = metrics['control_reproduction_max_abs_diff_vs_052']
    lines += ['', 'control复现检查（与052 control OOF最大绝对差）：' + ', '.join(f'{k}={v:.2e}' for k, v in cc.items()), '',
              '## P2诊断：被试内标签效应（描述性，使用全部训练标签，不参与建模）', '',
              '每被试内 d=(正类均值−负类均值)/合并sd；|d|按HC/DEP组平均；方向一致率=被试d符号与全体平均d符号一致的比例。', '',
              '|特征族|mean|d| 全体|mean|d| HC|mean|d| DEP|一致率 全体|一致率 HC|一致率 DEP|', '|---|---:|---:|---:|---:|---:|---:|']
    lines += [f'|{r["family"]}|{r["abs_d_all"]:.3f}|{r["abs_d_HC"]:.3f}|{r["abs_d_DEP"]:.3f}|{r["consistency_all"]:.3f}|{r["consistency_HC"]:.3f}|{r["consistency_DEP"]:.3f}|' for r in metrics['p2_family_effects']]
    lines += ['', 'Top-10 特征（按全体mean|d|，全表见p2_feature_effects.csv）：', '', '|特征|族|mean d|mean|d||HC|d||DEP|d||一致率|HC一致率|DEP一致率|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f'|{r["feature"]}|{r["family"]}|{r["mean_d_all"]:+.3f}|{r["abs_d_all"]:.3f}|{r["abs_d_HC"]:.3f}|{r["abs_d_DEP"]:.3f}|{r["consistency_all"]:.3f}|{r["consistency_HC"]:.3f}|{r["consistency_DEP"]:.3f}|' for r in metrics['p2_top10_features']]
    accepted = [m for m in dec if dec[m]]
    lines += ['', '注：每被试每类仅20窗，|d|在零效应下也有正偏（抽样噪声约0.3量级），一致率0.5为随机水平；应主要看一致率与组间差异。', '',
              '## 结论', '', ('达线方法：' + '、'.join(accepted) + '。') if accepted else '所有方法均未达预注册接受线，按协议停止，不追加调参。', '',
              f'真实运行耗时：{metrics["elapsed_seconds"]:.1f}秒。']
    (HERE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
