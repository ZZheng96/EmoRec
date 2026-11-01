import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('base032', HERE.parent / '032_repeated_subject_folds' / 'run_experiment.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
METHODS = ('control', 'soft_target_student', 'teacher_confidence_weighting')
KEYS = ('auc', 'f1', 'balanced_accuracy', 'subject_auc_mean', 'hc_auc', 'dep_auc', 'worst_subject_auc')
SW = 'histgradientboostingclassifier__sample_weight'


def load_data(config):
    spectral = base.load_cache((HERE / config['spectral_cache']).resolve())
    temporal = base.load_cache((HERE / config['temporal_cache']).resolve())
    base.assert_cache_alignment(spectral, temporal)
    keep = spectral['subject_ids'] != config['excluded_subject']
    subjects = spectral['subject_ids'][keep]
    sm = np.asarray([n.startswith('log_bandpower_') for n in spectral['feature_names']])
    assert sm.sum() == 150 and temporal['X_train'].shape[1] == 450
    from emo_rec.normalization import normalize_by_subject
    X = np.column_stack([normalize_by_subject(spectral['X_train'][keep][:, sm], subjects, 'zscore'), normalize_by_subject(temporal['X_train'][keep], subjects, 'zscore')])
    XP = np.column_stack([normalize_by_subject(spectral['X_public'][:, sm], spectral['public_user_ids'], 'zscore'), normalize_by_subject(temporal['X_public'], temporal['public_user_ids'], 'zscore')])
    ref = pd.read_csv((HERE / config['fixed_reference']).resolve())
    assert np.array_equal(ref.subject_id.to_numpy(), subjects) and np.array_equal(ref.y_true.to_numpy(), spectral['y_train'][keep])
    ref = ref[['subject_id', 'health_group', 'original_trial_id', 'fold', 'y_true']]
    assert X.shape == (2760, 600) and XP.shape == (80, 600) and np.isfinite(X).all() and np.isfinite(XP).all()
    return X, XP, ref, spectral


def trial_keys(ref):
    keys = (ref.subject_id.astype(str) + '_T' + (ref.original_trial_id + 4 * ref.y_true).astype(str)).to_numpy()
    counts = pd.Series(keys).value_counts()
    assert counts.eq(5).all() and len(counts) == ref.subject_id.nunique() * 8
    return keys


def model(config, seed):
    return base.make_model(config, seed)


def teacher_targets(X, y, subjects, groups, keys, config):
    """Trial-averaged inner-OOF probabilities; uses only the rows passed in (outer-training subjects)."""
    inner = base.make_subject_folds(subjects, groups, config['inner_seed'], config['inner_folds'])
    oof = np.full(len(y), np.nan)
    for fold in range(config['inner_folds']):
        tr, va = inner != fold, inner == fold
        fitted = model(config, config['inner_seed'] + fold); fitted.fit(X[tr], y[tr])
        oof[va] = fitted.predict_proba(X[va])[:, 1]
    assert np.isfinite(oof).all()
    assert pd.Series(keys).value_counts().eq(5).all()
    t = pd.Series(oof).groupby(keys).transform('mean').to_numpy()
    first = ~pd.Series(keys).duplicated().to_numpy()
    evidence = {'inner_seed': config['inner_seed'], 'training_subjects': np.unique(subjects).tolist(), 'teacher_window_auc': float(roc_auc_score(y, oof)), 'teacher_trial_auc': float(roc_auc_score(y[first], t[first])), 'teacher_mean': float(t.mean()), 'training_labels_sha256': hashlib.sha256(y.tobytes()).hexdigest()}
    return t, evidence


def fit_soft(X, soft, config, seed):
    fitted = model(config, seed)
    fitted.fit(np.vstack([X, X]), np.r_[np.ones(len(X), int), np.zeros(len(X), int)], **{SW: np.r_[soft, 1 - soft]})
    return fitted


def confidence_weights(y, t, config):
    w = config['confidence_weight_floor'] + (1 - config['confidence_weight_floor']) * np.where(y == 1, t, 1 - t)
    return w / w.mean()


def fit_predict_fold(X, y, subjects, groups, keys, train, X_eval, config, seed):
    """Fit all methods on train rows only and predict X_eval."""
    with threadpool_limits(limits=1):
        Xt, yt = X[train], y[train]
        t, evidence = teacher_targets(Xt, yt, subjects[train], groups[train], keys[train], config)
        out = {}
        fitted = model(config, seed); fitted.fit(Xt, yt); out['control'] = fitted.predict_proba(X_eval)[:, 1]
        soft = config['soft_alpha'] * yt + (1 - config['soft_alpha']) * t
        out['soft_target_student'] = fit_soft(Xt, soft, config, seed).predict_proba(X_eval)[:, 1]
        w = confidence_weights(yt, t, config)
        fitted = model(config, seed); fitted.fit(Xt, yt, **{SW: w}); out['teacher_confidence_weighting'] = fitted.predict_proba(X_eval)[:, 1]
    evidence.update({'soft_min': float(soft.min()), 'soft_max': float(soft.max()), 'weight_min': float(w.min()), 'weight_max': float(w.max())})
    return out, evidence


def summarize(y, p, ref, folds):
    result = base.summarize(y, p, ref, folds)
    result['worst_subject_auc'] = float(np.min([r['auc'] for r in result['subjects']]))
    return result


def scalars(result):
    return {**{k: result['overall'][k] for k in KEYS[:3]}, 'subject_auc_mean': result['subject_auc_mean'], 'hc_auc': result['health_groups']['HC']['pooled']['auc'], 'dep_auc': result['health_groups']['DEP']['pooled']['auc'], 'worst_subject_auc': result['worst_subject_auc']}


def job(X, y, subjects, groups, keys, split_name, folds, fold, config):
    train, valid = folds != fold, folds == fold
    out, evidence = fit_predict_fold(X, y, subjects, groups, keys, train, X[valid], config, config['outer_model_seed_base'] + fold)
    evidence.update({'split': split_name, 'outer_fold': fold, 'outer_train_rows': int(train.sum()), 'outer_validation_rows': int(valid.sum()), 'outer_validation_subjects': np.unique(subjects[valid]).tolist(), 'subject_disjoint': not bool(set(subjects[train]) & set(subjects[valid]))})
    return split_name, fold, out, evidence


def main():
    started = time.perf_counter(); config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    (HERE / 'predictions').mkdir(exist_ok=True); (HERE / 'input_metadata').mkdir(exist_ok=True)
    X, XP, ref, spectral = load_data(config); y = ref.y_true.to_numpy(); subjects = ref.subject_id.to_numpy(); groups = ref.health_group.to_numpy(); keys = trial_keys(ref)
    provenance = {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'inputs': {k: {'path': str((HERE / config[k]).resolve()), 'sha256': base.sha256((HERE / config[k]).resolve())} for k in ('spectral_cache', 'temporal_cache', 'fixed_reference', 'audit_per_subject_table')}, 'config_sha256': base.sha256(HERE / 'config.json'), 'train_shape': list(X.shape), 'public_shape': list(XP.shape), 'excluded': config['excluded_subject'], 'unlabeled_subject_zscore': True, 'n_trial_keys': int(len(set(keys))), 'trial_id_used_as_predictor': False}
    (HERE / 'input_metadata' / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    splits = [('fixed002', ref.fold.to_numpy())] + [(f'repeat{seed}', base.make_subject_folds(subjects, groups, seed)) for seed in config['repeat_seeds']]
    for _, folds in splits: base.assert_subject_isolation(subjects, folds)
    import sys; sys.path.insert(0, str(HERE)); import run_experiment as importable  # pickle job by reference for loky workers
    tasks = [delayed(importable.job)(X, y, subjects, groups, keys, name, folds, fold, config) for name, folds in splits for fold in range(5)]
    results = Parallel(n_jobs=config['n_jobs'], verbose=5)(tasks)
    print(f'outer folds done elapsed={time.perf_counter()-started:.1f}s', flush=True)
    preds = {name: {m: np.full(len(y), np.nan) for m in METHODS} for name, _ in splits}; fold_evidence = []
    split_folds = dict(splits)
    for name, fold, out, evidence in results:
        valid = split_folds[name] == fold
        for m in METHODS: preds[name][m][valid] = out[m]
        fold_evidence.append(evidence)
    (HERE / 'input_metadata' / 'fold_teacher_evidence.json').write_text(json.dumps(fold_evidence, indent=2), encoding='utf-8')
    runs = {m: {} for m in METHODS}; rows = []; control_check = {}
    for name, folds in splits:
        ref_split = ref.copy(); ref_split['fold'] = folds
        for m in METHODS:
            p = preds[name][m]; assert np.isfinite(p).all() and len(p) == 2760
            out = ref_split.copy(); out['prob_positive'] = p; out.to_csv(HERE / 'predictions' / f'{m}_{name}_oof.csv', index=False)
            result = summarize(y, p, ref_split, folds); runs[m][name] = result; rows.append({'method': m, 'split': name, **scalars(result)})
        old = HERE / config['control_reference_dir'] / f'spectral_temporal_hgb_control_{name}_oof.csv'
        if old.exists():
            control_check[name] = float(np.max(np.abs(pd.read_csv(old).prob_positive.to_numpy() - preds[name]['control'])))
    all_train = np.ones(len(y), bool)
    out, full_evidence = fit_predict_fold(X, y, subjects, groups, keys, all_train, XP, config, config['outer_model_seed_base'])
    (HERE / 'input_metadata' / 'full_train_teacher_evidence.json').write_text(json.dumps(full_evidence, indent=2), encoding='utf-8')
    for m in METHODS:
        pd.DataFrame({'user_id': spectral['public_user_ids'], 'trial_id': spectral['public_trial_ids'], 'prob. of positive emotion': out[m]}).to_csv(HERE / 'predictions' / f'{m}_public.csv', index=False)
    aggregate = {}
    for m in METHODS:
        vals = [scalars(runs[m][f'repeat{s}']) for s in config['repeat_seeds']]
        aggregate[m] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
        rows.append({'method': m, 'split': 'repeat_mean', **{k: aggregate[m][k]['mean'] for k in KEYS}})
    deltas, decisions = {}, {}
    for m in METHODS[1:]:
        deltas[m] = {'fixed002': {k: scalars(runs[m]['fixed002'])[k] - scalars(runs['control']['fixed002'])[k] for k in KEYS},
                     'repeat_mean': {k: aggregate[m][k]['mean'] - aggregate['control'][k]['mean'] for k in KEYS},
                     'repeat_paired_auc': [aggregate[m]['auc']['values'][i] - aggregate['control']['auc']['values'][i] for i in range(len(config['repeat_seeds']))]}
        d = deltas[m]['repeat_mean']; decisions[m] = bool(d['auc'] >= 0.005 and d['f1'] >= 0 and d['dep_auc'] >= 0)
    # P3: per-subject repeat-mean AUC gain vs 053 trial-window AUC gap
    audit = pd.read_csv((HERE / config['audit_per_subject_table']).resolve())
    audit['trial_minus_window'] = audit.trial_auc_mean6 - audit.window_auc_mean6
    p3 = {}; subj_rows = []
    for m in METHODS[1:]:
        gains = {}
        for s in config['repeat_seeds']:
            a = {r['subject_id']: r['auc'] for r in runs[m][f'repeat{s}']['subjects']}; c = {r['subject_id']: r['auc'] for r in runs['control'][f'repeat{s}']['subjects']}
            for subj in a: gains.setdefault(subj, []).append(a[subj] - c[subj])
        frame = pd.DataFrame({'subject_id': list(gains), f'gain_{m}': [float(np.mean(v)) for v in gains.values()]}).merge(audit[['subject_id', 'health_group', 'window_auc_mean6', 'trial_auc_mean6', 'trial_minus_window']], on='subject_id')
        assert len(frame) == 69
        sp = spearmanr(frame.trial_minus_window, frame[f'gain_{m}']); pe = pearsonr(frame.trial_minus_window, frame[f'gain_{m}'])
        hi = frame.trial_minus_window >= frame.trial_minus_window.median()
        p3[m] = {'n_subjects': int(len(frame)), 'spearman_rho': float(sp.statistic), 'spearman_p': float(sp.pvalue), 'pearson_r': float(pe.statistic), 'pearson_p': float(pe.pvalue), 'mean_gain_high_gap_half': float(frame.loc[hi, f'gain_{m}'].mean()), 'mean_gain_low_gap_half': float(frame.loc[~hi, f'gain_{m}'].mean()), 'mean_gain_all': float(frame[f'gain_{m}'].mean())}
        subj_rows.append(frame.set_index('subject_id'))
    per_subject = subj_rows[0].join(subj_rows[1][[f'gain_{METHODS[2]}']]); per_subject.to_csv(HERE / 'per_subject_gain_vs_gap.csv')
    elapsed = time.perf_counter() - started
    metrics = {'config': config, 'provenance': provenance, 'runs': runs, 'aggregate': aggregate, 'deltas_vs_control': deltas, 'decision_accept': decisions, 'control_reproduction_max_abs_diff_vs_052': control_check, 'p3_subject_gain_vs_trial_window_gap': p3, 'full_train_teacher': full_evidence, 'teacher_fold_auc_mean': {'window': float(np.mean([e['teacher_window_auc'] for e in fold_evidence])), 'trial': float(np.mean([e['teacher_trial_auc'] for e in fold_evidence]))}, 'elapsed_seconds': elapsed}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    table = pd.DataFrame(rows)
    write_summary(metrics, table)
    print(table.to_string(index=False)); print(json.dumps({'deltas_repeat_mean': {m: deltas[m]['repeat_mean'] for m in deltas}, 'decision': decisions, 'control_check': control_check, 'p3': p3, 'teacher': metrics['teacher_fold_auc_mean'], 'elapsed_seconds': elapsed}, indent=1), flush=True)


def write_summary(metrics, table):
    d = metrics['deltas_vs_control']; p3 = metrics['p3_subject_gain_vs_trial_window_gap']; dec = metrics['decision_accept']; tf = metrics['teacher_fold_auc_mean']
    lines = ['# 实验054：trial-teacher 蒸馏（检验053理论预测P3）', '',
             '数据与协议同052：001频谱150维+009时序450维，排除HC43，逐被试/user无标签z-score；fixed002五折+seeds 2027..2031重复，阈值0.5，threadpool_limits(1)。',
             'trial键=(subject, original_trial_id+4*y)，每键5窗（已断言）。trial id仅在外层训练被试内部用于平均内层OOF概率构造训练目标，从不作为预测输入，也不触及外层验证标签。',
             '', '方法（预注册，无调参）：',
             '- control：与052 control相同HGB（同种子2052+fold）。',
             '- soft_target_student：外层训练被试内5折被试划分（seed 2054）得内层OOF窗口概率，t=所属trial的OOF均值；软目标0.5*y+0.5*t；同HGB在复制行上训练（标签1权重soft，标签0权重1-soft，等价软标签logloss）。',
             '- teacher_confidence_weighting：硬标签，w=0.25+0.75*(y==1?t:1-t)，归一化为均值1。', '',
             f'teacher质量（30个外层折均值，训练被试内）：窗口OOF AUC {tf["window"]:.4f}，trial均值AUC {tf["trial"]:.4f}。', '',
             '|方法|分割|AUC|F1|BAcc|subject宏AUC|HC AUC|DEP AUC|最差被试AUC|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    lines += ['|' + r['method'] + '|' + r['split'] + '|' + '|'.join(f'{r[k]:.4f}' for k in KEYS) + '|' for _, r in table.iterrows()]
    lines += ['', '## 相对control（五重复均值差）', '', '|方法|ΔAUC|ΔF1|ΔDEP AUC|ΔHC AUC|Δ最差被试|fixed002 ΔAUC|逐重复ΔAUC|接受|', '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for m in d:
        r = d[m]['repeat_mean']
        lines.append(f'|{m}|{r["auc"]:+.4f}|{r["f1"]:+.4f}|{r["dep_auc"]:+.4f}|{r["hc_auc"]:+.4f}|{r["worst_subject_auc"]:+.4f}|{d[m]["fixed002"]["auc"]:+.4f}|' + ', '.join(f'{v:+.4f}' for v in d[m]['repeat_paired_auc']) + f'|{"是" if dec[m] else "否"}|')
    cc = metrics['control_reproduction_max_abs_diff_vs_052']
    lines += ['', f'control复现检查（与052 control OOF最大绝对差）：' + ', '.join(f'{k}={v:.2e}' for k, v in cc.items()), '',
              '## P3检验：被试AUC增益 vs 053 (trial AUC − window AUC)', '', '|方法|Spearman ρ (p)|Pearson r (p)|高gap半数平均增益|低gap半数平均增益|全体平均增益|', '|---|---|---|---:|---:|---:|']
    for m, v in p3.items():
        lines.append(f'|{m}|{v["spearman_rho"]:+.3f} ({v["spearman_p"]:.3f})|{v["pearson_r"]:+.3f} ({v["pearson_p"]:.3f})|{v["mean_gain_high_gap_half"]:+.4f}|{v["mean_gain_low_gap_half"]:+.4f}|{v["mean_gain_all"]:+.4f}|')
    lines += ['', '注：053的trial/window AUC来自同一control家族OOF，与本实验增益非独立，相关仅作描述性证据；逐被试明细见per_subject_gain_vs_gap.csv。', '',
              '## 决策（预注册）', '', '接受条件：五重复均值ΔAUC ≥ +0.005 且 ΔF1 ≥ 0 且 ΔDEP AUC ≥ 0。',
              ('结论：' + '；'.join(f'{m}{"接受" if dec[m] else "未达标"}' for m in dec) + '。' + ('' if any(dec.values()) else '两种蒸馏均未达标，按协议停止，不调参。')),
              '', f'真实运行耗时：{metrics["elapsed_seconds"]:.1f}秒。']
    (HERE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
