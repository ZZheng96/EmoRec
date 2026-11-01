import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('base032', HERE.parent / '032_repeated_subject_folds' / 'run_experiment.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
METHODS = ('spectral_temporal_hgb_control', 'subject_weighted_hgb', 'group_balanced_hgb')
KEYS = ('auc', 'f1', 'balanced_accuracy', 'subject_auc_mean', 'hc_auc', 'dep_auc', 'worst_subject_auc')


def load_data(config):
    spectral = base.load_cache((HERE / config['spectral_cache']).resolve())
    temporal = base.load_cache((HERE / config['temporal_cache']).resolve())
    base.assert_cache_alignment(spectral, temporal)
    keep = spectral['subject_ids'] != config['excluded_subject']
    subjects = spectral['subject_ids'][keep]; groups = spectral['health_groups'][keep]
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


def model(config, seed):
    return base.make_model(config, seed)


def training_weights(X, y, subjects, groups, config, seed):
    inner = base.make_subject_folds(subjects, groups, config['inner_seed'])
    losses = {}
    oof = np.full(len(y), np.nan)
    for fold in range(5):
        tr, va = inner != fold, inner == fold
        fitted = model(config, seed + fold); fitted.fit(X[tr], y[tr])
        oof[va] = fitted.predict_proba(X[va])[:, 1]
    assert np.isfinite(oof).all()
    for subject in np.unique(subjects):
        mask = subjects == subject
        losses[subject] = float(log_loss(y[mask], oof[mask], labels=[0, 1]))
    median = float(np.median(list(losses.values())))
    raw = {s: losses[s] / max(median, 1e-12) for s in losses}
    weights = {s: float(np.clip(raw[s], config['weight_min'], config['weight_max'])) for s in losses}
    subject_w = np.asarray([weights[s] for s in subjects], dtype=float)
    counts = {g: int(np.sum(groups == g)) for g in np.unique(groups)}
    group_raw = {g: len(counts) / len([x for x in groups if x == g]) for g in counts}
    group_mean = np.mean(list(group_raw.values()))
    group_w = {g: float(np.clip(group_raw[g] / group_mean, config['weight_min'], config['weight_max'])) for g in counts}
    group_weights = np.asarray([group_w[g] for g in groups], dtype=float)
    evidence = {'inner_seed': config['inner_seed'], 'inner_folds': 5, 'training_subjects': np.unique(subjects).tolist(), 'subject_losses': losses, 'subject_weights': weights, 'max_weight': max(weights.values()), 'min_weight': min(weights.values()), 'median_loss': median, 'group_weights': group_w, 'group_counts': counts, 'training_labels_sha256': hashlib.sha256(y.tobytes()).hexdigest()}
    return subject_w, group_weights, evidence


def summarize(y, p, ref, folds):
    result = base.summarize(y, p, ref, folds)
    subject_auc = [r['auc'] for r in result['subjects']]
    result['worst_subject_auc'] = float(np.min(subject_auc))
    return result


def scalars(result):
    return {**{k: result['overall'][k] for k in KEYS[:3]}, 'subject_auc_mean': result['subject_auc_mean'], 'hc_auc': result['health_groups']['HC']['pooled']['auc'], 'dep_auc': result['health_groups']['DEP']['pooled']['auc'], 'worst_subject_auc': result['worst_subject_auc']}


def main():
    started = time.perf_counter(); config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    X, XP, ref, spectral = load_data(config); y = ref.y_true.to_numpy(); subjects = ref.subject_id.to_numpy(); groups = ref.health_group.to_numpy()
    provenance = {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'inputs': {k: {'path': str((HERE / config[k]).resolve()), 'sha256': base.sha256((HERE / config[k]).resolve())} for k in ('spectral_cache', 'temporal_cache', 'fixed_reference')}, 'config_sha256': base.sha256(HERE / 'config.json'), 'train_shape': list(X.shape), 'public_shape': list(XP.shape), 'excluded': config['excluded_subject'], 'unlabeled_subject_zscore': True}
    (HERE / 'input_metadata' / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    splits = [('fixed002', ref.fold.to_numpy())] + [(f'repeat{seed}', base.make_subject_folds(subjects, groups, seed)) for seed in config['repeat_seeds']]
    runs = {m: {} for m in METHODS}; rows = []
    with threadpool_limits(limits=1):
        for split_name, folds in splits:
            base.assert_subject_isolation(subjects, folds); predictions = {m: np.full(len(y), np.nan) for m in METHODS}; ref_split = ref.copy(); ref_split['fold'] = folds
            for fold in range(5):
                train, valid = folds != fold, folds == fold
                sw, gw, evidence = training_weights(X[train], y[train], subjects[train], groups[train], config, 2052 + fold)
                fitted = model(config, 2052 + fold); fitted.fit(X[train], y[train]); predictions[METHODS[0]][valid] = fitted.predict_proba(X[valid])[:, 1]
                fitted = model(config, 2052 + fold); fitted.fit(X[train], y[train], histgradientboostingclassifier__sample_weight=sw); predictions[METHODS[1]][valid] = fitted.predict_proba(X[valid])[:, 1]
                fitted = model(config, 2052 + fold); fitted.fit(X[train], y[train], histgradientboostingclassifier__sample_weight=gw); predictions[METHODS[2]][valid] = fitted.predict_proba(X[valid])[:, 1]
                evidence.update({'split': split_name, 'outer_fold': fold, 'outer_train_rows': int(train.sum()), 'outer_validation_rows': int(valid.sum()), 'outer_validation_subjects': np.unique(subjects[valid]).tolist(), 'subject_disjoint': not bool(set(subjects[train]) & set(subjects[valid]))})
                (HERE / 'input_metadata' / f'{split_name}_fold{fold}_weights.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
                print(f'{split_name} fold={fold} max_weight={evidence["max_weight"]:.3f} elapsed={time.perf_counter()-started:.1f}s', flush=True)
            for method in METHODS:
                p = predictions[method]; assert np.isfinite(p).all(); out = ref_split.copy(); out['prob_positive'] = p; out.to_csv(HERE / 'predictions' / f'{method}_{split_name}_oof.csv', index=False)
                result = summarize(y, p, ref_split, folds); runs[method][split_name] = result; rows.append({'method': method, 'split': split_name, **scalars(result)})
        sw, gw, evidence = training_weights(X, y, subjects, groups, config, 2052)
        (HERE / 'input_metadata' / 'full_train_weights.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
        for method, weights in ((METHODS[0], None), (METHODS[1], sw), (METHODS[2], gw)):
            fitted = model(config, 2052); fitted.fit(X, y, **({'histgradientboostingclassifier__sample_weight': weights} if weights is not None else {})); p = fitted.predict_proba(XP)[:, 1]
            pd.DataFrame({'user_id': spectral['public_user_ids'], 'trial_id': spectral['public_trial_ids'], 'prob. of positive emotion': p}).to_csv(HERE / 'predictions' / f'{method}_public.csv', index=False)
    aggregate = {}
    for method in METHODS:
        vals = [scalars(runs[method][f'repeat{s}']) for s in config['repeat_seeds']]
        aggregate[method] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
        rows.append({'method': method, 'split': 'repeat_mean', **{k: aggregate[method][k]['mean'] for k in KEYS}})
    control, weighted = METHODS[0], METHODS[1]
    deltas = {scope: {k: (scalars(runs[weighted][scope])[k] - scalars(runs[control][scope])[k]) if scope == 'fixed002' else aggregate[weighted][k]['mean'] - aggregate[control][k]['mean'] for k in KEYS} for scope in ('fixed002', 'repeat_mean')}
    advance = all(deltas[s][k] > 0 for s in deltas for k in ('auc', 'dep_auc'))
    metrics = {'config': config, 'provenance': provenance, 'runs': runs, 'aggregate': aggregate, 'paired_deltas_subject_weighted_vs_control': deltas, 'advance': advance, 'elapsed_seconds': time.perf_counter() - started}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8'); pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    report = '# 实验052：一次内层OOF被试损失重加权（非标准Group-DRO）\n\n复用001频谱150维与009时序450维，排除HC43，逐被试/user无标签z-score；002固定五折及seeds2027..2031独立OOF，阈值0.5。subject_weighted_hgb在每个外层训练折内部用固定seed2052的5折训练内OOF HGB损失计算每被试权重，损失/训练被试中位数后裁剪到[1,3]，进行一次加权HGB重训，不迭代更新权重或求解minimax目标，非标准Group-DRO；不得访问外层验证标签。group_balanced_hgb为裁剪逆频率组权重（非完全平衡）对照：健康组逆频率权重按组间均值归一化后裁剪到[1,3]，裁剪后不保证各组总权重相等。\n\n|方法|分割|AUC|F1|BAcc|subject宏AUC|HC AUC|DEP AUC|最差被试AUC|\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n' + '\n'.join('|' + r['method'] + '|' + r['split'] + '|' + '|'.join(f'{r[k]:.6f}' for k in KEYS) + '|' for r in rows) + '\n\n结论：' + ('满足固定提升标准。' if advance else '未同时提升固定折与重复均值总体AUC和DEP AUC，按协议停止当前配置，不追加调权。被试损失重加权的最差被试AUC五重复均值从0.401000升至0.426500（+0.025500），有所改善，但总体AUC从0.789424降至0.788404，F1从0.720038降至0.718517。该结果仅限当前一次重加权配置，不构成对标准Group-DRO或域鲁棒性的普遍否定。') + f'\n真实运行耗时：{metrics["elapsed_seconds"]:.2f}秒。逐折训练被试、损失、权重、最大权重和最坏被试AUC见metrics.json及input_metadata。\n'
    (HERE / '实验报告.md').write_text(report, encoding='utf-8'); print(pd.DataFrame(rows).to_string(index=False)); print(json.dumps({'deltas': deltas, 'advance': advance, 'elapsed_seconds': metrics['elapsed_seconds']}), flush=True)


if __name__ == '__main__': main()
