import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('repeat032', HERE.parent / '032_repeated_subject_folds/run_experiment.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
METHODS = ('spectral_temporal_control', 'stable_selected')
KEYS = ('auc', 'f1', 'balanced_accuracy', 'subject_auc_mean', 'hc_auc', 'dep_auc')


def load_data(config):
    spectral = base.load_cache(HERE / config['spectral_cache'])
    temporal = base.load_cache(HERE / config['temporal_cache'])
    base.assert_cache_alignment(spectral, temporal)
    keep = spectral['subject_ids'] != 'HC43'
    subjects = spectral['subject_ids'][keep]
    spectral_mask = np.asarray([n.startswith('log_bandpower_') for n in spectral['feature_names']])
    assert spectral_mask.sum() == 150 and temporal['X_train'].shape[1] == 450
    X = np.column_stack([normalize_by_subject(spectral['X_train'][keep][:, spectral_mask], subjects, 'zscore'), normalize_by_subject(temporal['X_train'][keep], subjects, 'zscore')])
    XP = np.column_stack([normalize_by_subject(spectral['X_public'][:, spectral_mask], spectral['public_user_ids'], 'zscore'), normalize_by_subject(temporal['X_public'], temporal['public_user_ids'], 'zscore')])
    names = np.concatenate([spectral['feature_names'][spectral_mask], temporal['feature_names']])
    reference = pd.read_csv(HERE / config['fixed_reference'])
    for column, values in [('subject_id', subjects), ('health_group', spectral['health_groups'][keep]), ('original_trial_id', spectral['original_trial_ids'][keep]), ('y_true', spectral['y_train'][keep])]:
        assert np.array_equal(reference[column].to_numpy(), values), column
    reference = reference[['subject_id', 'health_group', 'original_trial_id', 'fold', 'y_true']]
    assert X.shape == (2760, 600) and XP.shape == (80, 600) and len(np.unique(subjects)) == 69
    assert np.isfinite(X).all() and np.isfinite(XP).all()
    return X, XP, reference, names, spectral


def select_stable(X, y, subjects, groups, names, config):
    folds = base.make_subject_folds(subjects, groups, config['inner_seed'])
    counts = np.zeros(X.shape[1], dtype=int)
    strengths = np.zeros(X.shape[1])
    evidence = []
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        train_subjects = np.unique(subjects[train]).tolist()
        valid_subjects = np.unique(subjects[valid]).tolist()
        assert not set(train_subjects) & set(valid_subjects)
        model = make_pipeline(StandardScaler(), LogisticRegression(**config['selector']))
        model.fit(X[train], y[train])
        weights = np.abs(model[-1].coef_[0])
        top = np.lexsort((np.arange(len(weights)), -weights))[:config['inner_top_k']]
        counts[top] += 1
        strengths += weights / 5
        evidence.append({'inner_fold': fold, 'training_subjects': train_subjects, 'held_out_subjects': valid_subjects, 'fit_rows': int(train.sum()), 'selected_indices': top.tolist(), 'training_labels_sha256': __import__('hashlib').sha256(y[train].tobytes()).hexdigest()})
    frequency = counts / 5
    candidates = np.flatnonzero(frequency >= config['frequency_threshold'])
    ranking = np.lexsort((candidates, -strengths[candidates], -frequency[candidates]))
    selected = np.sort(candidates[ranking[:config['max_selected']]])
    if not len(selected):
        raise RuntimeError('No stable features; preregistered abort, no fallback tuning')
    return selected, {'training_subjects': np.unique(subjects).tolist(), 'inner_folds': evidence, 'selected_indices': selected.tolist(), 'selected_names': names[selected].tolist(), 'feature_names': names.tolist(), 'frequencies': frequency.tolist(), 'mean_absolute_coefficients': strengths.tolist()}


def predict_outer(X, y, subjects, groups, names, train, valid, config, seed):
    selected, evidence = select_stable(X[train], y[train], subjects[train], groups[train], names, config)
    evidence['outer_validation_subjects'] = np.unique(subjects[valid]).tolist()
    assert not set(evidence['training_subjects']) & set(evidence['outer_validation_subjects'])
    predictions = {}
    for method, indices in [(METHODS[0], np.arange(X.shape[1])), (METHODS[1], selected)]:
        model = base.make_model(config, seed)
        model.fit(X[train][:, indices], y[train])
        predictions[method] = model.predict_proba(X[valid][:, indices])[:, 1]
    return predictions, evidence


def scalars(result):
    return {**{k: result['overall'][k] for k in KEYS[:3]}, 'subject_auc_mean': result['subject_auc_mean'], 'hc_auc': result['health_groups']['HC']['pooled']['auc'], 'dep_auc': result['health_groups']['DEP']['pooled']['auc']}


def public_frame(spectral, probabilities):
    frame = pd.DataFrame({'user_id': spectral['public_user_ids'], 'trial_id': spectral['public_trial_ids'], 'prob. of positive emotion': probabilities})
    validate_submission(frame, [f'P_test{i}' for i in range(1, 11)])
    return frame


def main():
    started = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    for directory in ('predictions', 'input_metadata'):
        (HERE / directory).mkdir(exist_ok=True)
    X, XP, reference, names, spectral = load_data(config)
    subjects, groups, y = (reference[c].to_numpy() for c in ('subject_id', 'health_group', 'y_true'))
    provenance = {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'inputs': {k: {'path': str((HERE / config[k]).resolve()), 'sha256': base.sha256(HERE / config[k])} for k in ('spectral_cache', 'temporal_cache', 'fixed_reference')}, 'config_sha256': base.sha256(HERE / 'config.json'), 'feature_names': names.tolist(), 'train_shape': list(X.shape), 'public_shape': list(XP.shape), 'excluded': 'HC43', 'unlabeled_subject_zscore': True}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    splits = [('fixed002', reference.fold.to_numpy())] + [(f'repeat{seed}', base.make_subject_folds(subjects, groups, seed)) for seed in config['repeat_seeds']]
    runs = {method: {} for method in METHODS}
    rows = []
    with threadpool_limits(limits=1):
        for split_name, folds in splits:
            base.assert_subject_isolation(subjects, folds)
            predictions = {method: np.full(len(y), np.nan) for method in METHODS}
            ref = reference.copy(); ref['fold'] = folds
            ref.drop(columns='y_true').to_csv(HERE / f'input_metadata/{split_name}_folds.csv', index=False)
            for fold in range(5):
                train, valid = folds != fold, folds == fold
                pred, evidence = predict_outer(X, y, subjects, groups, names, train, valid, config, 2026 + fold)
                evidence.update({'split': split_name, 'outer_fold': fold, 'outer_train_rows': int(train.sum()), 'outer_validation_rows': int(valid.sum()), 'outer_isolation_verified': True})
                (HERE / f'input_metadata/{split_name}_fold{fold}_selection.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
                for method in METHODS:
                    predictions[method][valid] = pred[method]
                print(f'{split_name} fold={fold} selected={len(evidence["selected_indices"])} elapsed={time.perf_counter()-started:.1f}s', flush=True)
            for method in METHODS:
                p = predictions[method]
                assert len(p) == 2760 and np.isfinite(p).all()
                output = ref.copy(); output['prob_positive'] = p
                output.to_csv(HERE / f'predictions/{method}_{split_name}_oof.csv', index=False)
                result = base.summarize(y, p, ref, folds)
                runs[method][split_name] = result
                rows.append({'method': method, 'split': split_name, **scalars(result)})
        selected, evidence = select_stable(X, y, subjects, groups, names, config)
        evidence['outer_validation_subjects'] = []
        (HERE / 'input_metadata/full_train_selection.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
        for method, indices in [(METHODS[0], np.arange(600)), (METHODS[1], selected)]:
            model = base.make_model(config, 2026); model.fit(X[:, indices], y)
            public_frame(spectral, model.predict_proba(XP[:, indices])[:, 1]).to_csv(HERE / f'predictions/{method}_public.csv', index=False)
    aggregate = {}
    for method in METHODS:
        vals = [scalars(runs[method][f'repeat{seed}']) for seed in config['repeat_seeds']]
        aggregate[method] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
        rows.append({'method': method, 'split': 'repeat_mean', **{k: aggregate[method][k]['mean'] for k in KEYS}})
    deltas = {scope: {k: (scalars(runs[METHODS[1]]['fixed002'])[k] - scalars(runs[METHODS[0]]['fixed002'])[k]) if scope == 'fixed002' else aggregate[METHODS[1]][k]['mean'] - aggregate[METHODS[0]][k]['mean'] for k in KEYS} for scope in ('fixed002', 'repeat_mean')}
    advance = all(deltas[s][k] > 0 for s in deltas for k in ('auc', 'dep_auc'))
    metrics = {'config': config, 'provenance': provenance, 'runs': runs, 'aggregate': aggregate, 'paired_deltas': deltas, 'advance': advance, 'elapsed_seconds': time.perf_counter() - started}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    report = '# 实验051：训练折内跨被试稳定特征选择\n\n001频谱150维+009时序450维；排除HC43；逐被试/user无标签z-score。002固定折直接读取已保存分配，重复折按032在69人中生成seed2027..2031，每次2760行独立OOF，不拼接。阈值0.5。\n\n内层训练被试按健康组固定seed2051五折；每次仅内层训练数据拟合StandardScaler+L2 LogisticRegression(C=1)，取绝对系数前200。频率≥0.6，再按频率/平均绝对系数/特征索引排序最多200维；空集则终止。内层保留折不参与拟合；外层验证仅transform/predict。选择与600维对照使用相同固定HGB。无行序、trial ID、邻窗或公开标签输入；ID只用于对齐与审计。预注册配置哈希已保存，未按外层结果调参。\n\n|条件|分割|AUC|F1|BAcc|subject宏AUC|HC AUC|DEP AUC|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for row in rows:
        report += '|' + row['method'] + '|' + row['split'] + '|' + '|'.join(f'{row[k]:.6f}' for k in KEYS) + '|\n'
    report += '\n重复均值的标准差、每折/被试指标详见metrics.json；每个外层折的特征频率、名称、训练被试与内外层隔离证据详见input_metadata。公开预测为全训练集内层重新筛选后一次拟合，不使用公开标签。\n\n结论：' + ('满足预注册提升标准；仍需独立验证。' if advance else '未满足固定折与重复均值的总体/DEP AUC同时提升标准，停止该方向，不追加调参。') + f'\n\n真实运行耗时：{metrics["elapsed_seconds"]:.2f}秒。\n'
    (HERE / '实验报告.md').write_text(report, encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(json.dumps({'paired_deltas': deltas, 'advance': advance, 'elapsed_seconds': metrics['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
