import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.data import load_train_segments, load_public_test_segments
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def window_covariances(segments, config):
    x = np.asarray(segments, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (30, 2500) or not np.isfinite(x).all():
        raise ValueError('Expected finite independent 10-second windows')
    bands = []
    for low, high in config['bands']:
        sos = butter(config['filter_order'], [low, high], btype='bandpass', fs=250, output='sos')
        z = sosfiltfilt(sos, x, axis=-1)
        z -= z.mean(axis=-1, keepdims=True)
        c = z @ z.transpose(0, 2, 1) / z.shape[-1]
        bands.append(c)
    return np.stack(bands, axis=1)


class FilterBankCSP:
    def __init__(self, regularization=1e-6):
        self.regularization = regularization

    def fit(self, covariances, y):
        if set(np.unique(y)) != {0, 1} or len(y) != len(covariances):
            raise ValueError('CSP requires aligned binary training labels')
        self.filters_ = []
        for band in range(5):
            c = covariances[:, band] + self.regularization * np.eye(30)
            c = c / np.trace(c, axis1=1, axis2=2)[:, None, None]
            c0, c1 = c[y == 0].mean(axis=0), c[y == 1].mean(axis=0)
            _, vectors = eigh(c1, c0 + c1)
            self.filters_.append(vectors[:, [0, 1, -2, -1]].T)
        self.filters_ = np.asarray(self.filters_)
        return self

    def transform(self, covariances):
        variance = np.einsum('bkc,nbcd,bkd->nbk', self.filters_, covariances, self.filters_)
        features = np.log(np.maximum(variance, 1e-30)).reshape(len(covariances), 20)
        if not np.isfinite(features).all():
            raise ValueError('Non-finite CSP features')
        return features


def make_folds(subjects, groups, seed):
    unique = np.unique(subjects)
    health = np.asarray([groups[np.flatnonzero(subjects == s)[0]] for s in unique])
    assignment = {}
    for fold, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(unique, health)):
        assignment.update({s: fold for s in unique[valid]})
    return np.asarray([assignment[s] for s in subjects])


def assert_disjoint(subjects, train, valid):
    if set(subjects[train]) & set(subjects[valid]):
        raise ValueError('Subject leakage')


def fit_csp_fold(cov, y, subjects, train, valid, config):
    assert_disjoint(subjects, train, valid)
    csp = FilterBankCSP(config['covariance_regularization']).fit(cov[train], y[train])
    model = make_pipeline(StandardScaler(), LogisticRegression(**config['logistic']))
    model.fit(csp.transform(cov[train]), y[train])
    return csp, model


def check_oof(p, visits, expected=2760):
    if len(p) != expected or not np.isfinite(p).all() or not np.all(visits == 1):
        raise ValueError('Incomplete or duplicate OOF rows')


def public_frame(data, p):
    frame = pd.DataFrame({'user_id': data['public_user_ids'], 'trial_id': data['public_trial_ids'], 'prob. of positive emotion': p})
    validate_submission(frame, [f'P_test{i}' for i in range(1, 11)])
    return frame


def metrics(y, p):
    return {'auc': float(roc_auc_score(y, p)), 'f1': float(f1_score(y, p >= .5)), 'balanced_accuracy': float(balanced_accuracy_score(y, p >= .5))}


def summarize(frame, p):
    out = {'overall': metrics(frame.y_true, p)}
    for key, column in [('folds', 'fold'), ('health_groups', 'health_group'), ('subjects', 'subject_id')]:
        out[key] = {str(value): metrics(frame.loc[frame[column] == value, 'y_true'], p[frame[column].to_numpy() == value]) for value in frame[column].unique()}
    out['subject_auc_mean'] = float(np.mean([v['auc'] for v in out['subjects'].values()]))
    return out


def build_cache(spectral, config):
    cache = HERE / 'cache' / 'window_covariances.npz'
    # Only label-free, independent-window covariance is cached; CSP is never cached globally.
    sources = []
    pieces = []; labels = []; subjects = []; groups = []; trials = []
    for path in sorted((ROOT / '数据' / '训练集').glob('*/*.mat')):
        if path.stem == 'HC43':
            continue
        x, y, ids = load_train_segments(path)
        pieces.append(window_covariances(x, config))
        labels.extend(y); subjects.extend([path.stem] * len(y)); trials.extend(ids)
        groups.extend(['DEP' if path.parent.name == '抑郁症患者' else 'HC'] * len(y))
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
        print(f'Window covariance: {path.stem}', flush=True)
    public = []; users = []; ptrials = []
    for path in sorted((ROOT / '数据' / '公开测试集').glob('*.mat'), key=lambda p: int(p.stem.replace('P_test', ''))):
        x = load_public_test_segments(path)
        public.append(window_covariances(x, config)); users.extend([path.stem] * len(x)); ptrials.extend(range(1, len(x) + 1))
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
    data = {'cov_train': np.concatenate(pieces), 'cov_public': np.concatenate(public), 'y_train': np.asarray(labels), 'subject_ids': np.asarray(subjects), 'health_groups': np.asarray(groups), 'original_trial_ids': np.asarray(trials), 'public_user_ids': np.asarray(users), 'public_trial_ids': np.asarray(ptrials)}
    keep = spectral['subject_ids'] != 'HC43'
    for key in ('y_train', 'subject_ids', 'health_groups', 'original_trial_ids', 'public_user_ids', 'public_trial_ids'):
        expected = spectral[key] if key.startswith('public') else spectral[key][keep]
        if not np.array_equal(data[key], expected):
            raise ValueError(f'Input alignment mismatch: {key}')
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, **data)
    metadata = {'sources': sources, 'spectral_sha256': sha256(HERE / config['spectral_cache']), 'data_loader_sha256': sha256(ROOT / 'src/emo_rec/data.py'), 'script_sha256': sha256(Path(__file__)), 'config': config, 'train_rows': len(labels), 'public_rows': len(users), 'alignment_checked': True}
    (HERE / 'input_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return data


def main():
    started = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    spectral = load_npz(HERE / config['spectral_cache'])
    output = HERE / 'predictions'; output.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        data = build_cache(spectral, config)
        keep = spectral['subject_ids'] != 'HC43'
        subjects, groups, y = data['subject_ids'], data['health_groups'], data['y_train']
        assert len(y) == 2760 and len(np.unique(subjects)) == 69
        sx = normalize_by_subject(spectral['X_train'][keep, :150], subjects, 'zscore')
        sp = normalize_by_subject(spectral['X_public'][:, :150], data['public_user_ids'], 'zscore')
        protocols = {'fixed': make_folds(spectral['subject_ids'], spectral['health_groups'], 2026)[keep]}
        protocols.update({f'repeat_{seed}': make_folds(subjects, groups, seed) for seed in config['repeat_seeds']})
        runs = {}; rows = []; audit = []
        for protocol, folds in protocols.items():
            frame = pd.DataFrame({'subject_id': subjects, 'health_group': groups, 'original_trial_id': data['original_trial_ids'], 'fold': folds, 'y_true': y})
            frame.to_csv(output / f'{protocol}_folds.csv', index=False)
            probabilities = {name: np.full(len(y), np.nan) for name in ('filterbank_csp', 'spectral_control')}
            visits = np.zeros(len(y), dtype=int)
            for fold in range(5):
                train, valid = folds != fold, folds == fold
                csp, lr = fit_csp_fold(data['cov_train'], y, subjects, train, valid, config)
                probabilities['filterbank_csp'][valid] = lr.predict_proba(csp.transform(data['cov_train'][valid]))[:, 1]
                hgb = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config['hgb'], random_state=2026 + fold))
                hgb.fit(sx[train], y[train])
                probabilities['spectral_control'][valid] = hgb.predict_proba(sx[valid])[:, 1]
                visits[valid] += 1
                audit.append({'protocol': protocol, 'fold': fold, 'csp_train_subjects': sorted(set(subjects[train])), 'validation_subjects': sorted(set(subjects[valid])), 'csp_train_rows': int(train.sum()), 'feature_dim': 20})
                print(f'{protocol}: fold {fold + 1}/5 completed', flush=True)
            runs[protocol] = {}
            for name, p in probabilities.items():
                check_oof(p, visits)
                pred = frame.copy(); pred['prob_positive'] = p
                pred.to_csv(output / f'{protocol}_{name}_oof.csv', index=False)
                result = summarize(frame, p); runs[protocol][name] = result
                rows.append({'protocol': protocol, 'method': name, **result['overall'], 'subject_auc_mean': result['subject_auc_mean'], 'hc_auc': result['health_groups']['HC']['auc'], 'dep_auc': result['health_groups']['DEP']['auc']})
        csp = FilterBankCSP(config['covariance_regularization']).fit(data['cov_train'], y)
        lr = make_pipeline(StandardScaler(), LogisticRegression(**config['logistic'])).fit(csp.transform(data['cov_train']), y)
        hgb = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config['hgb'], random_state=2026)).fit(sx, y)
        for name, p in [('filterbank_csp', lr.predict_proba(csp.transform(data['cov_public']))[:, 1]), ('spectral_control', hgb.predict_proba(sp)[:, 1])]:
            public_frame(data, p).to_csv(output / f'{name}_public.csv', index=False)
    aggregate = {}
    for name in ('filterbank_csp', 'spectral_control'):
        subset = [r for r in rows if r['method'] == name and r['protocol'] != 'fixed']
        aggregate[name] = {key: {'mean': float(np.mean([r[key] for r in subset])), 'std': float(np.std([r[key] for r in subset], ddof=1))} for key in ('auc', 'f1', 'balanced_accuracy', 'subject_auc_mean', 'hc_auc', 'dep_auc')}
    fixed_delta = runs['fixed']['filterbank_csp']['overall']['auc'] - runs['fixed']['spectral_control']['overall']['auc']
    repeated_delta = aggregate['filterbank_csp']['auc']['mean'] - aggregate['spectral_control']['auc']['mean']
    conclusion = '低于控制，停止该方向，不调参追分。' if fixed_delta < 0 or repeated_delta < 0 else '固定折和重复均值优于控制；尚非独立隐藏测试证据。'
    result = {'config': config, 'runs': runs, 'repeat_aggregate': aggregate, 'fixed_auc_delta': fixed_delta, 'repeat_mean_auc_delta': repeated_delta, 'conclusion': conclusion, 'elapsed_seconds': time.perf_counter() - started, 'python': platform.python_version(), 'sklearn': sklearn.__version__, 'protocol': {'threshold': .5, 'supervised_csp_training_fold_only': True, 'independent_window_filtering': True, 'no_existing_oof_read': True, 'public_labels_used': False, 'public_fit': 'one final fit on all 69 training subjects; no repeated public duplicates'}}
    (HERE / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (HERE / 'fit_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    report = '# 实验043：训练折内监督滤波器组CSP + LR\n\n训练与公开EEG均先切独立10秒窗，再逐窗4阶Butterworth SOS零相位五频带滤波。缓存仅含无标签逐窗协方差；各折仅训练被试标签拟合CSP（类均值迹归一化协方差，广义特征分解，两端各2，共20维），投影原始窗口协方差取log variance。StandardScaler+LR仅训练折拟合，无邻窗、行序或trial id输入。HC43完全排除。固定折在全部70被试生成后排除HC43，与002一致；重复2027..2031在69被试生成，与032一致。对照001前150维subject z-score + HGB每折独立重训，无读取历史OOF。对照subject z-score为无标签被试内转导统计，CSP不作subject normalization。阈值固定0.5，无调参。\n\n|协议|方法|AUC|F1|BAcc|subject宏AUC|HC AUC|DEP AUC|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        report += f"|{r['protocol']}|{r['method']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['balanced_accuracy']:.6f}|{r['subject_auc_mean']:.6f}|{r['hc_auc']:.6f}|{r['dep_auc']:.6f}|\n"
    report += f'\n固定折AUC差：{fixed_delta:+.6f}；重复折AUC均值差：{repeated_delta:+.6f}。{conclusion}\n\n'
    for name, stats in aggregate.items():
        report += f"{name} 重复AUC：{stats['auc']['mean']:.6f} ± {stats['auc']['std']:.6f}（样本标准差）。\n\n"
    report += '每折、每被试、HC/DEP详细指标见metrics.json；每次重复OOF独立保存，绝不拼接。公开仅保存全训练重训预测，无公开评估。fit_audit.json保存每折拟合被试清单。专项测试输出见test_protocol.txt；真实日志见run_output.txt。\n'
    (HERE / '实验报告.md').write_text(report, encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(conclusion, flush=True)


if __name__ == '__main__':
    main()
