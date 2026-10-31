import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.integrate import trapezoid
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.data import load_public_test_segments, load_train_segments
from emo_rec.features import CHANNEL_NAMES, FREQUENCY_BANDS
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = HERE / 'cache' / 'detrended_features.npz'
META_KEYS = ('subject_ids', 'health_groups', 'original_trial_ids', 'y_train', 'public_user_ids', 'public_trial_ids')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def extract_detrended_features(segments):
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (30, 2500) or len(segments) == 0:
        raise ValueError('Expected nonempty (n_windows, 30, 2500) EEG')
    if not np.isfinite(segments).all():
        raise ValueError('EEG must be finite')
    frequencies, psd = welch(segments, fs=250, window='hann', nperseg=500,
                             noverlap=250, detrend='constant', scaling='density', axis=-1)
    if not np.isfinite(psd).all():
        raise ValueError('PSD must be finite')
    selected = (frequencies >= 1) & (frequencies <= 45)
    f = frequencies[selected]
    log_psd = np.log(np.maximum(psd[..., selected], 1e-20))
    x = np.log(f)
    centered_x = x - x.mean()
    mean = log_psd.mean(axis=-1, keepdims=True)
    slope = np.sum((log_psd - mean) * centered_x, axis=-1, keepdims=True) / np.sum(centered_x ** 2)
    residual = log_psd - (mean + slope * centered_x)
    # Integrate the PSD/background ratio, then take its natural logarithm.
    powers = []
    for low, high in FREQUENCY_BANDS.values():
        mask = (f >= low) & (f < high)
        values = residual[..., mask]
        offset = values.max(axis=-1)
        powers.append(offset + np.log(trapezoid(np.exp(values - offset[..., None]), f[mask], axis=-1)))
    features = np.stack(powers, axis=-1).reshape(len(segments), 150)
    if not np.isfinite(features).all():
        raise ValueError('Features must be finite')
    return features


def assert_cache_alignment(spectral, detrended):
    for key in META_KEYS:
        if not np.array_equal(spectral[key], detrended[key]):
            raise ValueError(f'Cache metadata mismatch: {key}')
    for key, rows in (('X_train', len(spectral['y_train'])), ('X_public', len(spectral['public_user_ids']))):
        if detrended[key].shape != (rows, 150) or not np.isfinite(detrended[key]).all():
            raise ValueError(f'Invalid feature shape/values: {key}')


def build_cache(spectral, config):
    train = []; labels = []; subjects = []; groups = []; trials = []
    public = []; users = []; public_trials = []; sources = []
    paths = sorted((ROOT / '数据' / '训练集').glob('*/*.mat'))
    for index, path in enumerate(paths):
        segments, y, ids = load_train_segments(path)
        train.append(extract_detrended_features(segments))
        labels.extend(y); subjects.extend([path.stem] * len(y))
        groups.extend(['DEP' if path.parent.name == '抑郁症患者' else 'HC'] * len(y)); trials.extend(ids)
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
        print(f'Features {index + 1}/{len(paths)}: {path.stem}', flush=True)
    paths = sorted((ROOT / '数据' / '公开测试集').glob('*.mat'), key=lambda p: int(p.stem.replace('P_test', '')))
    for path in paths:
        segments = load_public_test_segments(path)
        public.append(extract_detrended_features(segments))
        users.extend([path.stem] * len(segments)); public_trials.extend(range(1, len(segments) + 1))
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
    data = dict(X_train=np.concatenate(train), y_train=np.asarray(labels), subject_ids=np.asarray(subjects),
                health_groups=np.asarray(groups), original_trial_ids=np.asarray(trials), X_public=np.concatenate(public),
                public_user_ids=np.asarray(users), public_trial_ids=np.asarray(public_trials),
                feature_names=np.asarray([f'detrended_log_bandpower_{c}_{b}' for c in CHANNEL_NAMES for b in FREQUENCY_BANDS]),
                train_window_indices=np.tile(np.arange(40), len(train)), public_window_indices=np.tile(np.arange(8), len(public)),
                metadata_json=np.asarray(json.dumps({'config': config, 'sources': sources, 'spectral_sha256': sha256(HERE / config['spectral_cache']),
                                                     'definition': 'window indices follow data.py; channel-major then band-major'}, ensure_ascii=False)))
    assert_cache_alignment(spectral, data)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, **data)
    return data


def load_reference(spectral, config):
    keep = spectral['subject_ids'] != 'HC43'
    frame = pd.read_csv(HERE / config['reference_oof'])
    for col, key in (('subject_id', 'subject_ids'), ('health_group', 'health_groups'), ('original_trial_id', 'original_trial_ids'), ('y_true', 'y_train')):
        if not np.array_equal(frame[col].to_numpy(), spectral[key][keep]):
            raise ValueError(f'002 OOF alignment mismatch: {col}')
    if len(frame) != 2760 or frame.subject_id.nunique() != 69:
        raise ValueError('Unexpected training rows/subjects')
    assert_subject_isolation(frame)
    return keep, frame.drop(columns='prob_positive')


def assert_subject_isolation(frame):
    if set(frame.fold) != set(range(5)) or frame.groupby('subject_id').fold.nunique().max() != 1:
        raise ValueError('Invalid folds or subject leakage')


def metrics(y, p):
    return {'auc': float(roc_auc_score(y, p)), 'f1': float(f1_score(y, p >= .5)),
            'balanced_accuracy': float(balanced_accuracy_score(y, p >= .5))}


def summarize(frame, p):
    result = {'overall': metrics(frame.y_true, p)}
    result['health_groups'] = {g: metrics(frame.loc[frame.health_group == g, 'y_true'], p[frame.health_group == g]) for g in ('HC', 'DEP')}
    result['folds'] = {str(f): metrics(frame.loc[frame.fold == f, 'y_true'], p[frame.fold == f]) for f in range(5)}
    result['subjects'] = {s: metrics(frame.loc[frame.subject_id == s, 'y_true'], p[frame.subject_id == s]) for s in frame.subject_id.unique()}
    result['subject_auc_mean'] = float(np.mean([m['auc'] for m in result['subjects'].values()]))
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--rebuild-features', action='store_true'); args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    spectral = load_cache(HERE / config['spectral_cache'])
    detrended = build_cache(spectral, config) if args.rebuild_features or not CACHE.exists() else load_cache(CACHE)
    assert_cache_alignment(spectral, detrended)
    metadata = json.loads(str(detrended['metadata_json']))
    if metadata['config'] != config or metadata['spectral_sha256'] != sha256(HERE / config['spectral_cache']):
        raise ValueError('Stale feature cache; rebuild required')
    keep, reference = load_reference(spectral, config)
    output = HERE / 'predictions'; output.mkdir(exist_ok=True)
    runs = {}; summary = []
    with threadpool_limits(limits=1):
        for name, data in (('spectral_control', spectral), ('detrended150', detrended)):
            X = normalize_by_subject(data['X_train'][keep, :150], data['subject_ids'][keep], 'zscore')
            XP = normalize_by_subject(data['X_public'][:, :150], data['public_user_ids'], 'zscore')
            y = reference.y_true.to_numpy(); p = np.full(len(y), np.nan)
            for fold in range(5):
                valid = reference.fold.to_numpy() == fold
                model = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config['hgb'], random_state=2026 + fold))
                model.fit(X[~valid], y[~valid]); p[valid] = model.predict_proba(X[valid])[:, 1]
                print(f'{name}: fold {fold + 1}/5', flush=True)
            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError('Invalid OOF probability')
            frame = reference.copy(); frame['prob_positive'] = p; frame.to_csv(output / f'{name}_oof.csv', index=False)
            model = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config['hgb'], random_state=2026))
            model.fit(X, y); pp = model.predict_proba(XP)[:, 1]
            public = pd.DataFrame({'user_id': data['public_user_ids'], 'trial_id': data['public_trial_ids'], 'prob. of positive emotion': pp})
            validate_submission(public, [f'P_test{i}' for i in range(1, 11)])
            public.to_csv(output / f'{name}_public.csv', index=False)
            runs[name] = summarize(reference, p)
            summary.append({'run_name': name, **runs[name]['overall'], 'subject_auc_mean': runs[name]['subject_auc_mean'],
                            **{f'{g.lower()}_{k}': v for g, m in runs[name]['health_groups'].items() for k, v in m.items()}})
    result = {'config': config, 'runs': runs, 'elapsed_seconds': time.perf_counter() - started,
              'provenance': {'spectral_sha256': sha256(HERE / config['spectral_cache']), 'detrended_sha256': sha256(CACHE),
                             'reference_sha256': sha256(HERE / config['reference_oof']), 'python': platform.python_version(),
                             'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
              'protocol': {'cache_alignment_checked': True, 'subject_isolation_checked': True, 'train_rows': 2760, 'public_rows': 80,
                           'subjectwise_unlabeled_zscore': True, 'exploratory_reused_folds': True}}
    (HERE / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame(summary).to_csv(HERE / 'summary.csv', index=False)
    report = '# 实验033：逐窗口去1/f背景频谱HGB\n\n## 特征定义与协议\n\n直接按data.py读取原始训练/公开EEG。每10秒、30通道窗口，Welch(fs=250,nperseg=500,noverlap=250,Hann,detrend=constant,scaling=density)。在1–45Hz（含端点）以自然对数PSD（下限1e-20）对log频率进行逐窗口逐通道普通最小二乘拟合，包含截距，不使用标签。此处是直线背景估计，不是稳健峰剔除或完整FOOOF，振荡峰可能影响拟合。\n\n去背景定义为r=logPSD−拟合背景；对exp(r)=PSD/背景在五频带[1,4)、[4,8)、[8,13)、[13,30)、[30,45)上梯形积分后取自然对数，以log-sum-exp式平移避免溢出，形成通道优先150维。与001保持半开频带及梯形积分约定。不是对有正负的log残差直接取对数。001已有前150维作为原始log bandpower对照。缓存记录全部训练/公开元数据、窗口索引、原始文件SHA256及配置，并逐项核验001元数据。\n\n排除HC43，69被试2760窗口；复用002固定5折并核验逐行顺序和被试隔离。每被试无标签z-score（验证/公开被试使用其自身所有窗口，属于批量无标签适配，不代表单窗口在线推理）。StandardScaler仅拟合训练折，HGB固定016参数，外层seed=2026+fold，全量公开模型seed=2026，阈值0.5。不搜索参数或融合权重。公开80行无标签，不能报告公开性能。\n\n## 实测结果\n\n|方法|AUC|F1|BAcc|HC AUC|HC F1|DEP AUC|DEP F1|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for row in summary:
        report += f"|{row['run_name']}|{row['auc']:.6f}|{row['f1']:.6f}|{row['balanced_accuracy']:.6f}|{row['hc_auc']:.6f}|{row['hc_f1']:.6f}|{row['dep_auc']:.6f}|{row['dep_f1']:.6f}|\n"
    delta = runs['detrended150']['overall']['auc'] - runs['spectral_control']['overall']['auc']
    report += f'\n去背景相对对照的OOF AUC差为{delta:+.6f}。本结果只代表反复使用002折的探索性比较，不是独立确认或统计显著性证据。无论升降，均不能据此宣称普遍跨被试优势。\n\n运行用时{result["elapsed_seconds"]:.2f}秒。命令：`.venv\\Scripts\\python.exe -B test/experiments/033_detrended_spectral_hgb/run_experiment.py --rebuild-features`。测试命令：`.venv\\Scripts\\python.exe -B -m pytest -q -p no:cacheprovider test/experiments/033_detrended_spectral_hgb/test_protocol.py`。\n'
    (HERE / '实验报告.md').write_text(report, encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
