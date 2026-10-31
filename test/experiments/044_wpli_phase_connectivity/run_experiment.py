import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.signal import butter, hilbert, sosfiltfilt
from threadpoolctl import threadpool_limits

from emo_rec.data import load_train_segments, load_public_test_segments
from emo_rec.features import CHANNEL_NAMES
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
spec = importlib.util.spec_from_file_location('repeat032', HERE.parent / '032_repeated_subject_folds/run_experiment.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
PAIRS = np.asarray([[CHANNEL_NAMES.index(a), CHANNEL_NAMES.index(b)] for a, b in CONFIG['pairs']])


def extract_wpli(segments):
    x = np.asarray(segments, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (30, 2500) or not np.isfinite(x).all():
        raise ValueError('Expected finite (windows, 30, 2500) EEG')
    features = []
    for low, high in CONFIG['bands']:
        sos = butter(4, [low, high], fs=250, btype='bandpass', output='sos')
        z = hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)
        imaginary = np.imag(z[:, PAIRS[:, 0]] * np.conj(z[:, PAIRS[:, 1]]))
        denominator = np.mean(np.abs(imaginary), axis=-1)
        features.append(np.divide(np.abs(np.mean(imaginary, axis=-1)), denominator,
                                  out=np.zeros_like(denominator), where=denominator > 0))
    return np.stack(features, axis=-1).reshape(len(x), 50)


def build_cache():
    arrays = {k: [] for k in ('X_train', 'y_train', 'subject_ids', 'health_groups', 'original_trial_ids',
                              'X_public', 'public_user_ids', 'public_trial_ids')}
    metadata = []
    paths = sorted((ROOT / '数据/训练集').glob('*/*.mat'))
    for i, path in enumerate(paths):
        x, y, trials = load_train_segments(path)
        arrays['X_train'].append(extract_wpli(x)); arrays['y_train'].append(y)
        arrays['subject_ids'].append(np.repeat(path.stem, len(y)))
        arrays['health_groups'].append(np.repeat('DEP' if path.parent.name == '抑郁症患者' else 'HC', len(y)))
        arrays['original_trial_ids'].append(trials)
        metadata.append({'path': str(path.relative_to(ROOT)), 'sha256': base.sha256(path), 'windows': len(y)})
        print(f'Features train {i+1}/{len(paths)} {path.stem}', flush=True)
    for path in sorted((ROOT / '数据/公开测试集').glob('*.mat'), key=lambda p: int(p.stem.replace('P_test', ''))):
        x = load_public_test_segments(path)
        arrays['X_public'].append(extract_wpli(x))
        arrays['public_user_ids'].append(np.repeat(path.stem, len(x)))
        arrays['public_trial_ids'].append(np.arange(1, len(x)+1))
        metadata.append({'path': str(path.relative_to(ROOT)), 'sha256': base.sha256(path), 'windows': len(x)})
        print(f'Features public {path.stem}', flush=True)
    data = {k: np.concatenate(v) for k, v in arrays.items()}
    np.savez_compressed(HERE / 'cache/features.npz', **data)
    (HERE / 'input_metadata/raw_files.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return data


def validate_oof(frame, expected):
    assert len(frame) == expected
    assert np.isfinite(frame.prob_positive).all()
    assert frame.prob_positive.between(0, 1).all()
    base.assert_subject_isolation(frame.subject_id.to_numpy(), frame.fold.to_numpy())
    assert frame.groupby('subject_id').size().eq(40).all()


def public_frame(data, p):
    frame = pd.DataFrame({'user_id': data['public_user_ids'], 'trial_id': data['public_trial_ids'],
                          'prob. of positive emotion': p})
    validate_submission(frame, [f'P_test{i}' for i in range(1, 11)])
    return frame


def main():
    start = time.perf_counter()
    for folder in ('cache', 'predictions', 'input_metadata'):
        (HERE / folder).mkdir(exist_ok=True)
    data = build_cache()
    feature_seconds = time.perf_counter() - start
    spectral_path = (HERE / CONFIG['spectral_cache']).resolve()
    spectral = base.load_cache(spectral_path)
    base.assert_cache_alignment(spectral, data)
    keep = data['subject_ids'] != 'HC43'
    subjects = data['subject_ids'][keep]; groups = data['health_groups'][keep]; y = data['y_train'][keep]
    assert len(y) == 2760 and len(np.unique(subjects)) == 69
    reference = pd.DataFrame({'subject_id': subjects, 'health_group': groups,
                              'original_trial_id': data['original_trial_ids'][keep], 'y_true': y})
    fixed = base.make_subject_folds(data['subject_ids'], data['health_groups'], 2026)[keep]
    schedules = [('fixed2026', 2026, fixed)] + [(f'repeat{seed}', seed, base.make_subject_folds(subjects, groups, seed)) for seed in CONFIG['random_seeds']]
    for name, seed, folds in schedules:
        base.assert_subject_isolation(subjects, folds)
        ref = reference.copy(); ref['fold'] = folds
        ref.to_csv(HERE / f'input_metadata/folds_{name}.csv', index=False)
    sx = normalize_by_subject(spectral['X_train'][keep, :150], subjects, 'zscore')
    sp = normalize_by_subject(spectral['X_public'][:, :150], spectral['public_user_ids'], 'zscore')
    runs = {}; rows = []
    with threadpool_limits(limits=1):
        for name, X, XP in [('spectral_control', sx, sp), ('wpli_hgb', data['X_train'][keep], data['X_public'])]:
            results = []
            for schedule, seed, folds in schedules:
                p = np.full(len(y), np.nan)
                counts = np.zeros(len(y), dtype=int)
                for fold in range(5):
                    tr, va = folds != fold, folds == fold
                    model = base.make_model(CONFIG, 2026 + fold)
                    model.fit(X[tr], y[tr]); p[va] = model.predict_proba(X[va])[:, 1]; counts[va] += 1
                    print(f'{name} {schedule} fold {fold+1}/5 complete', flush=True)
                assert (counts == 1).all()
                ref = reference.copy(); ref['fold'] = folds; ref['prob_positive'] = p
                validate_oof(ref, 2760)
                ref.to_csv(HERE / f'predictions/{name}_{schedule}_oof.csv', index=False)
                result = base.summarize(y, p, ref, folds); result['schedule'] = schedule; result['seed'] = seed
                results.append(result)
                rows.append({'method': name, 'schedule': schedule, **result['overall'],
                             'subject_macro_auc': result['subject_auc_mean'],
                             'hc_auc': result['health_groups']['HC']['pooled']['auc'],
                             'dep_auc': result['health_groups']['DEP']['pooled']['auc']})
                print(json.dumps(rows[-1]), flush=True)
            model = base.make_model(CONFIG, 2026); model.fit(X, y)
            public_frame(data, model.predict_proba(XP)[:, 1]).to_csv(HERE / f'predictions/{name}_public.csv', index=False)
            repeated = [r for r in rows if r['method'] == name and r['schedule'].startswith('repeat')]
            aggregate = {key: {'mean': float(np.mean([r[key] for r in repeated])),
                               'std': float(np.std([r[key] for r in repeated], ddof=1))}
                         for key in ('auc', 'f1', 'balanced_accuracy', 'subject_macro_auc', 'hc_auc', 'dep_auc')}
            runs[name] = {'n_features': X.shape[1], 'evaluations': results, 'repeated_aggregate': aggregate}
    delta = runs['wpli_hgb']['repeated_aggregate']['auc']['mean'] - runs['spectral_control']['repeated_aggregate']['auc']['mean']
    decision = '低于频谱控制，停止该方向，不调参、不融合。' if delta < 0 else '重复均值不低于控制；本次不追加调参，仍需独立确认。'
    elapsed = time.perf_counter() - start
    metrics = {'config': CONFIG, 'status': 'completed_fixed_and_all_five_repeats', 'runs': runs,
               'delta_repeated_auc': delta, 'decision': decision, 'feature_seconds': feature_seconds, 'elapsed_seconds': elapsed,
               'provenance': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                              'sklearn': sklearn.__version__, 'spectral_cache_sha256': base.sha256(spectral_path),
                              'phase_cache_sha256': base.sha256(HERE / 'cache/features.npz')}}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    report = '# 实验044：独立10秒窗口wPLI相位连接\n\n'
    report += '真实完成002固定5折（70人分折后排除HC43）及032式2027..2031五次重复（69人重新分折）。2760训练窗口、80公开窗口。仅复用001原始频谱特征缓存，不读取历史OOF概率；所有控制逐折重训。\n\n'
    report += '## 定义与协议\n\n10个预先固定左右通道对 × 5频带 = 50维。各独立10秒窗口先四阶Butterworth带通+sosfiltfilt默认边缘填充，再Hilbert解析信号。wPLI = |mean(Im(zL conj(zR)))| / mean(|Im(zL conj(zR))|)，分母零时为0；时间点是期望样本，未作去偏修正或边缘裁切。这是时间域解析信号wPLI估计，不是Welch近似、不是相位角PLV，也不是跨试次谱wPLI。有限窗口/滤波边缘会产生估计偏差。\n\n'
    report += '特征提取完全不接收标签、行序、trial id或邻窗；训练/验证/公开使用相同函数，wPLI无被试跨窗归一化。每折StandardScaler+HGB只fit训练被试，验证仅transform/predict。对照按要求保留150维subject-zscore无标签转导归一化（使用同被试其他窗口），这不属于严格单窗口处理，和wPLI路径明确区分。HGB参数与032一致，seed=2026+fold，阈值0.5。不调参，不用公开标签。每重复独立OOF；公开仅一次全训练拟合保存预测，不报告公开指标。\n\n'
    report += '## 五次重复均值±样本标准差\n\n|方法|AUC|F1|BAcc|subject宏AUC|HC AUC|DEP AUC|\n|---|---|---|---|---|---|---|\n'
    for name, run in runs.items():
        report += '|' + name + '|' + '|'.join(f"{v['mean']:.6f}±{v['std']:.6f}" for v in run['repeated_aggregate'].values()) + '|\n'
    report += '\n## 固定折\n\n'
    for row in rows:
        if row['schedule'] == 'fixed2026':
            report += f"- {row['method']}: AUC={row['auc']:.6f}, F1={row['f1']:.6f}, BAcc={row['balanced_accuracy']:.6f}, subject宏AUC={row['subject_macro_auc']:.6f}, HC={row['hc_auc']:.6f}, DEP={row['dep_auc']:.6f}。\n"
    report += f'\n## 结论\n\n重复AUC差值={delta:+.6f}。{decision}\n\n提取耗时{feature_seconds:.2f}秒，总运行{elapsed:.2f}秒。每折/每被试完整指标见metrics.json，各重复独立OOF见predictions；测试输出见test_protocol.txt，运行输出见run_output.txt。\n'
    (HERE / '实验报告.md').write_text(report, encoding='utf-8')
    print(f'COMPLETE elapsed_seconds={elapsed:.3f} delta_auc={delta:+.6f} {decision}', flush=True)


if __name__ == '__main__':
    main()
