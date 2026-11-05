import hashlib
import importlib.util
import itertools
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import f1_score, roc_auc_score
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
TEMPLATE = HERE.parent / '122_DEP02_window_regularized_csp_pairs/run_experiment.py'
spec = importlib.util.spec_from_file_location('csp122', TEMPLATE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
CANDIDATES = list(itertools.product(CFG['lambdas'], CFG['components_per_end'], CFG['blocks'], CFG['models'], CFG['C']))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def metrics(y, p):
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return {'auc': float(roc_auc_score(y, p)), 'f1': float(f1_score(y, p >= .5, zero_division=0))}


def scores(y, p):
    # Vectorized exact AUC, including half credit for ties.
    a, b = p[:, y == 1, None], p[:, None, y == 0]
    auc = ((a > b) + .5 * (a == b)).mean(axis=(1, 2))
    pred = p >= .5
    tp = (pred & (y == 1)).sum(axis=1)
    f1 = 2 * tp / np.maximum(pred.sum(axis=1) + y.sum(), 1)
    return auc, f1


def best_index(auc, f1):
    return int(np.lexsort((np.arange(len(auc)), -f1, -auc))[0])


def baseline(subject, y, trial, source):
    folder = HERE.parent / '057_multiscale_permutation_entropy'
    metadata = {'status': 'unavailable', 'reason': None}
    try:
        config = json.loads((folder / 'config.json').read_text(encoding='utf8'))
        prov = json.loads((folder / 'input_metadata/provenance.json').read_text(encoding='utf8'))
        raw = json.loads((folder / 'input_metadata/raw_inputs.json').read_text(encoding='utf8'))
        assert prov['subject_isolation'] is True
        assert sha(folder / 'config.json') == prov['config_sha256']
        raw_record = [r for r in raw if Path(r['path']).stem == subject]
        assert len(raw_record) == 1 and raw_record[0]['sha256'] == sha(source)
        paths = [folder / CFG['baseline'].split('057_multiscale_permutation_entropy/')[1]]
        frame = pd.read_csv(paths[0])
        sub = frame[frame.subject_id == subject].reset_index(drop=True)
        assert len(sub) == 40 and sub.fold.nunique() == 1
        assert np.array_equal(sub.y_true.to_numpy(), y)
        assert np.array_equal(sub.original_trial_id.to_numpy(), trial % 4 + 1)
        for key in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference'):
            path = (folder / config[key]).resolve()
            assert sha(path) == prov['inputs'][key]['sha256'], key
            paths.append(path)
            if key == 'fixed_reference':
                ref = pd.read_csv(path)
                cols = ['subject_id', 'original_trial_id', 'y_true', 'fold']
                assert np.array_equal(frame[cols].to_numpy(), ref[cols].to_numpy())
            else:
                with np.load(path, allow_pickle=False) as z:
                    keep = z['subject_ids'] != config['excluded_subject']
                    for col, field in [('subject_id', 'subject_ids'), ('original_trial_id', 'original_trial_ids'), ('y_true', 'y_train')]:
                        assert np.array_equal(frame[col].to_numpy(), z[field][keep])
        sub['window_in_trial'] = sub.groupby(['subject_id', 'original_trial_id', 'y_true'], sort=False).cumcount()
        assert np.array_equal(sub.window_in_trial.to_numpy(), np.tile(np.arange(5), 8))
        sub['row_index'] = np.arange(40)
        sub['trial_index'] = trial
        sub.to_csv(HERE / f'input_metadata/{subject}_baseline_alignment.csv', index=False)
        p = sub.prob_positive.to_numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        paths += [folder / 'config.json', folder / 'run_experiment.py', folder / 'input_metadata/provenance.json', folder / 'input_metadata/raw_inputs.json', ROOT / 'src/emo_rec/data.py']
        metadata.update(status='aligned', hashes={str(p.relative_to(ROOT)): sha(p) for p in paths},
                        reason='Exact full cache/reference key sequence and raw MAT SHA256 match; within-trial window position follows documented loader order.',
                        subject_model_training_labels_used=False, normalization='057 frozen whole-subject unlabeled zscore; transductive covariates')
        return p, metadata
    except (AssertionError, KeyError, ValueError, OSError) as exc:
        metadata['reason'] = f'{type(exc).__name__}: {exc}; no forced alignment or fusion'
        return None, metadata


class FitCache:
    def __init__(self, cov, y, trial):
        self.cov, self.y, self.trial = cov, y, trial
        self.values = {}
        self.hits = 0

    def get(self, train_trials):
        key = tuple(sorted(int(t) for t in train_trials))
        if key in self.values:
            self.hits += 1
            return self.values[key]
        tr = base.window_indices(self.trial, key)
        va = np.setdiff1d(np.arange(40), tr)
        assert not set(self.trial[tr]) & set(self.trial[va])
        result = np.full((len(CANDIDATES), 40), np.nan)
        feature_cache = {}
        for ci, candidate in enumerate(CANDIDATES):
            lam, comp, block, _, _ = candidate
            fk = (lam, comp)
            if fk not in feature_cache:
                feature_cache[fk] = base.csp_fit_transform(self.cov[tr], self.y[tr], self.cov[va], lam, comp)
            ft, fv = feature_cache[fk]
            result[ci, va] = base.predict(base.block_features(ft, block), self.y[tr], base.block_features(fv, block), candidate)
        assert np.isfinite(result[:, va]).all()
        self.values[key] = result
        return result


def inner_select(cache, train_trials, pe):
    trials = np.asarray(train_trials)
    sums = np.zeros((len(CANDIDATES), 40))
    counts = np.zeros(40, dtype=int)
    folds = []
    for tr_i, va_i in base.trial_pairs(trials):
        tt, vt = trials[tr_i], trials[va_i]
        assert not set(tt) & set(vt)
        vi = base.window_indices(cache.trial, vt)
        sums[:, vi] += cache.get(tt)[:, vi]
        counts[vi] += 1
        folds.append({'train_trials': tt.tolist(), 'valid_trials': vt.tolist()})
    ii = base.window_indices(cache.trial, trials)
    assert np.all(counts[ii] == 3)
    trial_p = np.stack([(sums[:, cache.trial == t] / counts[cache.trial == t]).mean(axis=1) for t in trials], axis=1)
    yy = np.asarray([cache.y[cache.trial == t][0] for t in trials])
    directed = np.stack([trial_p, 1 - trial_p], axis=1).reshape(-1, len(trials))
    auc, f1 = scores(yy, directed)
    idx = best_index(auc, f1)
    rows = [{'candidate_index': k // 2, 'direction': CFG['directions'][k % 2], 'auc': float(auc[k]), 'f1': float(f1[k])} for k in range(len(auc))]
    weight, weights = None, []
    if pe is not None:
        pe_trial = np.asarray([pe[cache.trial == t].mean() for t in trials])
        blend = np.asarray([w * directed[idx] + (1 - w) * pe_trial for w in CFG['csp_weights']])
        wa, wf = scores(yy, blend)
        weight = CFG['csp_weights'][best_index(wa, wf)]
        weights = [{'csp_weight': w, 'auc': float(a), 'f1': float(f)} for w, a, f in zip(CFG['csp_weights'], wa, wf)]
    return idx // 2, idx % 2, weight, rows, weights, folds


def run_subject(subject, number):
    start = time.perf_counter()
    print(f'[{number}/7] {subject}: load / baseline alignment', flush=True)
    paths = list((ROOT / '数据/训练集').rglob(subject + '.mat'))
    assert len(paths) == 1, (subject, paths)
    source = paths[0]
    sys.path.insert(0, str(ROOT / 'src'))
    from emo_rec.data import load_train_segments
    x, y, original_trial = load_train_segments(source)
    trial = original_trial - 1 + 4 * y
    assert x.shape == (40, 30, 2500) and np.isfinite(x).all()
    assert np.array_equal(np.bincount(trial), np.full(8, 5))
    assert np.array_equal(y.reshape(8, 5), np.repeat(np.repeat([0, 1], 4)[:, None], 5, axis=1))
    pe, alignment = baseline(subject, y, trial, source)
    cache = FitCache(base.covariances(x), y, trial)
    methods = ['csp'] + (['baseline_057_plus_pe', 'csp_plus_pe'] if pe is not None else [])
    if subject in ('DEP03', 'DEP17'):
        methods += ['csp_raw_control', 'csp_reverse_control']
    sums = {m: np.zeros(40) for m in methods}
    counts = np.zeros(40, dtype=int)
    pair_rows, audits, pair_results = [], [], []
    for pair_id, (tt, vt) in enumerate(base.trial_pairs(np.arange(8))):
        print(f'[{number}/7] {subject} pair {pair_id + 1}/16 train={tt.tolist()} query={vt.tolist()}', flush=True)
        assert not set(tt) & set(vt)
        ci, di, weight, candidate_scores, weight_scores, folds = inner_select(cache, tt, pe)
        vi = base.window_indices(trial, vt)
        raw = cache.get(tt)[ci, vi]
        directed = raw if di == 0 else 1 - raw
        outputs = {'csp': directed}
        if pe is not None:
            outputs.update(baseline_057_plus_pe=pe[vi], csp_plus_pe=weight * directed + (1 - weight) * pe[vi])
        if subject in ('DEP03', 'DEP17'):
            outputs.update(csp_raw_control=raw, csp_reverse_control=1 - raw)
        counts[vi] += 1
        for method, p in outputs.items():
            sums[method][vi] += p
            tp = np.asarray([p[trial[vi] == t].mean() for t in sorted(vt)])
            ty = np.asarray([y[trial == t][0] for t in sorted(vt)])
            wm, tm = metrics(y[vi], p), metrics(ty, tp)
            pair_results.append({'subject': subject, 'method': method, 'pair_id': pair_id, 'window_auc': wm['auc'], 'window_f1': wm['f1'], 'trial_auc': tm['auc'], 'trial_f1': tm['f1']})
            for i, prob in zip(vi, p):
                pair_rows.append({'subject': subject, 'method': method, 'pair_id': pair_id, 'row_index': int(i), 'trial_index': int(trial[i]), 'original_trial_id': int(trial[i] % 4 + 1), 'window_in_trial': int(i % 5), 'y_true': int(y[i]), 'prob_positive': float(prob)})
        audits.append({'pair_id': pair_id, 'train_trials': tt.tolist(), 'query_trials': vt.tolist(), 'train_windows': 30, 'query_windows': 10, 'inner_folds': folds, 'candidate_index': ci, 'candidate': CANDIDATES[ci], 'direction': CFG['directions'][di], 'csp_weight': weight, 'inner_candidate_scores': candidate_scores, 'inner_weight_scores': weight_scores, 'outer_validation_labels_used_for_selection': False, 'public_labels_used': False})
    assert np.all(counts == 4)
    summary, windows, trials_out = [], [], []
    for method in methods:
        p = sums[method] / counts
        tp = p.reshape(8, 5).mean(axis=1)
        yy = y.reshape(8, 5)[:, 0]
        quota = np.zeros(8, dtype=int)
        quota[np.argsort(-tp, kind='stable')[:4]] = 1
        wm, tm = metrics(y, p), metrics(yy, tp)
        pm = [r for r in pair_results if r['method'] == method]
        row = {'subject': subject, 'method': method, 'window_auc': wm['auc'], 'window_f1': wm['f1'], 'trial_auc': tm['auc'], 'trial_f1': tm['f1'], 'quota4_f1': float(f1_score(yy, quota)), 'predictions_finite': True}
        for key in ('window_auc', 'window_f1', 'trial_auc', 'trial_f1'):
            row['pair_mean_' + key] = float(np.mean([r[key] for r in pm]))
            row['worst_pair_' + key] = float(min(r[key] for r in pm))
        row['auc_gain_vs_baseline'] = wm['auc'] - metrics(y, pe)['auc'] if pe is not None else None
        summary.append(row)
        for i, prob in enumerate(p):
            windows.append({'subject': subject, 'method': method, 'row_index': i, 'trial_index': int(trial[i]), 'original_trial_id': int(trial[i] % 4 + 1), 'window_in_trial': i % 5, 'y_true': int(y[i]), 'prob_positive': float(prob), 'prediction_count': int(counts[i]), 'prediction_fixed05': int(prob >= .5)})
        for i, prob in enumerate(tp):
            trials_out.append({'subject': subject, 'method': method, 'trial_index': i, 'original_trial_id': i % 4 + 1, 'y_true': int(yy[i]), 'prob_positive': float(prob), 'prediction_fixed05': int(prob >= .5), 'prediction_quota4': int(quota[i]), 'prediction_count': 4})
    assert len(windows) == 40 * len(methods) and len(trials_out) == 8 * len(methods)
    for name, rows in [('windows', windows), ('trials', trials_out), ('pairs', pair_rows)]:
        pd.DataFrame(rows).to_csv(HERE / f'predictions/{subject}_{name}.csv', index=False)
    write_json(HERE / f'selections/{subject}.json', audits)
    stats = {'direction_counts': dict(Counter(r['direction'] for r in audits)), 'reverse_ratio': sum(r['direction'] == 'reverse' for r in audits) / 16, 'candidate_counts': dict(Counter(str(r['candidate_index']) for r in audits)), 'weight_counts': dict(Counter(str(r['csp_weight']) for r in audits)), 'cached_train_subsets': len(cache.values), 'cache_hits': cache.hits, 'baseline': alignment, 'source_sha256': sha(source), 'source': str(source), 'elapsed_seconds': time.perf_counter() - start, 'physical_trial_disjoint': True, 'outer_validation_labels_used_for_selection': False, 'public_labels_used': False}
    write_json(HERE / f'input_metadata/{subject}.json', stats)
    print(f'{subject} complete {stats["elapsed_seconds"]:.1f}s: ' + str([(r['method'], round(r['window_auc'], 4), round(r['window_f1'], 4)) for r in summary]), flush=True)
    return summary, pair_results, stats


def report(rows, stats, elapsed):
    lines = ['# 实验205：剩余被试AUC优先首轮', '', '## 协议与边界', '', '仅运行预定7人。每人16个外层正负物理trial pair；训练6个trial，query 2个trial。内层9个pair，每个训练trial预测3次，先平均窗口预测再求trial均值，以trial AUC、固定0.5 F1、候选顺序依次决胜（参照141）。', '144个基础候选 × raw/reverse = 288；lambda=0/.1/.3，components=2/4（沿用122，每端组件数，实际4/8滤波器），blocks=all/band_2/band_3/band_4，LinearSVC/LogisticRegression，C=.1/1/10。先选择CSP，再仅在相同内层OOF上选择CSP权重.25/.5/.75。', '057固定为plus_pe_fixed002，不根据本轮效果挑选重复。057模型训练排除整名被试；保留其无标签整被试zscore，存在transductive covariate访问，非纯训练trial归一化。行对齐依据完整缓存/参考序列、原始MAT哈希和已读取数据加载器的trial内连续窗口顺序；CSV本身不含独立窗口ID，此为明确的顺序来源假设。', '每窗口外层预测4次后平均；trial为5窗口平均。pair mean为16个pair指标算术平均，不与pooled混用。quota4仅在8 trial汇总使用，不能等同固定阈值F1。raw/reverse控制仅诊断同一内层选中候选，不是事后选方向。', '外层query标签仅用于既定分层pair构造和最终指标，不参与模型/方向/权重选择。未加载公开数组或公开标签。缓存仅按训练trial集合复用拟合，不跨被试；所有训练与验证物理trial交集为空。SciPy/小样本sklearn CPU单线程，未使用GPU，避免额外数据迁移。', '', '## 全部窗口与trial指标', '', '|被试|方法|窗口AUC|窗口F1|trial AUC|trial F1|quota4 F1|pair mean窗口AUC/F1|', '|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f'|{r["subject"]}|{r["method"]}|{r["window_auc"]:.6f}|{r["window_f1"]:.6f}|{r["trial_auc"]:.6f}|{r["trial_f1"]:.6f}|{r["quota4_f1"]:.6f}|{r["pair_mean_window_auc"]:.6f}/{r["pair_mean_window_f1"]:.6f}|')
    lines += ['', '## 每人最佳主要方法（事后描述，不作为新选择器）', '', '|被试|baseline AUC/F1|最佳主要方法|AUC/F1|AUC增益|AUC≥.95|reverse占比|融合同时改善窗口/trial AUC（相对CSP）|', '|---|---|---|---|---|---|---|---|']
    for subject in CFG['subjects']:
        rr = [r for r in rows if r['subject'] == subject and not r['method'].endswith('_control')]
        best = max(rr, key=lambda r: r['window_auc'])
        b = next((r for r in rr if r['method'] == 'baseline_057_plus_pe'), None)
        c = next(r for r in rr if r['method'] == 'csp')
        f = next((r for r in rr if r['method'] == 'csp_plus_pe'), None)
        baseline_text = f'{b["window_auc"]:.6f}/{b["window_f1"]:.6f}' if b else '不可对齐'
        delta = f'{best["window_auc"] - b["window_auc"]:+.6f}' if b else 'NA'
        improvement = bool(f and f['window_auc'] > c['window_auc'] and f['trial_auc'] > c['trial_auc'])
        lines.append(f'|{subject}|{baseline_text}|{best["method"]}|{best["window_auc"]:.6f}/{best["window_f1"]:.6f}|{delta}|{best["window_auc"] >= .95}|{stats[subject]["reverse_ratio"]:.1%}|{improvement}|')
        if b is None:
            lines.append(f'\n{subject}基线/融合缺失原因：{stats[subject]["baseline"]["reason"]}\n')
    lines += ['', '## 扩展判断与局限', 'AUC≥0.95为开发目标；F1仅同步观察，不能将高AUC但低F1称为完整达标。']
    for method in ('csp', 'csp_plus_pe'):
        rr = [r for r in rows if r['method'] == method and r['auc_gain_vs_baseline'] is not None]
        improved = sum(r['auc_gain_vs_baseline'] > 0 for r in rr)
        regressed = [r['subject'] for r in rr if r['auc_gain_vs_baseline'] < 0]
        lines.append(f'- {method}：{improved}/7人窗口AUC改善；退化被试：{regressed}。保护个体未在本轮额外运行且“明显破坏”无预定数值界限，不能据此宣布通过回归保护或自动扩展。')
    for s in ('DEP03', 'DEP17'):
        lines.append(f'- {s} reverse选择{stats[s]["direction_counts"].get("reverse", 0)}/16 pair；pair相互重叠且未运行独立重复，不据此宣称稳定可泛化方向。')
    lines += ['本结果为有标签被试内开发OOF诊断/上限，不是公开测试成绩；不得把095 adaptive、单人CSP或事后最佳方法转化为公开集无标签方案。单人8 trial，未进行独立重复或推断显著性。', '所有有限值、行数、每窗4次预测、外层/内层物理trial隔离在计算阶段断言；来源/脚本/配置SHA256见input_metadata，完整候选分数及方向/权重见selections。未修改历史文件，也未更新目录外总进度文件（按本任务禁止修改历史文件的更严格约束）；本报告作为205进度记录。', f'总耗时：{elapsed:.2f}秒。']
    (HERE / '实验报告.md').write_text('\n'.join(lines) + '\n', encoding='utf8')


def main():
    started = time.perf_counter()
    for name in ('predictions', 'input_metadata', 'selections'):
        (HERE / name).mkdir(exist_ok=True)
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), HERE / 'config.json', TEMPLATE, ROOT / 'src/141_HC16_csp_transfer_pairs/run_experiment.py', ROOT / '数据/数据集说明文档.md', ROOT / '.trae/documents/剩余被试AUC优先突破实验计划.md']}
    write_json(HERE / 'input_metadata/freeze.json', frozen)
    write_json(HERE / 'selections/candidates.json', [{'candidate_index': i, 'candidate': c} for i, c in enumerate(CANDIDATES)])
    rows, pairs, stats = [], [], {}
    with threadpool_limits(limits=CFG['threads']):
        for number, subject in enumerate(CFG['subjects'], 1):
            r, p, s = run_subject(subject, number)
            rows.extend(r)
            pairs.extend(p)
            stats[subject] = s
    assert list(stats) == CFG['subjects'] and len(stats) == 7
    assert all(sha(ROOT / p) == value for p, value in frozen.items())
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(pairs).to_csv(HERE / 'predictions/pair_metrics.csv', index=False)
    elapsed = time.perf_counter() - started
    result = {'experiment': HERE.name, 'summary': rows, 'pair_metrics': pairs, 'subjects': stats, 'candidate_count': len(CANDIDATES), 'candidate_direction_count': len(CANDIDATES) * 2, 'elapsed_seconds': elapsed, 'outer_validation_labels_used_for_selection': False, 'public_labels_used': False, 'all_physical_trial_splits_disjoint': True, 'all_predictions_finite': True, 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}}
    write_json(HERE / 'metrics.json', result)
    report(rows, stats, elapsed)
    outputs = [p for p in HERE.rglob('*') if p.is_file() and p.name != 'output_sha256.json']
    write_json(HERE / 'input_metadata/output_sha256.json', {str(p.relative_to(HERE)): sha(p) for p in outputs})
    print(f'COMPLETE 7/7; {elapsed:.1f}s; all finite / split assertions passed', flush=True)


if __name__ == '__main__':
    main()
