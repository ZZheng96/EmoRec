import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE = (HERE / CFG['stft_cache']).resolve()
REF_PATH = (HERE / CFG['fixed_outer_reference']).resolve()
BASE_PATH = (HERE / CFG['baseline_oof']).resolve()
FIXED_PATH = (HERE / CFG['fixed_residual_oof']).resolve()


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def metric(y, p):
    pred = p >= CFG['threshold']
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'auc': float(roc_auc_score(y, p)), 'f1': float(f1_score(y, pred, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'threshold': CFG['threshold'],
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def make_features(x):
    # Fold-safe feature extraction: no labels and no cross-subject statistics.
    return np.concatenate([x.mean(axis=(2, 3)), x.std(axis=(2, 3))], axis=1).astype('float32')


def fit_predict(kind, xtr, ytr, xva, fold, group):
    if kind == 'hc':
        c = CFG['hc_expert']; estimator = LogisticRegression(C=c['C'], max_iter=c['max_iter'], random_state=CFG['seed'] + fold)
    else:
        c = CFG['dep_expert']; estimator = HistGradientBoostingClassifier(max_iter=c['max_iter'], learning_rate=c['learning_rate'], max_leaf_nodes=c['max_leaf_nodes'], min_samples_leaf=c['min_samples_leaf'], l2_regularization=c['l2_regularization'], early_stopping=False, random_state=CFG['seed'] + fold)
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(xtr, ytr)
    print(f'fold {fold + 1}/{CFG["folds"]} {group} expert fitted train_rows={len(ytr)}', flush=True)
    return model.predict_proba(xva)[:, 1]


def details(name, pred, ref, y, groups, folds, subjects):
    frame = ref.copy(); frame['prob_positive'] = pred; frame['physical_trial_index'] = np.arange(len(frame)) // 5
    trial = frame.groupby(['subject_id', 'health_group', 'physical_trial_index', 'fold'], as_index=False).agg(y_true=('y_true', 'first'), prob_positive=('prob_positive', 'mean'))
    row = {'method': name, **metric(y, pred)}
    for g in ('HC', 'DEP'):
        row.update({f'{g.lower()}_{k}': v for k, v in metric(y[groups == g], pred[groups == g]).items()})
    row.update({f'window_{k}': v for k, v in metric(y, pred).items()})
    row.update({f'trial_{k}': v for k, v in metric(trial.y_true.to_numpy(), trial.prob_positive.to_numpy()).items()})
    fm = [metric(y[folds == f], pred[folds == f]) for f in range(CFG['folds'])]
    row['worst_fold_auc'] = min(q['auc'] for q in fm); row['worst_fold_f1'] = min(q['f1'] for q in fm)
    frame.to_csv(HERE / 'predictions' / f'{name}_oof.csv', index=False)
    trial.to_csv(HERE / 'predictions' / f'{name}_trials.csv', index=False)
    return row, {'summary': row, 'folds': [{'fold': i, **fm[i]} for i in range(CFG['folds'])], 'subjects': [{'subject_id': s, **metric(y[subjects == s], pred[subjects == s])} for s in np.unique(subjects)]}


def main():
    started = time.perf_counter(); seed_all(CFG['seed'])
    with np.load(CACHE, allow_pickle=False) as z: data = {k: z[k] for k in z.files}
    ref = pd.read_csv(REF_PATH)
    keys = ['subject_id', 'health_group', 'original_trial_id', 'fold', 'y_true']
    raw_keep = data['subject_ids'] != CFG['excluded_subject']
    raw = pd.DataFrame({'subject_id': data['subject_ids'][raw_keep], 'original_trial_id': data['original_trial_ids'][raw_keep]})
    raw['occurrence'] = raw.groupby(['subject_id', 'original_trial_id']).cumcount()
    target = ref[['subject_id', 'original_trial_id']].copy(); target['occurrence'] = target.groupby(['subject_id', 'original_trial_id']).cumcount()
    order = target.merge(raw.reset_index(), on=['subject_id', 'original_trial_id', 'occurrence'], how='left', sort=False)['index'].to_numpy()
    if len(order) != len(ref) or np.any(pd.isna(order)): raise RuntimeError('STFT/reference key alignment failed')
    x = data['X_train'][raw_keep][order].astype('float32'); y = data['y_train'][raw_keep][order].astype('int64')
    subjects = data['subject_ids'][raw_keep][order]; groups = data['health_groups'][raw_keep][order]
    if len(y) != CFG['expected_oof_rows']: raise RuntimeError(f'expected {CFG["expected_oof_rows"]}, got {len(y)}')
    expected = pd.DataFrame({'subject_id': subjects, 'health_group': groups, 'original_trial_id': data['original_trial_ids'][raw_keep][order], 'y_true': y})
    for c in ['subject_id', 'health_group', 'original_trial_id', 'y_true']:
        if not np.array_equal(ref[c].to_numpy(), expected[c].to_numpy()): raise RuntimeError(f'fixed reference mismatch: {c}')
    ref = ref[keys].copy(); folds = ref.fold.to_numpy(int)
    if set(folds) != set(range(CFG['folds'])): raise RuntimeError('fixed reference must contain folds 0..4')
    for f in range(CFG['folds']):
        if set(subjects[folds == f]) & set(subjects[folds != f]): raise RuntimeError('subject leakage')
    trial_keys = list(zip(subjects.tolist(), data['original_trial_ids'][raw_keep][order].tolist()))
    for f in range(CFG['folds']):
        if set(trial_keys[i] for i in np.flatnonzero(folds == f)) & set(trial_keys[i] for i in np.flatnonzero(folds != f)): raise RuntimeError('physical trial leakage')
    base = pd.read_csv(BASE_PATH); fixed = pd.read_csv(FIXED_PATH)
    for other in (base, fixed):
        if not np.array_equal(ref[keys].to_numpy(), other[keys].to_numpy()): raise RuntimeError('frozen baseline metadata mismatch')
    X = make_features(x); P = {'hc_expert': np.full(len(y), np.nan), 'dep_expert': np.full(len(y), np.nan), 'hc_dep_expert_fusion': np.full(len(y), np.nan)}
    audits = []
    print(f'211 starting device={DEVICE} cache={CACHE.name} rows={len(y)}', flush=True)
    for fold in range(CFG['folds']):
        tr, va = folds != fold, folds == fold
        ph = np.full(va.sum(), np.nan); pdp = np.full(va.sum(), np.nan)
        for g, kind, out in [('HC', 'hc', ph), ('DEP', 'dep', pdp)]:
            ti, vi = tr & (groups == g), va & (groups == g)
            out[groups[va] == g] = fit_predict(kind, X[ti], y[ti], X[vi], fold, g)
        P['hc_expert'][va] = ph; P['dep_expert'][va] = pdp
        fused = np.where(groups[va] == 'HC', ph, pdp); P['hc_dep_expert_fusion'][va] = fused
        audits.append({'fold': fold, 'train_subjects': sorted(np.unique(subjects[tr]).tolist()), 'valid_subjects': sorted(np.unique(subjects[va]).tolist()), 'hc_train_rows': int((tr & (groups == 'HC')).sum()), 'dep_train_rows': int((tr & (groups == 'DEP')).sum()), 'subject_intersection': 0, 'physical_trial_intersection': 0})
        print(f'fold {fold + 1}/{CFG["folds"]} complete train_subjects={len(np.unique(subjects[tr]))} valid_subjects={len(np.unique(subjects[va]))}', flush=True)
    P['baseline_057_plus_pe'] = base.prob_positive.to_numpy(float)
    P['fixed_residual_stft_0.2'] = fixed.prob_positive.to_numpy(float)
    P['hc_expert'] = np.where(groups == 'HC', P['hc_expert'], P['baseline_057_plus_pe'])
    P['dep_expert'] = np.where(groups == 'DEP', P['dep_expert'], P['baseline_057_plus_pe'])
    rows = []; runs = {}
    for name, p in P.items():
        if not np.isfinite(p).all(): raise RuntimeError(f'nonfinite predictions: {name}')
        r, d = details(name, p, ref, y, groups, folds, subjects); rows.append(r); runs[name] = d
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    ref.to_csv(HERE / 'input_metadata/folds.csv', index=False)
    (HERE / 'input_metadata/training_audit.json').write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding='utf8')
    sources = [CACHE, REF_PATH, BASE_PATH, FIXED_PATH, HERE / 'config.json', HERE / 'run_experiment.py', ROOT / '数据' / '数据集说明文档.md']
    prov = {'inputs': {str(p.relative_to(ROOT)): sha(p) for p in sources}, 'device': str(DEVICE), 'cuda': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'torch': torch.__version__, 'python': platform.python_version(), 'strict_subjectwise_oof': True, 'physical_trial_splits_disjoint': True, 'training_fold_only_group_experts': True, 'public_labels_used': False, 'public_prediction': False, 'selection': 'HC LogisticRegression and DEP HistGradientBoosting predeclared in config; no validation-label selection'}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding='utf8')
    elapsed = time.perf_counter() - started
    (HERE / 'metrics.json').write_text(json.dumps({'config': CFG, 'runs': runs, 'training_audit': audits, 'provenance': prov, 'elapsed_seconds': elapsed, 'public_auc': None, 'public_f1': None}, ensure_ascii=False, indent=2), encoding='utf8')
    tbl = ['|方法|窗口AUC|窗口F1|HC AUC/F1|DEP AUC/F1|trial AUC/F1|最差折AUC/F1|', '|---|---:|---:|---:|---:|---:|---:|'] + [f"|{r['method']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['hc_auc']:.6f}/{r['hc_f1']:.6f}|{r['dep_auc']:.6f}/{r['dep_f1']:.6f}|{r['trial_auc']:.6f}/{r['trial_f1']:.6f}|{r['worst_fold_auc']:.6f}/{r['worst_fold_f1']:.6f}|" for r in rows]
    baseline_auc = next(r['auc'] for r in rows if r['method'] == 'fixed_residual_stft_0.2'); best = max(rows, key=lambda q: q['auc'])
    report = ['# 实验211：已知健康组标签条件下的HC/DEP专家融合', '', '## 协议', '', '严格subject-wise固定5折，复用实验210 STFT缓存、fixed_outer_reference和057 plus_pe OOF。每个被试只进入一个外层折，物理trial不跨折。HC专家为预先声明的训练折内LogisticRegression，DEP专家为预先声明的训练折内HistGradientBoosting；每个专家只使用外层训练被试中对应健康组样本。验证标签不参与模型选择、权重或阈值选择，固定阈值为0.5。', '', '## 结果', ''] + tbl + ['', f'固定实验210 fixed_residual_stft_0.2窗口AUC={baseline_auc:.6f}；本轮最高窗口AUC为{best["method"]}={best["auc"]:.6f}，相对0.805161（精确值0.8051606805）差异={best["auc"] - 0.8051606805293006:+.6f}。', '这是“已知组标签条件性结果”：实际公开测试标签未知，因此不生成公开预测；公开AUC/F1未知。逐折、逐被试和输入哈希见metrics.json及input_metadata。', f'训练设备：{DEVICE}（{prov["cuda"]}）；耗时：{elapsed:.2f}秒。']
    (HERE / '实验报告.md').write_text('\n'.join(report) + '\n', encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False), flush=True); print(f'COMPLETE elapsed={elapsed:.1f}s', flush=True)


if __name__ == '__main__': main()
