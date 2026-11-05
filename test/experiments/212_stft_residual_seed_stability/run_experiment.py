import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE = (HERE / CFG['stft_cache']).resolve()
REF_PATH = (HERE / CFG['fixed_outer_reference']).resolve()
BASE_PATH = (HERE / CFG['baseline_oof']).resolve()
KEYS = ['subject_id', 'health_group', 'physical_trial_index', 'window_index', 'y_true', 'fold']


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def metric(y, p):
    pred = p >= CFG['threshold']; tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'auc': float(roc_auc_score(y, p)), 'f1': float(f1_score(y, pred, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'threshold': CFG['threshold'],
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


class TinySpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(30, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.Conv2d(16, 32, 1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(32, 1)
    def forward(self, x): return self.head(self.features(x).flatten(1)).squeeze(1)


def fit_scaler(x):
    mean = x.mean(axis=(0, 2, 3), keepdims=True).astype('float32')
    std = np.maximum(x.std(axis=(0, 2, 3), keepdims=True), 1e-6).astype('float32')
    return mean, std


def train_predict(xtr, ytr, xva, seed, fold):
    seed_all(seed); model = TinySpectrogramCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])
    xt = torch.from_numpy(xtr).to(DEVICE); yt = torch.from_numpy(ytr.astype('float32')).to(DEVICE); xv = torch.from_numpy(xva).to(DEVICE)
    rng = np.random.default_rng(seed); losses = []
    for epoch in range(CFG['epochs']):
        order = rng.permutation(len(ytr)); total = 0.0; model.train()
        for start in range(0, len(order), CFG['batch_size']):
            ix = order[start:start + CFG['batch_size']]; opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xt[ix]), yt[ix]); loss.backward(); opt.step(); total += float(loss.item()) * len(ix)
        losses.append(total / len(ytr)); print(f'seed={seed} fold={fold+1}/5 epoch={epoch+1}/{CFG["epochs"]} loss={losses[-1]:.6f}', flush=True)
    model.eval()
    with torch.no_grad(): p = torch.sigmoid(model(xv)).cpu().numpy()
    return p, losses


def details(name, pred, ref, groups, folds, subjects):
    trial = ref.assign(prob_positive=pred).groupby(['subject_id', 'health_group', 'physical_trial_index', 'fold'], as_index=False).agg(y_true=('y_true', 'first'), prob_positive=('prob_positive', 'mean'))
    y = ref.y_true.to_numpy(int); row = {'method': name, **metric(y, pred)}
    for g in ('HC', 'DEP'): row.update({f'{g.lower()}_{k}': v for k, v in metric(y[groups == g], pred[groups == g]).items()})
    row.update({f'window_{k}': v for k, v in metric(y, pred).items()}); row.update({f'trial_{k}': v for k, v in metric(trial.y_true.to_numpy(), trial.prob_positive.to_numpy()).items()})
    fm = [metric(y[folds == f], pred[folds == f]) for f in range(5)]
    row['worst_fold_auc'] = min(x['auc'] for x in fm); row['worst_fold_f1'] = min(x['f1'] for x in fm)
    out = ref.assign(prob_positive=pred); out.to_csv(HERE / 'predictions' / f'{name}_oof.csv', index=False); trial.to_csv(HERE / 'predictions' / f'{name}_trials.csv', index=False)
    return row, {'summary': row, 'folds': [{'fold': i, **fm[i]} for i in range(5)], 'subjects': [{'subject_id': s, **metric(y[subjects == s], pred[subjects == s])} for s in np.unique(subjects)]}


def main():
    started = time.perf_counter(); seed_all(CFG['seeds'][0])
    (HERE / 'predictions').mkdir(exist_ok=True); (HERE / 'input_metadata').mkdir(exist_ok=True)
    with np.load(CACHE, allow_pickle=False) as z: data = {k: z[k] for k in z.files}
    ref0 = pd.read_csv(REF_PATH); base = pd.read_csv(BASE_PATH)
    raw_keep = data['subject_ids'] != CFG['excluded_subject']; raw = pd.DataFrame({'subject_id': data['subject_ids'][raw_keep], 'original_trial_id': data['original_trial_ids'][raw_keep]})
    raw['window_index'] = raw.groupby(['subject_id', 'original_trial_id']).cumcount()
    target = ref0[['subject_id', 'original_trial_id']].copy(); target['window_index'] = target.groupby(['subject_id', 'original_trial_id']).cumcount()
    order = target.merge(raw.reset_index(), on=['subject_id', 'original_trial_id', 'window_index'], how='left', sort=False)['index'].to_numpy()
    if len(order) != len(ref0) or np.any(pd.isna(order)): raise RuntimeError('STFT/reference key alignment failed')
    x = data['X_train'][raw_keep][order].astype('float32'); y = data['y_train'][raw_keep][order].astype('int64')
    subjects = data['subject_ids'][raw_keep][order]; groups = data['health_groups'][raw_keep][order]
    if len(y) != CFG['expected_oof_rows']: raise RuntimeError(f'expected {CFG["expected_oof_rows"]}, got {len(y)}')
    ref = ref0.copy(); ref['window_index'] = target.window_index.to_numpy()
    ref['physical_trial_index'] = ref.groupby(['subject_id', 'original_trial_id']).cumcount() // 5 + (ref['original_trial_id'] - 1) * 2
    ref = ref[['subject_id', 'health_group', 'physical_trial_index', 'window_index', 'y_true', 'fold']]
    expected = pd.DataFrame({'subject_id': subjects, 'health_group': groups, 'physical_trial_index': ref.physical_trial_index, 'window_index': target.window_index.to_numpy(), 'y_true': y, 'fold': ref.fold})
    if not ref[KEYS].equals(expected[KEYS]): raise RuntimeError('canonical metadata mismatch')
    base = base.copy(); base['window_index'] = target.window_index.to_numpy(); base['physical_trial_index'] = base.groupby(['subject_id', 'original_trial_id']).cumcount() // 5 + (base['original_trial_id'] - 1) * 2
    if not base[KEYS].equals(ref[KEYS]): raise RuntimeError('baseline metadata mismatch')
    folds = ref.fold.to_numpy(int); baseline = base.prob_positive.to_numpy(float)
    if set(folds) != set(range(5)): raise RuntimeError('fixed reference must contain folds 0..4')
    for f in range(5):
        if set(subjects[folds == f]) & set(subjects[folds != f]): raise RuntimeError('subject leakage')
    if ref.groupby(['subject_id', 'physical_trial_index']).size().nunique() != 1 or ref.groupby(['subject_id', 'physical_trial_index']).size().iloc[0] != 5: raise RuntimeError('trial window count mismatch')
    ref.to_csv(HERE / 'input_metadata/canonical_keys.csv', index=False)
    print(f'212 starting device={DEVICE} seeds={CFG["seeds"]} rows={len(y)}', flush=True)
    seed_preds = {}; audits = []
    for seed in CFG['seeds']:
        pred = np.full(len(y), np.nan)
        for fold in range(5):
            tr, va = folds != fold, folds == fold; mean, std = fit_scaler(x[tr]); tx = (x[tr] - mean) / std; vx = (x[va] - mean) / std
            print(f'seed={seed} fold={fold+1}/5 train_subjects={len(np.unique(subjects[tr]))} valid_subjects={len(np.unique(subjects[va]))}', flush=True)
            p, losses = train_predict(tx, y[tr], vx, seed + fold, fold); pred[va] = p
            audits.append({'seed': seed, 'fold': fold, 'train_subjects': int(np.unique(subjects[tr]).size), 'valid_subjects': int(np.unique(subjects[va]).size), 'losses': losses})
            print(f'seed={seed} fold={fold+1}/5 complete', flush=True)
        if not np.isfinite(pred).all(): raise RuntimeError('non-finite OOF probability')
        seed_preds[seed] = pred
    key_hashes = {str(seed): hashlib.sha256(pd.util.hash_pandas_object(ref[KEYS], index=False).values.tobytes()).hexdigest() for seed in CFG['seeds']}
    if len(set(key_hashes.values())) != 1: raise RuntimeError('seed canonical key hash mismatch')
    avg_pred = np.mean(np.stack([seed_preds[s] for s in CFG['seeds']]), axis=0)
    runs = {}; rows = []
    for seed in CFG['seeds']:
        p = (1 - CFG['residual_weights'][1]) * baseline + CFG['residual_weights'][1] * seed_preds[seed]
        r, d = details(f'seed_{seed}_residual_0.2', p, ref, groups, folds, subjects); rows.append(r); runs[f'seed_{seed}'] = d
    p = (1 - CFG['residual_weights'][1]) * baseline + CFG['residual_weights'][1] * avg_pred
    r, d = details('repeated_seed_mean_residual_0.2', p, ref, groups, folds, subjects); rows.append(r); runs['repeated_seed_mean'] = d
    r, d = details('baseline_057_plus_pe', baseline, ref, groups, folds, subjects); rows.append(r); runs['baseline'] = d
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(audits).to_json(HERE / 'input_metadata/training_audit.json', orient='records', force_ascii=False, indent=2)
    sources = [CACHE, REF_PATH, BASE_PATH, HERE / 'config.json', HERE / 'run_experiment.py', ROOT / '数据' / '数据集说明文档.md']
    prov = {'inputs': {str(p.relative_to(ROOT)): sha(p) for p in sources}, 'device': str(DEVICE), 'cuda': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'torch': torch.__version__, 'python': platform.python_version(), 'strict_subjectwise_oof': True, 'physical_trial_splits_disjoint': True, 'canonical_key_columns': KEYS, 'canonical_key_hashes': key_hashes, 'canonical_key_validation': 'passed before averaging', 'training_fold_scaler_only': True, 'public_labels_used': False, 'public_prediction': False, 'selection': 'fixed predeclared seeds and residual weight 0.2'}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding='utf8')
    elapsed = time.perf_counter() - started
    result = {'config': CFG, 'runs': runs, 'training_audit': audits, 'provenance': prov, 'elapsed_seconds': elapsed, 'public_auc': None, 'public_f1': None}
    (HERE / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    tbl = ['|方法|窗口AUC|窗口F1|HC AUC/F1|DEP AUC/F1|trial AUC/F1|最差折AUC/F1|', '|---|---:|---:|---:|---:|---:|---:|'] + [f"|{r['method']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['hc_auc']:.6f}/{r['hc_f1']:.6f}|{r['dep_auc']:.6f}/{r['dep_f1']:.6f}|{r['trial_auc']:.6f}/{r['trial_f1']:.6f}|{r['worst_fold_auc']:.6f}/{r['worst_fold_f1']:.6f}|" for r in rows]
    baseline_auc = next(r['auc'] for r in rows if r['method'] == 'baseline_057_plus_pe'); avg_auc = next(r['auc'] for r in rows if r['method'] == 'repeated_seed_mean_residual_0.2')
    report = ['# 实验212：STFT残差重复种子稳定性验证', '', '## 协议', '', '严格复用实验210的STFT缓存、固定subject-wise 5折、HC43排除、2760窗口/552物理trial及057 plus_pe冻结OOF。预声明种子为210、211、212；每个seed每折仅用训练被试拟合STFT均值/std并训练2 epoch，残差权重固定0.2，阈值固定0.5。所有seed先按规范键 subject_id+health_group+physical_trial_index+window_index+y_true+fold 校验键集合、数量、标签和fold完全一致，再按键平均概率。公开标签未读取，未生成公开预测。', '', '## 结果', ''] + tbl + ['', f'重复种子平均窗口AUC={avg_auc:.6f}，相对057冻结基线{avg_auc-baseline_auc:+.6f}；实验210残差0.2窗口AUC=0.805161。公开AUC/F1：未知。', f'设备：{DEVICE}（{prov["cuda"]}）；耗时：{elapsed:.2f}秒。规范键审计、逐折/逐被试指标见metrics.json与input_metadata。']
    (HERE / '实验报告.md').write_text('\n'.join(report) + '\n', encoding='utf8'); print(pd.DataFrame(rows).to_string(index=False), flush=True); print(f'COMPLETE elapsed={elapsed:.1f}s', flush=True)


if __name__ == '__main__': main()
