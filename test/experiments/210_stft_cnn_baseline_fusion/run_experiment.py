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
        losses.append(total / len(ytr)); print(f'fold {fold+1}/5 epoch {epoch+1}/{CFG["epochs"]} loss={losses[-1]:.6f}', flush=True)
    model.eval()
    with torch.no_grad(): p = torch.sigmoid(model(xv)).cpu().numpy()
    return p, losses


def details(name, pred, ref, y, groups, folds, subjects):
    trial = ref.assign(physical_trial_index=np.arange(len(y)) // 5, prob_positive=pred).groupby(
        ['subject_id', 'health_group', 'physical_trial_index', 'fold'], as_index=False).agg(y_true=('y_true', 'first'), prob_positive=('prob_positive', 'mean'))
    row = {'method': name, **metric(y, pred)}
    for g in ('HC', 'DEP'): row.update({f'{g.lower()}_{k}': v for k, v in metric(y[groups == g], pred[groups == g]).items()})
    row.update({f'window_{k}': v for k, v in metric(y, pred).items()}); row.update({f'trial_{k}': v for k, v in metric(trial.y_true.to_numpy(), trial.prob_positive.to_numpy()).items()})
    fm = [metric(y[folds == f], pred[folds == f]) for f in range(5)]
    row['worst_fold_auc'] = min(x['auc'] for x in fm); row['worst_fold_f1'] = min(x['f1'] for x in fm)
    out = ref.assign(prob_positive=pred); out.to_csv(HERE / 'predictions' / f'{name}_oof.csv', index=False); trial.to_csv(HERE / 'predictions' / f'{name}_trials.csv', index=False)
    return row, {'summary': row, 'folds': [{'fold': i, **fm[i]} for i in range(5)], 'subjects': [{'subject_id': s, **metric(y[subjects == s], pred[subjects == s])} for s in np.unique(subjects)]}


def main():
    started = time.perf_counter(); seed_all(CFG['seed'])
    (HERE / 'predictions').mkdir(exist_ok=True); (HERE / 'input_metadata').mkdir(exist_ok=True)
    with np.load(CACHE, allow_pickle=False) as z: data = {k: z[k] for k in z.files}
    ref = pd.read_csv(REF_PATH); keys = ['subject_id', 'health_group', 'original_trial_id', 'fold', 'y_true']
    raw_keep = data['subject_ids'] != CFG['excluded_subject']; raw = pd.DataFrame({'subject_id': data['subject_ids'][raw_keep], 'original_trial_id': data['original_trial_ids'][raw_keep]})
    raw['occurrence'] = raw.groupby(['subject_id', 'original_trial_id']).cumcount()
    target = ref[['subject_id', 'original_trial_id']].copy(); target['occurrence'] = target.groupby(['subject_id', 'original_trial_id']).cumcount()
    order = target.merge(raw.reset_index(), on=['subject_id', 'original_trial_id', 'occurrence'], how='left', sort=False)['index'].to_numpy()
    if len(order) != len(ref) or np.any(pd.isna(order)): raise RuntimeError('STFT/reference key alignment failed')
    x = data['X_train'][raw_keep][order].astype('float32'); y = data['y_train'][raw_keep][order].astype('int64')
    subjects = data['subject_ids'][raw_keep][order]; groups = data['health_groups'][raw_keep][order]
    original_trials = data['original_trial_ids'][raw_keep][order]
    if len(y) != CFG['expected_oof_rows']: raise RuntimeError(f'expected {CFG["expected_oof_rows"]}, got {len(y)}')
    expected = pd.DataFrame({'subject_id': subjects, 'health_group': groups, 'original_trial_id': original_trials, 'y_true': y})
    for c in ['subject_id', 'health_group', 'original_trial_id', 'y_true']:
        if not np.array_equal(ref[c].to_numpy(), expected[c].to_numpy()): raise RuntimeError(f'fixed reference mismatch: {c}')
    ref = ref[keys].copy(); folds = ref.fold.to_numpy(int)
    if set(folds) != set(range(5)): raise RuntimeError('fixed reference must contain folds 0..4')
    for f in range(5):
        if set(subjects[folds == f]) & set(subjects[folds != f]): raise RuntimeError('subject leakage')
    trial_index = np.arange(len(y)) // 5
    if len(set(zip(subjects[folds == 0], trial_index[folds == 0])) & set(zip(subjects[folds != 0], trial_index[folds != 0]))): raise RuntimeError('trial leakage')
    base = pd.read_csv(BASE_PATH)
    if not np.array_equal(ref[keys].to_numpy(), base[keys].to_numpy()): raise RuntimeError('baseline metadata mismatch')
    baseline = base.prob_positive.to_numpy(float); pred = np.full(len(y), np.nan); audits = []
    print(f'210 starting device={DEVICE} cache={CACHE.name} rows={len(y)}', flush=True)
    for fold in range(5):
        tr, va = folds != fold, folds == fold; mean, std = fit_scaler(x[tr]); tx = (x[tr] - mean) / std; vx = (x[va] - mean) / std
        print(f'fold {fold+1}/5 train_subjects={len(np.unique(subjects[tr]))} valid_subjects={len(np.unique(subjects[va]))}', flush=True)
        p, losses = train_predict(tx, y[tr], vx, CFG['seed'] + fold, fold); pred[va] = p
        audits.append({'fold': fold, 'train_subjects': int(np.unique(subjects[tr]).size), 'valid_subjects': int(np.unique(subjects[va]).size), 'mean': mean.reshape(-1).tolist(), 'std': std.reshape(-1).tolist(), 'losses': losses})
        print(f'fold {fold+1}/5 complete', flush=True)
    rows = []; runs = {}
    for name, p in [('stft_cnn', pred), ('baseline_057_plus_pe', baseline)]:
        r, d = details(name, p, ref, y, groups, folds, subjects); rows.append(r); runs[name] = d
    for w in CFG['residual_weights']:
        p = (1 - w) * baseline + w * pred; name = f'fixed_residual_stft_{w:.1f}'; r, d = details(name, p, ref, y, groups, folds, subjects); rows.append(r); runs[name] = d
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False); ref.to_csv(HERE / 'input_metadata/folds.csv', index=False); pd.DataFrame(audits).to_json(HERE / 'input_metadata/training_audit.json', orient='records', force_ascii=False, indent=2)
    sources = [CACHE, REF_PATH, BASE_PATH, HERE / 'config.json', HERE / 'run_experiment.py', ROOT / '数据' / '数据集说明文档.md']
    prov = {'inputs': {str(p.relative_to(ROOT)): sha(p) for p in sources}, 'device': str(DEVICE), 'cuda': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'torch': torch.__version__, 'python': platform.python_version(), 'strict_subjectwise_oof': True, 'physical_trial_splits_disjoint': True, 'training_fold_scaler_only': True, 'public_labels_used': False, 'public_prediction': False, 'selection': 'fixed epochs and predeclared residual weights; no outer-label tuning'}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding='utf8')
    elapsed = time.perf_counter() - started; result = {'config': CFG, 'runs': runs, 'training_audit': audits, 'provenance': prov, 'elapsed_seconds': elapsed, 'public_auc': None, 'public_f1': None}
    (HERE / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    tbl = ['|方法|窗口AUC|窗口F1|HC AUC/F1|DEP AUC/F1|trial AUC/F1|最差折AUC/F1|', '|---|---:|---:|---:|---:|---:|---:|'] + [f"|{r['method']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['hc_auc']:.6f}/{r['hc_f1']:.6f}|{r['dep_auc']:.6f}/{r['dep_f1']:.6f}|{r['trial_auc']:.6f}/{r['trial_f1']:.6f}|{r['worst_fold_auc']:.6f}/{r['worst_fold_f1']:.6f}|" for r in rows]
    best = max(rows, key=lambda q: q['auc']); baseline_auc = next(r['auc'] for r in rows if r['method'] == 'baseline_057_plus_pe')
    report = ['# 实验210：STFT二维CNN与冻结057 plus_pe基线融合', '', '## 协议', '', '复用047的stft_windows.npz；排除HC43，使用fixed_outer_reference的2760行固定5折。每个被试只进入一个折，物理trial的5个窗口不跨折。每折仅用训练折拟合30通道STFT均值/std，CNN固定2 epoch、AdamW，无外层验证标签调参。冻结baseline来自057 plus_pe_fixed002_oof，并逐列校验metadata。公开标签未读取，因此公开AUC/F1未知。', '', '## 结果', ''] + tbl + ['', f'冻结057 plus_pe窗口AUC={baseline_auc:.6f}；本轮最高窗口AUC为{best["method"]}={best["auc"]:.6f}，相对冻结基线{best["auc"] - baseline_auc:+.6f}。与0.799903比较：{"超过" if best["auc"] > 0.799903 else "未超过"}。', f'训练设备：{DEVICE}（{prov["cuda"]}）；耗时：{elapsed:.2f}秒。逐折/逐被试指标和审计见metrics.json、input_metadata。']
    (HERE / '实验报告.md').write_text('\n'.join(report) + '\n', encoding='utf8'); print(pd.DataFrame(rows).to_string(index=False), flush=True); print(f'COMPLETE elapsed={elapsed:.1f}s', flush=True)


if __name__ == '__main__': main()
