import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from emo_rec.normalization import normalize_by_subject

CFG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def metric(y, p):
    return {'auc': float(roc_auc_score(y, p)), 'f1': float(f1_score(y, p >= CFG['threshold'], zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, p >= CFG['threshold']))}


class BlockMLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        h = CFG['hidden']; self.net = nn.Sequential(nn.Linear(n, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


class FeatureCNN(nn.Module):
    def __init__(self, channels, width):
        super().__init__(); h = CFG['hidden']
        self.net = nn.Sequential(nn.Conv1d(channels, h, 3, padding=1), nn.ReLU(), nn.Conv1d(h, h, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))
    def forward(self, x): return self.head(self.net(x).squeeze(-1)).squeeze(-1)


def fit_predict(kind, xtr, ytr, xva, seed):
    seed_all(seed)
    model = BlockMLP(xtr.shape[1]) if kind == 'mlp' else FeatureCNN(xtr.shape[1], xtr.shape[2])
    model.to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])
    xt = torch.as_tensor(xtr, dtype=torch.float32, device=DEVICE); yt = torch.as_tensor(ytr, dtype=torch.float32, device=DEVICE)
    xv = torch.as_tensor(xva, dtype=torch.float32, device=DEVICE)
    for epoch in range(CFG['epochs']):
        model.train(); order = torch.randperm(len(xt), device=DEVICE); losses = []
        for ix in order.split(CFG['batch_size']):
            opt.zero_grad(set_to_none=True); loss = nn.functional.binary_cross_entropy_with_logits(model(xt[ix]), yt[ix]); loss.backward(); opt.step(); losses.append(loss.item())
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == CFG['epochs']:
            print(f'    {kind} epoch {epoch+1}/{CFG["epochs"]} loss={np.mean(losses):.5f}', flush=True)
    model.eval()
    with torch.no_grad(): p = torch.sigmoid(model(xv)).cpu().numpy()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def load():
    paths = {k: (HERE / CFG[k]).resolve() for k in ('de_cache','spectral_cache','temporal_cache','pe_cache','fixed_reference','baseline_oof')}
    arr = {}
    for k in ('de_cache','spectral_cache','temporal_cache','pe_cache'):
        with np.load(paths[k], allow_pickle=False) as z: arr[k] = {n: z[n] for n in z.files}
    d, s, t, p = [arr[k] for k in ('de_cache','spectral_cache','temporal_cache','pe_cache')]
    keys = ('subject_ids','health_groups','original_trial_ids','y_train')
    for q in (s, t, p):
        for k in keys: assert np.array_equal(d[k], q[k]), f'alignment:{k}'
    keep = d['subject_ids'] != CFG['excluded_subject']; subs = d['subject_ids'][keep]; groups = d['health_groups'][keep]; y = d['y_train'][keep]
    ref = pd.read_csv(paths['fixed_reference']); refkeys = ['subject_id','health_group','original_trial_id','fold','y_true']
    for col, val in [('subject_id',subs),('health_group',groups),('original_trial_id',d['original_trial_ids'][keep]),('y_true',y)]: assert np.array_equal(ref[col].to_numpy(), val), col
    ref = ref[refkeys].copy(); folds = ref.fold.to_numpy()
    z = lambda x: normalize_by_subject(x, subs, 'zscore')
    de = z(d['X_train'][keep]); spectral = z(s['X_train'][keep, :150]); pe = z(p['X_train'][keep])
    plus = np.column_stack([spectral, de, pe])
    cnn = np.concatenate([spectral.reshape(-1,5,30), de.reshape(-1,5,30), pe.reshape(-1,4,30)], axis=1)
    assert plus.shape == (2760, 420) and cnn.shape == (2760, 14, 30)
    return paths, ref, y, subs, groups, folds, plus.astype('float32'), cnn.astype('float32')


def main():
    start = time.perf_counter(); seed_all(CFG['seed'])
    for d in ('predictions','input_metadata'): (HERE / d).mkdir(exist_ok=True)
    paths, ref, y, subs, groups, folds, X_mlp, X_cnn = load()
    print(f'209 starting device={DEVICE} mlp={X_mlp.shape} cnn={X_cnn.shape}', flush=True)
    base = pd.read_csv(paths['baseline_oof']); assert np.array_equal(ref[['subject_id','health_group','original_trial_id','fold','y_true']].to_numpy(), base[['subject_id','health_group','original_trial_id','fold','y_true']].to_numpy())
    P = {'mlp_plus_features': np.full(len(y), np.nan), 'cnn_time_frequency': np.full(len(y), np.nan), 'baseline_057_plus_pe': base.prob_positive.to_numpy()}
    diagnostics = []
    for fold in range(CFG['folds']):
        tr, va = folds != fold, folds == fold
        assert not set(subs[tr]) & set(subs[va])
        trial_index = np.arange(len(y)) // 5
        assert not set(zip(subs[tr], trial_index[tr])) & set(zip(subs[va], trial_index[va]))
        print(f'fold {fold+1}/5 train_subjects={len(set(subs[tr]))} valid_subjects={len(set(subs[va]))}', flush=True)
        for kind, X in (('mlp', X_mlp), ('cnn', X_cnn)):
            tic = time.perf_counter(); out = fit_predict(kind, X[tr], y[tr], X[va], CFG['seed'] + fold * 10 + (0 if kind == 'mlp' else 1))
            name = 'mlp_plus_features' if kind == 'mlp' else 'cnn_time_frequency'; P[name][va] = out
            diagnostics.append({'fold': fold, 'model': name, 'train_subjects': int(len(set(subs[tr]))), 'valid_subjects': int(len(set(subs[va]))), 'seconds': time.perf_counter() - tic})
        print(f'fold {fold+1}/5 complete', flush=True)
    rows=[]; details={}
    trial_index = np.arange(len(y)) // 5
    for name, pred in P.items():
        assert np.isfinite(pred).all()
        frame = ref.assign(prob_positive=pred); trial = frame.assign(physical_trial_index=trial_index).groupby(['subject_id','health_group','physical_trial_index','fold'], as_index=False).agg(y_true=('y_true','first'), prob_positive=('prob_positive','mean'))
        r = {'method': name, **metric(y,pred)}
        for g in ('HC','DEP'): r.update({g.lower()+'_'+k:v for k,v in metric(y[groups==g],pred[groups==g]).items()})
        r.update({'trial_'+k:v for k,v in metric(trial.y_true.to_numpy(),trial.prob_positive.to_numpy()).items()})
        fm = [metric(y[folds==f], pred[folds==f]) for f in range(5)]; r['worst_fold_auc'] = min(x['auc'] for x in fm); r['worst_fold_f1'] = min(x['f1'] for x in fm)
        rows.append(r); details[name] = {'summary':r,'folds':[{'fold':i,**fm[i]} for i in range(5)],'subjects':[{'subject_id':sid,**metric(y[subs==sid],pred[subs==sid])} for sid in np.unique(subs)]}
        frame.to_csv(HERE / 'predictions' / f'{name}_oof.csv', index=False); trial.to_csv(HERE / 'predictions' / f'{name}_trials.csv', index=False)
    basep = P['baseline_057_plus_pe']
    for w in CFG['cnn_weights']:
        pred = (1-w)*basep + w*P['cnn_time_frequency']; name=f'fixed_residual_cnn_{w:.1f}'
        frame=ref.assign(prob_positive=pred); trial=frame.assign(physical_trial_index=trial_index).groupby(['subject_id','health_group','physical_trial_index','fold'],as_index=False).agg(y_true=('y_true','first'),prob_positive=('prob_positive','mean'))
        r={'method':name,**metric(y,pred)}
        for g in ('HC','DEP'): r.update({g.lower()+'_'+k:v for k,v in metric(y[groups==g],pred[groups==g]).items()})
        r.update({'trial_'+k:v for k,v in metric(trial.y_true.to_numpy(),trial.prob_positive.to_numpy()).items()}); fm=[metric(y[folds==f],pred[folds==f]) for f in range(5)]; r['worst_fold_auc']=min(x['auc'] for x in fm); r['worst_fold_f1']=min(x['f1'] for x in fm)
        rows.append(r); details[name]={'summary':r,'folds':[{'fold':i,**fm[i]} for i in range(5)]}; frame.to_csv(HERE/'predictions'/f'{name}_oof.csv',index=False); trial.to_csv(HERE/'predictions'/f'{name}_trials.csv',index=False)
    ref.to_csv(HERE/'input_metadata/folds.csv',index=False); pd.DataFrame(diagnostics).to_csv(HERE/'input_metadata/training_diagnostics.csv',index=False)
    source = list(paths.values()) + [Path(__file__), HERE/'config.json', ROOT/'排行.md', ROOT/'数据/数据集说明文档.md']
    prov={'inputs':{str(q.relative_to(ROOT)):sha(q) for q in source},'device':str(DEVICE),'cuda':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'torch':torch.__version__,'strict_subjectwise_oof':True,'physical_trial_splits_disjoint':True,'public_labels_used':False,'public_prediction':False,'selection':'fixed epochs and predeclared residual weights; no outer validation labels'}
    (HERE/'input_metadata/provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False); elapsed=time.perf_counter()-start
    (HERE/'metrics.json').write_text(json.dumps({'config':CFG,'runs':details,'diagnostics':diagnostics,'provenance':prov,'elapsed_seconds':elapsed,'public_auc':None,'public_f1':None},ensure_ascii=False,indent=2),encoding='utf8')
    tbl=['|方法|AUC|F1|HC AUC/F1|DEP AUC/F1|trial AUC/F1|最差折AUC/F1|','|---|---:|---:|---|---|---|---|']+[f"|{r['method']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['hc_auc']:.6f}/{r['hc_f1']:.6f}|{r['dep_auc']:.6f}/{r['dep_f1']:.6f}|{r['trial_auc']:.6f}/{r['trial_f1']:.6f}|{r['worst_fold_auc']:.6f}/{r['worst_fold_f1']:.6f}|" for r in rows]
    best=max(rows,key=lambda q:q['auc']); lines=['# 实验209：轻量特征CNN与057残差融合','','## 协议','','严格subject-wise固定5折，排除HC43；每被试只在一个验证折，5个窗口物理trial不跨折。输入复用DE/频谱/PE缓存，无重复滤波。A为420维DE+频谱+PE块MLP，B为14×30特征块1D-CNN；固定35 epoch、AdamW，不使用外层验证标签早停/选模。057 plus_pe作为冻结基线，残差权重预先固定为0.1/0.2/0.3。',f'训练设备：{DEVICE}（{prov["cuda"]}）；公开标签未读取，公开AUC/F1为未知。','','## 结果','']+tbl+['',f'本轮最高AUC：{best["method"]}，AUC={best["auc"]:.6f}，F1={best["f1"]:.6f}。仅在明确超过冻结057 AUC 0.799903时才允许公开候选；本轮不自动生成公开候选。','完整逐折、逐被试、trial级指标及输入哈希见metrics.json和input_metadata。',f'总耗时：{elapsed:.2f}秒。']
    (HERE/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8'); print(pd.DataFrame(rows).to_string(index=False),flush=True); print(f'COMPLETE elapsed={elapsed:.1f}s',flush=True)

if __name__ == '__main__': main()
