import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('exp067', HERE.parent / '067_adaptation_context_size' / 'run_experiment.py')
base067 = importlib.util.module_from_spec(spec); spec.loader.exec_module(base067)


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(config):
    caches = [base067.load_train_cache((HERE / config[k]).resolve()) for k in ('spectral_cache', 'temporal_cache', 'pe_cache')]
    s, t, pe = caches
    for other in (t, pe):
        for key in base067.TRAIN_KEYS[1:5]: np.testing.assert_array_equal(s[key], other[key])
    keep = s['subject_ids'] != config['excluded_subject']
    mask = np.asarray([str(n).startswith('log_bandpower_') for n in s['feature_names']])
    raw = np.column_stack([s['X_train'][keep][:, mask], t['X_train'][keep], pe['X_train'][keep]])
    ref = pd.DataFrame({'row_index': np.arange(keep.sum()), 'subject': s['subject_ids'][keep], 'group': s['health_groups'][keep], 'trial': s['original_trial_ids'][keep], 'y': s['y_train'][keep]})
    fixed = pd.read_csv((HERE / config['fixed_reference']).resolve())
    for col, old in [('subject', 'subject_id'), ('group', 'health_group'), ('trial', 'original_trial_id'), ('y', 'y_true')]: np.testing.assert_array_equal(ref[col], fixed[old])
    ref['position'] = base067.positions(ref)
    a8 = np.empty_like(raw, dtype=np.float64)
    for i in range(5):
        ix = ref.position.to_numpy() == i
        a8[ix] = base067.normalize(raw[ix], ref.subject.to_numpy()[ix])
    ref = ref.rename(columns={'subject': 'subject_id', 'group': 'health_group', 'trial': 'original_trial_id', 'y': 'y_true'})
    ref = ref[['row_index', 'subject_id', 'health_group', 'original_trial_id', 'position', 'y_true']]
    assert raw.shape == (2760, 720) and a8.shape == raw.shape and ref.subject_id.nunique() == 69
    assert np.isfinite(a8).all() and not config['public_access']
    return a8.astype(np.float32), ref, fixed.fold.to_numpy()


def trial_keys(ref):
    keys = (ref.subject_id.astype(str) + '_T' + (ref.original_trial_id + 4 * ref.y_true).astype(str)).to_numpy()
    counts = pd.Series(keys).value_counts()
    assert counts.eq(5).all() and len(counts) == 69 * 8
    return keys


class Net(torch.nn.Module):
    def __init__(self, width, emb, head):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(720, width), torch.nn.ReLU(), torch.nn.Linear(width, emb))
        self.head = torch.nn.Sequential(torch.nn.Linear(emb, head), torch.nn.ReLU(), torch.nn.Linear(head, 1))
    def forward(self, x):
        z = torch.nn.functional.normalize(self.encoder(x), dim=1)
        return self.head(z).squeeze(1), z


def loss_fn(logits, z, y, key_ids, cfg):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    sim = z @ z.T / cfg['temperature']
    eye = torch.eye(len(y), device=y.device, dtype=torch.bool)
    same = y[:, None].eq(y[None, :]) & ~eye
    key_codes = pd.factorize(np.asarray(key_ids))[0]
    key_tensor = torch.as_tensor(key_codes, device=y.device)
    strong = key_tensor[:, None].eq(key_tensor[None, :]) & ~eye
    positive = same.float() + strong.float()
    positive = positive.clamp(max=1.0)
    valid = positive.sum(1) > 0
    logits_c = sim.masked_fill(eye, -1e9)
    logprob = logits_c - torch.logsumexp(logits_c, dim=1, keepdim=True)
    supcon = -(positive * logprob).sum(1)[valid] / positive.sum(1)[valid]
    supcon = supcon.mean() if valid.any() else bce * 0
    pos = logits[y > 0.5]; neg = logits[y <= 0.5]
    if len(pos) and len(neg):
        ranking = torch.nn.functional.softplus(-(pos[:, None] - neg[None, :])).mean()
    else: ranking = bce * 0
    p = torch.sigmoid(logits)
    tp = (p * y).sum(); fp = (p * (1 - y)).sum(); fn = ((1 - p) * y).sum()
    soft_f1 = 1 - (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    return cfg['lambda_bce'] * bce + cfg['lambda_supcon'] * supcon + cfg['lambda_rank'] * ranking + cfg['lambda_soft_f1'] * soft_f1


def fit_predict(X, y, keys, train, X_eval, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda')
    model = Net(cfg['encoder_width'], cfg['embedding_width'], cfg['head_width']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    groups = {}
    for i, k in enumerate(keys[train]): groups.setdefault(k, []).append(i)
    group_list = list(groups.values()); rng = np.random.default_rng(seed)
    xt = torch.from_numpy(X[train]).to(device); yt = torch.from_numpy(y[train].astype(np.float32)).to(device)
    for epoch in range(cfg['epochs']):
        rng.shuffle(group_list)
        for start in range(0, len(group_list), max(1, cfg['batch_size'] // 5)):
            ids = np.concatenate(group_list[start:start + max(1, cfg['batch_size'] // 5)])
            logits, z = model(xt[ids])
            key_ids = np.asarray(keys[train][ids])
            loss = loss_fn(logits, z, yt[ids], key_ids, cfg)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = []
        xe = torch.from_numpy(X_eval).to(device)
        for start in range(0, len(xe), 2048): out.append(torch.sigmoid(model(xe[start:start + 2048])[0]).cpu().numpy())
    del model, opt, xt, yt
    torch.cuda.empty_cache()
    return np.concatenate(out)


def metrics(y, p):
    pred = p >= 0.5; tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'auc': float(roc_auc_score(y, p)), 'f1': float(f1_score(y, pred, zero_division=0)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'threshold': 0.5, 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def summarize(y, p, ref):
    subjects = []
    for s in np.unique(ref.subject_id):
        m = ref.subject_id.to_numpy() == s
        subjects.append({'subject_id': s, 'health_group': ref.loc[m, 'health_group'].iloc[0], **metrics(y[m], p[m])})
    groups = {}
    for g in ('HC', 'DEP'):
        m = ref.health_group.to_numpy() == g; rows = [r for r in subjects if r['health_group'] == g]
        groups[g] = {'n_subjects': len(rows), 'pooled': metrics(y[m], p[m]), 'subject_auc_mean': float(np.mean([r['auc'] for r in rows]))}
    return {'overall': metrics(y, p), 'subject_auc_mean': float(np.mean([r['auc'] for r in subjects])), 'worst_subject_auc': float(np.min([r['auc'] for r in subjects])), 'health_groups': groups, 'subjects': subjects}


def scalars(r): return {'auc': r['overall']['auc'], 'f1': r['overall']['f1'], 'balanced_accuracy': r['overall']['balanced_accuracy'], 'subject_auc_mean': r['subject_auc_mean'], 'hc_auc': r['health_groups']['HC']['pooled']['auc'], 'dep_auc': r['health_groups']['DEP']['pooled']['auc'], 'worst_subject_auc': r['worst_subject_auc']}


def main():
    start = time.perf_counter(); cfg = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for experiment 089')
    X, ref, fixed = load_data(cfg); y = ref.y_true.to_numpy(); subjects = ref.subject_id.to_numpy(); groups = ref.health_group.to_numpy(); keys = trial_keys(ref)
    splits = [('fixed002', fixed)] + [(f'repeat{s}', base067.base.make_subject_folds(subjects, groups, s)) for s in cfg['repeat_seeds']]
    for _, folds in splits: base067.base.assert_subject_isolation(subjects, folds)
    control = {}; runs = {}; evidence = []
    for name, folds in splits:
        control[name] = pd.read_csv(HERE.parent / '070_train_context_match_new_splits' / 'predictions' / f'{name}_oof.csv').p_train8.to_numpy()
        p = np.full(len(y), np.nan)
        for fold in range(cfg['outer_folds']):
            train, valid = folds != fold, folds == fold
            pv = fit_predict(X, y, keys, train, X[valid], cfg, cfg['model_seed_base'] + fold)
            p[valid] = pv
            evidence.append({'split': name, 'fold': fold, 'train_subjects': np.unique(subjects[train]).tolist(), 'valid_subjects': np.unique(subjects[valid]).tolist(), 'subject_disjoint': bool(not bool(set(subjects[train]) & set(subjects[valid])))})
        assert np.isfinite(p).all() and np.max(np.abs(p - np.clip(p, 0, 1))) == 0
        runs[name] = {'control': summarize(y, control[name], ref), 'treatment': summarize(y, p, ref)}
        frame = ref.copy(); frame['fold'] = folds; frame['prob_positive'] = p; frame.to_csv(HERE / 'predictions' / f'{name}_oof.csv', index=False)
    provenance = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__, 'cuda': torch.cuda.get_device_name(0), 'inputs': {k: sha256((HERE / cfg[k]).resolve()) for k in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')}, 'sources': {str(p): sha256(p) for p in [HERE / 'run_experiment.py', HERE / 'config.json', HERE / 'test_protocol.txt', ROOT / '数据' / '数据集说明文档.md']}, 'public_arrays_loaded': False, 'model_fits': 30, 'train_shape': list(X.shape)}
    (HERE / 'input_metadata' / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf8')
    (HERE / 'input_metadata' / 'fold_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf8')
    aggregate = {}
    for arm in ('control', 'treatment'):
        vals = [scalars(runs[f'repeat{s}'][arm]) for s in cfg['repeat_seeds']]
        aggregate[arm] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in vals[0]}
    d = {k: aggregate['treatment'][k]['mean'] - aggregate['control'][k]['mean'] for k in aggregate['control']}
    checks = {'mean_delta_auc': d['auc'] >= .005, 'mean_delta_f1': d['f1'] >= .005, 'auc_nonnegative': sum(v >= 0 for v in np.array(aggregate['treatment']['auc']['values']) - aggregate['control']['auc']['values']) >= 4, 'f1_nonnegative': sum(v >= 0 for v in np.array(aggregate['treatment']['f1']['values']) - aggregate['control']['f1']['values']) >= 4, 'hc_no_excess_drop': d['hc_auc'] >= -.005, 'dep_no_excess_drop': d['dep_auc'] >= -.005}
    result = {'config': cfg, 'provenance': provenance, 'runs': runs, 'aggregate': aggregate, 'delta_repeat_mean': d, 'acceptance': {'accepted': all(checks.values()), 'checks': checks}, 'fold_evidence': evidence, 'elapsed_seconds': time.perf_counter() - start}
    (HERE / 'metrics.json').write_text(json.dumps(result, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf8')
    lines = ['# 实验089：监督对比 + trial关系 + 排序/soft-F1', '', '|方法|AUC|F1|', '|---|---:|---:|']
    for arm in ('control', 'treatment'): lines.append(f"|{arm}|{aggregate[arm]['auc']['mean']:.6f}|{aggregate[arm]['f1']['mean']:.6f}|")
    lines += ['', '## 五重复均值差异', f"ΔAUC={d['auc']:+.6f}，ΔF1={d['f1']:+.6f}，ΔHC AUC={d['hc_auc']:+.6f}，ΔDEP AUC={d['dep_auc']:+.6f}。", '', '## 接受判断', json.dumps(result['acceptance'], ensure_ascii=False, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), '未达到接受线的配置按协议停止，不进行结果后调参。', '', f"真实运行耗时：{result['elapsed_seconds']:.1f}秒；GPU：{provenance['cuda']}。"]
    (HERE / '实验报告.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({'aggregate': aggregate, 'delta': d, 'acceptance': result['acceptance']}, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), flush=True)

if __name__ == '__main__': main()
