import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / '070_train_context_match_new_splits'
HELPER = HERE.parent / '087_class_conditional_prototype'
spec = importlib.util.spec_from_file_location('dependency087', HELPER / 'run_experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def configure(seed):
    if not torch.cuda.is_available():
        raise RuntimeError('088 requires CUDA; CPU fallback is forbidden')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)


def group_indices(ref, folds):
    assert 'HC43' not in set(ref.subject)
    e.base.assert_subject_isolation(ref.subject.to_numpy(), folds)
    indices = np.stack([np.asarray(v) for v in ref.groupby(['subject', 'position'], sort=False).indices.values()])
    assert indices.shape == (345, 8)
    np.testing.assert_array_equal(np.sort(indices.ravel()), np.arange(len(ref)))
    assert np.all(folds[indices] == folds[indices[:, :1]])
    return indices, folds[indices[:, 0]]


class WindowNet(nn.Module):
    def __init__(self, features, config, context):
        super().__init__()
        t = config['training']
        self.context = context
        self.phi = nn.Sequential(nn.Linear(features, t['encoder_width']), nn.ReLU(), nn.Linear(t['encoder_width'], t['embedding_width']), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3 * t['embedding_width'], t['head_width']), nn.ReLU(), nn.Linear(t['head_width'], 1))

    def forward(self, x):
        h = self.phi(x)
        c = h.mean(dim=1, keepdim=True).expand_as(h) if self.context else torch.zeros_like(h)
        return self.head(torch.cat([h, c, h - c], dim=-1)).squeeze(-1)


@torch.no_grad()
def predict(model, x):
    model.eval()
    p = torch.sigmoid(model(x))
    assert torch.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def fit_predict(x_train, y_train, x_valid, config, context, seed):
    # No validation labels or metadata enter fitting or inference.
    configure(seed)
    assert x_train.is_cuda and x_valid.is_cuda and y_train.is_cuda
    model = WindowNet(x_train.shape[-1], config, context).cuda()
    t = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
    for _ in range(t['epochs']):
        model.train()
        order = torch.randperm(len(x_train), device='cuda')
        for ix in order.split(t['batch_sets']):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(x_train[ix]), y_train[ix])
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
    p = predict(model, x_valid)
    permutation = torch.randperm(8, device='cuda')
    inverse = torch.argsort(permutation)
    permuted = predict(model, x_valid[:, permutation])[:, inverse]
    error = (p - permuted).abs().max().item()
    assert error <= config['permutation_atol'], error
    return p.cpu().numpy(), error


def check_control(name, ref, folds, historical):
    path = OLD / 'predictions' / f'{name}_oof.csv'
    old = pd.read_csv(path)
    e.validate_metadata(old, ref.assign(fold=folds))
    result = e.scalars(e.summarize(old, 'p_train8'))
    expected = e.scalars(historical['runs'][name]['all_rows']['8'])
    for key in e.KEYS:
        assert abs(result[key] - expected[key]) < 1e-12
    assert np.isfinite(old.p_train8).all() and old.p_train8.between(0, 1).all()
    return old.p_train8.to_numpy()


def aggregate(runs, config):
    result = {}
    for arm in ('A', 'B', 'C', 'C_minus_A', 'C_minus_B'):
        values = []
        for seed in config['repeat_seeds']:
            r = runs[f'repeat{seed}']['all_rows']
            values.append(r[arm] if arm in r else {k: r['C'][k] - r[arm[-1]][k] for k in e.KEYS})
        result[arm] = {k: {'mean': float(np.mean([v[k] for v in values])), 'std': float(np.std([v[k] for v in values], ddof=1)), 'values': [v[k] for v in values]} for k in e.KEYS}
    return result


def main():
    start = time.perf_counter()
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    configure(config['training']['seed_base'])
    assert config['repeat_seeds'] == list(range(2042, 2047)) and not config['public_access']
    assert config['excluded_subject'] == 'HC43' and config['threshold'] == 0.5
    assert config['training']['optimizer'] == 'AdamW' and not config['training']['early_stopping']
    for folder in ('predictions', 'input_metadata'):
        (HERE / folder).mkdir(exist_ok=True)
    data_config = json.loads((OLD / 'config.json').read_text(encoding='utf8'))
    historical = json.loads((OLD / 'metrics.json').read_text(encoding='utf8'))
    history_prov = json.loads((OLD / 'input_metadata/provenance.json').read_text(encoding='utf8'))
    inputs = [(OLD / data_config[k]).resolve() for k in ('spectral_cache', 'temporal_cache', 'pe_cache', 'fixed_reference')]
    for p in inputs:
        assert e.sha256(p) == history_prov['inputs'][str(p)], f'070 input changed: {p}'
    with e.public_guard() as guard:
        raw, ref, fixed = e.load_data(data_config)
    assert set(guard['loaded_paths']) <= {str(p) for p in inputs}
    _, a8 = e.contexts(raw, ref.subject.to_numpy(), ref.position.to_numpy())
    splits = e.splits_for(ref, fixed, config)
    inputs += [OLD / 'predictions' / f'{name}_oof.csv' for name, _ in splits]
    inputs += [OLD / 'metrics.json', OLD / 'input_metadata/provenance.json']
    sources = [HERE / n for n in ('config.json', 'run_experiment.py', 'test_088.py', 'test_protocol.txt')]
    sources += [HELPER / 'run_experiment.py', HELPER / 'config.json']
    sources += [Path(p) for p in history_prov['sources']]
    source_hashes = {str(p.resolve()): e.sha256(p) for p in sources}
    prov = {'inputs': {str(p.resolve()): e.sha256(p) for p in inputs}, 'sources': source_hashes,
            'public_arrays_loaded': False, 'guard_evidence': guard, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0), 'deterministic_algorithms': True, 'tf32': False,
            'python': sys.version, 'numpy': np.__version__, 'raw_shape': list(raw.shape),
            'stage_fits': {'A': 0, 'B': 0, 'C': 0}, 'reused_control_fits': 30,
            'control_verification': '070 input SHA256, full row metadata/folds, recomputed metrics against frozen metrics.json',
            'normalization': '070 a8: label-free zscore within each subject-position set; no cross-subject fitted preprocessing'}
    runs, diagnostics = {}, []
    for split_id, (name, folds) in enumerate(splits):
        indices, set_folds = group_indices(ref, folds)
        frame = ref.assign(fold=folds, p_A=check_control(name, ref, folds, historical), p_B=np.nan, p_C=np.nan)
        ref.assign(fold=folds).to_csv(HERE / 'input_metadata' / f'folds_{name}.csv', index=False)
        x = torch.as_tensor(a8[indices], dtype=torch.float32, device='cuda')
        y = torch.as_tensor(ref.y.to_numpy()[indices], dtype=torch.float32, device='cuda')
        for fold in range(5):
            train, valid = set_folds != fold, set_folds == fold
            assert not set(ref.subject.iloc[indices[train].ravel()]) & set(ref.subject.iloc[indices[valid].ravel()])
            for arm in ('B', 'C'):
                tic = time.perf_counter()
                p, error = fit_predict(x[train], y[train], x[valid], config, arm == 'C', config['training']['seed_base'] + 100 * split_id + fold)
                prov['stage_fits'][arm] += 1
                frame.loc[indices[valid].ravel(), f'p_{arm}'] = p.ravel()
                diagnostics.append({'split': name, 'fold': fold, 'arm': arm, 'train_sets': int(train.sum()), 'valid_sets': int(valid.sum()), 'permutation_max_abs_diff': error, 'seconds': time.perf_counter() - tic})
                print(f'{name} fold={fold} arm={arm} done', flush=True)
        assert np.isfinite(frame[['p_A', 'p_B', 'p_C']]).all().all()
        frame.to_csv(HERE / 'predictions' / f'{name}_oof.csv', index=False)
        runs[name] = {}
        for subset in ['all_rows'] + [str(i) for i in config['positions']]:
            selected = frame if subset == 'all_rows' else frame[frame.position == int(subset)]
            runs[name][subset] = {arm: e.scalars(e.summarize(selected, f'p_{arm}')) for arm in ('A', 'B', 'C')}
    assert prov['stage_fits'] == config['expected_stage_fits']
    agg = aggregate(runs, config)
    decisions = {key: e.acceptance({'all_rows': {'delta_treatment_minus_control': agg[key]}}, config) for key in config['acceptance_comparisons']}
    decision = {'accepted': all(d['accepted'] for d in decisions.values()), 'comparisons': decisions}
    assert all(e.sha256(Path(p)) == h for p, h in source_hashes.items()), 'Preregistered source changed during run'
    metrics = {'runs': runs, 'aggregate': agg, 'acceptance': decision, 'diagnostics': diagnostics, 'stage_fits': prov['stage_fits'], 'elapsed_seconds': time.perf_counter() - start}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    lines = ['# 实验088：8窗置换等变 DeepSets', '', '训练前冻结 config；固定60 epoch，无早停、无外层标签调参。GPU：' + prov['device'], '', 'A=070冻结HGB；B=逐窗MLP [h,0,h]；C=DeepSets [h,c,h-c]。两网络结构、初始化种子及训练日程相同。B仍复用070的无标签a8归一化，但无可学习集合上下文。', '', '## 真实结果', '', '|split|arm|AUC|F1|', '|---|---|---:|---:|']
    for name, r in runs.items():
        for arm, v in r['all_rows'].items():
            lines.append(f"|{name}|{arm}|{v['auc']:.6f}|{v['f1']:.6f}|")
    for arm, v in agg.items():
        lines.append(f"|repeat均值|{arm}|{v['auc']['mean']:.6f}|{v['f1']['mean']:.6f}|")
    lines += ['', '## 冻结接受线', 'C相对A及B均须：repeat平均ΔAUC≥0.005、ΔF1≥0.005；各至少4/5非负；HC/DEP平均AUC降幅不超过0.005。fixed002仅描述，不参与接受。', f"结论：{'达到' if decision['accepted'] else '未达到'}冻结接受线。", '```json', json.dumps(decision, indent=2, ensure_ascii=False), '```', '', f"实际新拟合：{prov['stage_fits']}；A复用30个历史折模型的OOF，不重新拟合。", f"置换最大误差：{max(d['permutation_max_abs_diff'] for d in diagnostics):.9g}。", '验证标签仅用于最终评分；按subject-position整体预测。未访问公开测试数组。重复split并非独立样本，未进行显著性推断。', f"运行耗时：{metrics['elapsed_seconds']:.1f}秒。"]
    (HERE / '实验报告.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    outputs = list((HERE / 'predictions').glob('*.csv')) + list((HERE / 'input_metadata').glob('*.csv')) + [HERE / 'metrics.json', HERE / '实验报告.md']
    prov['outputs'] = {str(p.resolve()): e.sha256(p) for p in outputs}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf8')
    print(json.dumps({'aggregate': agg, 'acceptance': decision}, indent=2), flush=True)


if __name__ == '__main__':
    main()
