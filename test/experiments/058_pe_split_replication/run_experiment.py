import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE.parent / '057_multiscale_permutation_entropy'
spec = importlib.util.spec_from_file_location('exp057_for_058', SOURCE / 'run_experiment.py')
exp057 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp057)
base = exp057.base
METHODS = ('control_600', 'plus_pe')
KEYS = ('auc', 'f1', 'balanced_accuracy', 'subject_auc_mean', 'hc_auc', 'dep_auc', 'worst_subject_auc')


def column_sets():
    return {'control_600': np.arange(600), 'plus_pe': np.arange(720)}


def scalars(result):
    return {**{k: result['overall'][k] for k in KEYS[:3]},
            'subject_auc_mean': result['subject_auc_mean'],
            'hc_auc': result['health_groups']['HC']['pooled']['auc'],
            'dep_auc': result['health_groups']['DEP']['pooled']['auc'],
            'worst_subject_auc': min(v['auc'] for v in result['subjects'])}


def fit_fold(A, y, folds, fold, source_config):
    train, valid = folds != fold, folds == fold
    out = {}
    with threadpool_limits(limits=1):
        for method, cols in column_sets().items():
            model = base.make_model(source_config, source_config['outer_model_seed_base'] + fold)
            model.fit(A[train][:, cols], y[train])
            out[method] = model.predict_proba(A[valid][:, cols])[:, 1]
    return fold, out


def main():
    start = time.perf_counter()
    sys.stdout = Tee(HERE / 'run_output.txt')
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    source_config = json.loads((SOURCE / 'config.json').read_text(encoding='utf8'))
    (HERE / 'predictions').mkdir(exist_ok=True)
    (HERE / 'input_metadata').mkdir(exist_ok=True)
    A, AP, ref, spectral, pe = exp057.load_data(source_config)
    y = ref.y_true.to_numpy(); subjects = ref.subject_id.to_numpy(); groups = ref.health_group.to_numpy()
    assert A.shape == (2760, 720) and AP.shape == (80, 720)
    provenance = {
        'python': platform.python_version(), 'config_sha256': base.sha256(HERE / 'config.json'),
        'source_config_sha256': base.sha256(SOURCE / 'config.json'), 'script_sha256': base.sha256(Path(__file__)),
        'source_script_sha256': base.sha256(SOURCE / 'run_experiment.py'),
        'source_inputs': {k: {'path': str((SOURCE / source_config[k]).resolve()), 'sha256': base.sha256((SOURCE / source_config[k]).resolve())}
                          for k in ('spectral_cache', 'temporal_cache', 'fixed_reference', 'pe_cache')},
        'train_shape': list(A.shape), 'public_shape': list(AP.shape), 'public_metrics': False,
        'data_loaded_by': '057.load_data', 'alignment_checked': True, 'subject_isolation': True,
        'new_split_seeds': config['split_seeds']}
    (HERE / 'input_metadata/provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf8')
    runs, rows, positive_counts = {m: {} for m in METHODS}, [], {}
    for seed in config['split_seeds']:
        folds = base.make_subject_folds(subjects, groups, seed)
        base.assert_subject_isolation(subjects, folds)
        pd.DataFrame({'subject_id': subjects, 'health_group': groups, 'original_trial_id': ref.original_trial_id,
                      'y_true': y, 'fold': folds}).to_csv(HERE / 'input_metadata' / f'folds_{seed}.csv', index=False)
        result = Parallel(n_jobs=config['n_jobs'])(delayed(fit_fold)(A, y, folds, f, source_config) for f in range(5))
        pred = {m: np.full(len(y), np.nan) for m in METHODS}
        for fold, values in result:
            pred['control_600'][folds == fold] = values['control_600']
            pred['plus_pe'][folds == fold] = values['plus_pe']
        rr = ref.assign(fold=folds)
        for method in METHODS:
            assert np.isfinite(pred[method]).all()
            output = rr.assign(prob_positive=pred[method])
            output.to_csv(HERE / 'predictions' / f'{method}_split{seed}_oof.csv', index=False)
            summary = base.summarize(y, pred[method], rr, folds)
            runs[method][str(seed)] = summary
            rows.append({'method': method, 'split': str(seed), **scalars(summary)})
        positive_counts[str(seed)] = int(scalars(runs['plus_pe'][str(seed)])['auc'] > scalars(runs['control_600'][str(seed)])['auc'])
    aggregate = {}
    for method in METHODS:
        vals = [scalars(runs[method][str(seed)]) for seed in config['split_seeds']]
        aggregate[method] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'std': float(np.std([v[k] for v in vals], ddof=1)), 'values': [v[k] for v in vals]} for k in KEYS}
        rows.append({'method': method, 'split': 'mean', **{k: aggregate[method][k]['mean'] for k in KEYS}})
    delta = {k: {metric: aggregate[k][metric]['mean'] - aggregate['control_600'][metric]['mean'] for metric in KEYS} for k in ('plus_pe',)}
    d = delta['plus_pe']; accept = bool(d['auc'] >= config['acceptance']['delta_auc_min'] and d['f1'] >= config['acceptance']['delta_f1_min'] and d['dep_auc'] >= config['acceptance']['delta_dep_auc_min'])
    metrics = {'config': config, 'source_config': source_config, 'provenance': provenance, 'runs': runs,
               'aggregate': aggregate, 'deltas_vs_control': delta, 'positive_auc_gain_count': sum(positive_counts.values()),
               'positive_auc_gain_by_split': positive_counts, 'decision_accept': accept, 'elapsed_seconds': time.perf_counter() - start}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    lines = ['# 实验058：PE新分割敏感性检查', '', '同057数据与冻结协议；新被试级分割，不是独立验证。仅训练control_600与plus_pe，不训练公开集。', '', '|方法|AUC均值±SD|F1均值|DEP AUC均值|', '|---|---:|---:|---:|']
    for method in METHODS:
        a = aggregate[method]; lines.append(f"|{method}|{a['auc']['mean']:.6f}±{a['auc']['std']:.6f}|{a['f1']['mean']:.6f}|{a['dep_auc']['mean']:.6f}|")
    lines += ['', f"plus_pe相对control均值：ΔAUC={d['auc']:+.6f}，ΔF1={d['f1']:+.6f}，ΔDEP AUC={d['dep_auc']:+.6f}。", f"AUC正增益次数：{metrics['positive_auc_gain_count']}/5（仅报告，不增加门槛）。", f'接受线：{accept}。真实耗时：{metrics["elapsed_seconds"]:.2f}秒。', '每个新split的两方法OOF、折分配、指标见predictions、input_metadata和metrics.json。']
    (HERE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


class Tee:
    def __init__(self, path):
        self.file = path.open('w', encoding='utf8'); self.stdout = sys.stdout
    def write(self, text):
        self.stdout.write(text); self.file.write(text); self.file.flush()
    def flush(self):
        self.stdout.flush(); self.file.flush()


if __name__ == '__main__':
    main()
