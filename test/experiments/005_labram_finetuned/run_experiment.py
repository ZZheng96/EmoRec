import argparse
import gc
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from emo_rec.finetuning import (
    assert_subject_disjoint, make_optimizer, predict_logits, score_probabilities,
    seed_training, train_epoch,
)
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('labram004', HERE.parent / '004_labram_frozen' / 'run_experiment.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
CACHE = HERE / 'cache'


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def configuration():
    config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    inherited = json.loads((HERE / config['reuse_config']).read_text(encoding='utf-8'))
    inherited['extraction'].update(device='cuda', amp=config['training']['amp'])
    config['official'] = inherited['official']
    config['channels'] = inherited['channels']
    config['preprocessing'] = inherited['preprocessing']
    return config, inherited


def metadata():
    frame, baseline = previous.load_baseline()
    if len(frame) != 2760 or frame.subject_id.nunique() != 69 or 'HC43' in set(frame.subject_id):
        raise RuntimeError('Invalid 002 reference protocol')
    assert_subject_disjoint(frame.subject_id, frame.fold)
    if set(frame.fold) != set(range(5)) or not (frame.groupby('subject_id').size() == 40).all():
        raise RuntimeError('Invalid fold or subject sizes')
    return frame, baseline


def prepare_data(frame):
    paths = {p.stem: p for p in (ROOT / '数据' / '训练集').glob('*/*.mat') if p.stem != 'HC43'}
    train_path, public_path = CACHE / 'train.npy', CACHE / 'public.npy'
    signature = previous.sha256_file(previous.BASELINE_DIR / 'predictions' / f'{previous.BASELINE_RUN}_oof.csv')
    marker = CACHE / 'preprocessing.json'
    if not marker.exists():
        train = np.lib.format.open_memmap(train_path, mode='w+', dtype='float32', shape=(2760, 30, 10, 200))
        for subject in frame.subject_id.unique():
            indices = np.flatnonzero(frame.subject_id.to_numpy() == subject)
            segments, labels, trials, _ = previous.preprocess_train_subject(paths[subject])
            if not np.array_equal(labels, frame.y_true.to_numpy()[indices]) or not np.array_equal(trials, frame.original_trial_id.to_numpy()[indices]):
                raise RuntimeError('002 sample order mismatch')
            train[indices] = previous.prepare_model_input(segments).numpy()
            print(f'Preprocessed {subject}', flush=True)
        train.flush()
        public = np.lib.format.open_memmap(public_path, mode='w+', dtype='float32', shape=(80, 30, 10, 200))
        for i in range(10):
            segments, _ = previous.preprocess_public_user(ROOT / '数据' / '公开测试集' / f'P_test{i + 1}.mat')
            public[i * 8:(i + 1) * 8] = previous.prepare_model_input(segments).numpy()
        public.flush()
        save_json(marker, {'baseline_sha256': signature, 'preprocessing_004_sha256': previous.sha256_file(Path(previous.__file__))})
    provenance = json.loads(marker.read_text(encoding='utf-8'))
    if provenance['baseline_sha256'] != signature or provenance['preprocessing_004_sha256'] != previous.sha256_file(Path(previous.__file__)):
        raise RuntimeError('Stale preprocessed cache')
    return np.load(train_path, mmap_mode='r'), np.load(public_path, mmap_mode='r')


def smoke(config, inherited, frame):
    seed_training(2026)
    model, channels, info, device = previous.load_labram(inherited)
    optimizer = make_optimizer(model)
    segments, labels, _, _ = previous.preprocess_train_subject(ROOT / '数据' / '训练集' / '正常人' / 'HC01.mat')
    samples = previous.prepare_model_input(segments).numpy()
    amp = config['training']['amp']
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    batch_size = config['training']['batch_size']
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss = train_epoch(model, channels, samples, labels, np.arange(batch_size), optimizer, scaler, device, batch_size, amp)
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            norm = parameter.grad.float().norm().item()
            if not np.isfinite(norm):
                raise RuntimeError('Non-finite gradient')
            for group in ('head.', 'fc_norm.', 'blocks.'):
                if name.startswith(group):
                    gradients[group] = max(gradients.get(group, 0), norm)
    if not all(gradients.get(group, 0) > 0 for group in ('head.', 'fc_norm.', 'blocks.')):
        raise RuntimeError('Backbone/head/fc_norm gradient check failed')
    logits = predict_logits(model, channels, samples, np.arange(batch_size), device, batch_size, amp)
    public, _ = previous.preprocess_public_user(ROOT / '数据' / '公开测试集' / 'P_test1.mat')
    public_logits = predict_logits(model, channels, previous.prepare_model_input(public).numpy(), np.arange(8), device, batch_size, amp)
    torch.cuda.synchronize()
    result = {'status': 'passed', 'loss': loss, 'gradient_norms': gradients,
              'all_parameters_unfrozen': all(p.requires_grad for p in model.parameters()),
              'train_input_shape': list(samples.shape), 'logits_shape': list(logits.shape),
              'public_logits_shape': list(public_logits.shape), 'subjects_disjoint': True,
              'n_subjects': frame.subject_id.nunique(), 'n_samples': len(frame),
              'seconds': time.perf_counter() - start,
              'peak_cuda_gib': torch.cuda.max_memory_allocated() / 2**30,
              'weight_loading_before_unfreeze': info}
    save_json(HERE / 'smoke_test.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def fit_predict(config, inherited, samples, labels, train_indices, target, target_indices, name, seed):
    output = CACHE / f'{name}_logits.npy'
    if output.exists():
        print(f'Reusing completed {name}', flush=True)
        return np.load(output)
    seed_training(seed)
    model, channels, info, device = previous.load_labram(inherited)
    settings = config['training']
    optimizer = make_optimizer(model, settings['backbone_lr'], settings['head_lr'], settings['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    batch_size = settings['batch_size']
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for epoch in range(1, settings['epochs'] + 1):
        start = time.perf_counter()
        loss = train_epoch(model, channels, samples, labels, train_indices, optimizer, scaler, device, batch_size, settings['amp'])
        torch.cuda.synchronize()
        row = {'fit': name, 'seed': seed, 'epoch': epoch, 'train_loss': loss, 'batch_size': batch_size,
               'n_train': len(train_indices), 'seconds': time.perf_counter() - start,
               'peak_cuda_gib': torch.cuda.max_memory_allocated() / 2**30}
        rows.append(row)
        pd.DataFrame(rows).to_csv(HERE / f'{name}_training_loss.csv', index=False)
        print(json.dumps(row), flush=True)
    logits = predict_logits(model, channels, target, target_indices, device, batch_size, settings['amp'])
    save_json(HERE / f'{name}_provenance.json', {'seed': seed, 'epochs': settings['epochs'],
              'all_parameters_unfrozen': all(p.requires_grad for p in model.parameters()),
              'checkpoint_reloaded': True, 'weight_loading_before_unfreeze': info})
    np.save(output, logits)
    del optimizer, model, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return logits


def evaluate(config, frame, baseline, oof_logits, public_logits):
    subjects = frame.subject_id.to_numpy()
    public_subjects = np.repeat([f'P_test{i}' for i in range(1, 11)], 8)
    oof_scores = score_probabilities(oof_logits, subjects)
    public_scores = score_probabilities(public_logits, public_subjects)
    predictions = HERE / 'predictions'
    predictions.mkdir(exist_ok=True)
    metrics = {'experiment_config': config, 'runtime': previous.runtime_info(),
               'provenance': previous.verify_provenance(config),
               'smoke_test': json.loads((HERE / 'smoke_test.json').read_text(encoding='utf-8')),
               'public_labels_used': False, 'runs': {}}
    summary = []
    y = frame.y_true.to_numpy()
    for condition, probabilities in oof_scores.items():
        name = f'labram_finetuned_{condition}'
        oof = frame.drop(columns=['prob_positive']).copy()
        oof['logit_difference'] = oof_logits[:, 1] - oof_logits[:, 0]
        oof['prob_positive'] = probabilities
        oof.to_csv(predictions / f'{name}_oof.csv', index=False)
        public = pd.DataFrame({'user_id': public_subjects, 'trial_id': np.tile(np.arange(1, 9), 10),
                               'prob. of positive emotion': public_scores[condition]})
        validate_submission(public, [f'P_test{i}' for i in range(1, 11)])
        public.to_csv(predictions / f'{name}_public.csv', index=False)
        fold_metrics = [{'fold': fold, **previous.classification_metrics(y[frame.fold == fold], probabilities[frame.fold == fold])} for fold in range(5)]
        subject_rows = [{'subject_id': subject, 'health_group': frame.health_group[subjects == subject].iloc[0],
                         **previous.classification_metrics(y[subjects == subject], probabilities[subjects == subject])} for subject in np.unique(subjects)]
        result = {'n_subjects': 69, 'n_samples': 2760, 'overall': previous.classification_metrics(y, probabilities),
                  'folds': fold_metrics, 'subject_metrics': subject_rows,
                  'fold_auc_mean': float(np.mean([r['roc_auc'] for r in fold_metrics])),
                  'fold_auc_std': float(np.std([r['roc_auc'] for r in fold_metrics], ddof=1)),
                  'subject_auc_mean': float(np.mean([r['roc_auc'] for r in subject_rows])), 'health_groups': {}}
        for health in ('HC', 'DEP'):
            mask = frame.health_group.to_numpy() == health
            result['health_groups'][health] = {'pooled': previous.classification_metrics(y[mask], probabilities[mask]),
                'subject_auc_mean': float(np.mean([r['roc_auc'] for r in subject_rows if r['health_group'] == health]))}
        previous.add_baseline_deltas(result, baseline)
        metrics['runs'][name] = result
        summary.append({'run_name': name, **result['overall'], 'fold_auc_mean': result['fold_auc_mean'],
                        'subject_auc_mean': result['subject_auc_mean'],
                        'delta_auc_vs_002': result['delta_vs_002_best']['overall_roc_auc']})
    save_json(HERE / 'metrics.json', metrics)
    pd.DataFrame(summary).to_csv(HERE / 'summary.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    CACHE.mkdir(exist_ok=True)
    lock = CACHE / 'running.lock'
    with lock.open('x') as file:
        file.write(str(os.getpid()))
    try:
        config, inherited = configuration()
        torch.set_num_threads(4)
        frame, baseline = metadata()
        if args.smoke_test:
            smoke(config, inherited, frame)
            return
        if not (HERE / 'smoke_test.json').exists():
            raise RuntimeError('Run --smoke-test first')
        signature = {'config': config, 'entry_sha256': previous.sha256_file(Path(__file__)),
                     'training_sha256': previous.sha256_file(ROOT / 'src' / 'emo_rec' / 'finetuning.py')}
        signature_path = CACHE / 'run_signature.json'
        if signature_path.exists() and json.loads(signature_path.read_text(encoding='utf-8')) != signature:
            raise RuntimeError('Training code/config changed; use a fresh cache')
        save_json(signature_path, signature)
        save_json(HERE / 'resolved_config.json', config)
        samples, public = prepare_data(frame)
        labels, folds = frame.y_true.to_numpy(), frame.fold.to_numpy()
        oof_logits = np.full((len(frame), 2), np.nan, dtype=np.float32)
        for fold in range(5):
            validation = np.flatnonzero(folds == fold)
            oof_logits[validation] = fit_predict(config, inherited, samples, labels, np.flatnonzero(folds != fold),
                                               samples, validation, f'fold_{fold}', 2026 + fold)
        public_logits = fit_predict(config, inherited, samples, labels, np.arange(len(frame)),
                                    public, np.arange(80), 'full', config['training']['final_seed'])
        evaluate(config, frame, baseline, oof_logits, public_logits)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
