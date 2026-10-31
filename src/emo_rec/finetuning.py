import random

import numpy as np
import torch
from scipy.special import expit

from emo_rec.normalization import normalize_by_subject


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_optimizer(model, backbone_lr=1e-5, head_lr=1e-4, weight_decay=0.01):
    model.requires_grad_(True)
    backbone, head = [], []
    for name, parameter in model.named_parameters():
        (head if name.startswith(('head.', 'fc_norm.')) else backbone).append(parameter)
    return torch.optim.AdamW([
        {'params': backbone, 'lr': backbone_lr, 'name': 'backbone'},
        {'params': head, 'lr': head_lr, 'name': 'head_and_fc_norm'},
    ], weight_decay=weight_decay)


def assert_subject_disjoint(subjects, folds):
    subjects, folds = np.asarray(subjects), np.asarray(folds)
    for fold in np.unique(folds):
        if set(subjects[folds == fold]) & set(subjects[folds != fold]):
            raise ValueError('Subject leakage across folds')


def score_probabilities(logits, subjects):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(subjects), 2) or not np.isfinite(logits).all():
        raise ValueError('Expected finite (n_samples, 2) logits')
    difference = logits[:, 1] - logits[:, 0]
    standardized = normalize_by_subject(difference[:, None], subjects, 'zscore')[:, 0]
    return {'raw': expit(difference), 'subject_zscore': expit(standardized)}


def train_epoch(model, channels, samples, labels, indices, optimizer, scaler, device, batch_size, amp):
    model.train()
    order = np.random.permutation(indices)
    total_loss = 0.0
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        x = torch.from_numpy(np.array(samples[batch], copy=True)).to(device)
        y = torch.as_tensor(labels[batch], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x, input_chans=channels)
            loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(batch)
    return total_loss / len(order)


@torch.no_grad()
def predict_logits(model, channels, samples, indices, device, batch_size, amp):
    model.eval()
    outputs = []
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        x = torch.from_numpy(np.array(samples[batch], copy=True)).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x, input_chans=channels)
        outputs.append(logits.float().cpu().numpy())
    result = np.concatenate(outputs)
    if result.shape != (len(indices), 2) or not np.isfinite(result).all():
        raise RuntimeError('Invalid prediction logits')
    return result
