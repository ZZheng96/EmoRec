import numpy as np
import pytest
import torch
from scipy.special import expit

from emo_rec.finetuning import (
    assert_subject_disjoint, make_optimizer, predict_logits, score_probabilities,
    seed_training, train_epoch,
)


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 4)
        self.fc_norm = torch.nn.LayerNorm(4)
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x, input_chans=None):
        return self.head(self.fc_norm(self.backbone(x)))


def test_optimizer_groups_and_training():
    seed_training(2026)
    model = TinyClassifier().requires_grad_(False)
    optimizer = make_optimizer(model)
    assert all(p.requires_grad for p in model.parameters())
    assert [g['lr'] for g in optimizer.param_groups] == [1e-5, 1e-4]
    assert all(g['weight_decay'] == 0.01 for g in optimizer.param_groups)
    head_ids = {id(p) for g in (model.head, model.fc_norm) for p in g.parameters()}
    assert {id(p) for p in optimizer.param_groups[1]['params']} == head_ids
    before = model.backbone.weight.detach().clone()
    samples = np.random.randn(8, 3).astype('float32')
    labels = np.tile([0, 1], 4)
    device = torch.device('cpu')
    loss = train_epoch(model, [], samples, labels, np.arange(8), optimizer,
                       torch.amp.GradScaler('cuda', enabled=False), device, 3, False)
    assert np.isfinite(loss)
    assert not torch.equal(before, model.backbone.weight)
    logits = predict_logits(model, [], samples, np.arange(8), device, 3, False)
    assert logits.shape == (8, 2)


def test_score_normalization_is_label_free_and_subject_local():
    logits = np.array([[0, 1], [0, 3], [0, 9], [0, 9]], dtype=float)
    subjects = np.array(['a', 'a', 'b', 'b'])
    scores = score_probabilities(logits, subjects)
    np.testing.assert_allclose(scores['raw'], expit(logits[:, 1]))
    np.testing.assert_allclose(scores['subject_zscore'], expit([-1, 1, 0, 0]))
    logits[:2, 1] = logits[:2, 1] * 2 + 100
    np.testing.assert_allclose(score_probabilities(logits, subjects)['subject_zscore'], scores['subject_zscore'])
    with pytest.raises(ValueError):
        score_probabilities(np.full((4, 2), np.nan), subjects)


def test_subject_disjoint():
    assert_subject_disjoint(['a', 'a', 'b', 'b'], [0, 0, 1, 1])
    with pytest.raises(ValueError, match='leakage'):
        assert_subject_disjoint(['a', 'a', 'b', 'b'], [0, 1, 1, 1])


def test_training_seed_repeatable():
    seed_training(2026)
    first = (np.random.rand(3), torch.rand(3))
    seed_training(2026)
    np.testing.assert_array_equal(first[0], np.random.rand(3))
    assert torch.equal(first[1], torch.rand(3))
