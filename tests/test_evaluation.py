import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.fairness_metrics import calculate_dice, evaluate_fairness


class _IdentityDummyModel(nn.Module):
    """Stand-in model: treats its input directly as mask logits (bypassing any real
    encoder/decoder) so tests can control Dice scores exactly."""

    def forward(self, x):
        tone_logits = torch.zeros(x.size(0), 6)
        return x, tone_logits


def test_calculate_dice_perfect_and_zero_overlap():
    true_mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    perfect_pred = true_mask.clone()
    assert calculate_dice(perfect_pred, true_mask) == pytest.approx(1.0, abs=1e-4)

    inverted_pred = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    assert calculate_dice(inverted_pred, true_mask) == pytest.approx(0.0, abs=1e-4)


def test_evaluate_fairness_groups_by_tone_and_computes_gap():
    high_logit, low_logit = 10.0, -10.0
    ones = torch.ones(1, 4, 4)

    # Sample A: tone 0, perfect prediction.
    # Sample B: tone 0, worst-case prediction (true mask is all burn, predicted all background).
    # Sample C: tone 1, perfect prediction.
    images = torch.stack([ones * high_logit, ones * low_logit, ones * high_logit])
    masks = torch.stack([ones, ones, ones])
    tone_labels = torch.tensor([0, 0, 1])

    dataloader = DataLoader(TensorDataset(images, masks, tone_labels), batch_size=3, shuffle=False)

    result = evaluate_fairness(_IdentityDummyModel(), dataloader, torch.device("cpu"))

    dice_by_tone = result["dice_by_tone"]
    assert dice_by_tone[0] == pytest.approx(0.5, abs=1e-4)
    assert dice_by_tone[1] == pytest.approx(1.0, abs=1e-4)
    for tone in range(2, 6):
        assert dice_by_tone[tone] is None

    assert result["fairness_gap"] == pytest.approx(0.5, abs=1e-4)
