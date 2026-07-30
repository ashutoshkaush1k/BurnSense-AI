import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model import DebiasedBurnUNet


def test_output_shapes():
    model = DebiasedBurnUNet(in_channels=3, base_channels=16, depth=4, num_tone_classes=6)
    x = torch.randn(2, 3, 256, 256)

    mask, tone_logits = model(x)

    assert mask.shape == (2, 1, 256, 256)
    assert tone_logits.shape == (2, 6)
