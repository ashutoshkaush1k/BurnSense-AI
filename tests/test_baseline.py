import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.baseline import StandardBurnUNet
from training.train_baseline import train_baseline_epoch


def test_forward_output_shape():
    model = StandardBurnUNet(in_channels=3, base_channels=16, depth=4)
    x = torch.randn(2, 3, 256, 256)

    mask_logits = model(x)

    assert mask_logits.shape == (2, 1, 256, 256)


def test_train_baseline_epoch_runs_without_error():
    device = torch.device("cpu")
    batch_size, height, width = 2, 64, 64

    images = torch.randn(batch_size, 3, height, width)
    masks = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    tone_labels = torch.randint(0, 6, (batch_size,))

    dataloader = DataLoader(TensorDataset(images, masks, tone_labels), batch_size=batch_size)

    model = StandardBurnUNet(in_channels=3, base_channels=16, depth=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_seg = nn.BCEWithLogitsLoss()

    avg_seg_loss = train_baseline_epoch(model, dataloader, optimizer, criterion_seg, device)

    assert isinstance(avg_seg_loss, float)
    assert not math.isnan(avg_seg_loss)

    encoder_param = next(model.encoder.parameters())
    assert encoder_param.grad is not None
