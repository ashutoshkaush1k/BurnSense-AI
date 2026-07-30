import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model import DebiasedBurnUNet
from training.train import train_one_epoch


def test_train_one_epoch_runs_without_error():
    device = torch.device("cpu")
    batch_size, height, width = 2, 64, 64

    images = torch.randn(batch_size, 3, height, width)
    masks = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    tone_labels = torch.randint(0, 6, (batch_size,))

    dataloader = DataLoader(TensorDataset(images, masks, tone_labels), batch_size=batch_size)

    model = DebiasedBurnUNet(in_channels=3, base_channels=16, depth=4, num_tone_classes=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_seg = nn.BCEWithLogitsLoss()
    criterion_adv = nn.CrossEntropyLoss()

    avg_seg_loss, avg_adv_loss = train_one_epoch(
        model, dataloader, optimizer, criterion_seg, criterion_adv, device
    )

    assert isinstance(avg_seg_loss, float)
    assert isinstance(avg_adv_loss, float)
    assert not math.isnan(avg_seg_loss)
    assert not math.isnan(avg_adv_loss)

    encoder_param = next(model.encoder.parameters())
    assert encoder_param.grad is not None
