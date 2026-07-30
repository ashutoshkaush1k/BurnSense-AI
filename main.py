import argparse
import itertools
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from data.dataset import get_dataloaders
from model.model import DebiasedBurnUNet
from training.train import train_one_epoch

DATA_ROOT = Path("data/real")
METADATA_PATH = DATA_ROOT / "metadata.csv"
IMAGES_DIR = DATA_ROOT / "images"
MASKS_DIR = DATA_ROOT / "masks"
CHECKPOINT_PATH = Path("checkpoints/debiased_model.pth")
BATCH_SIZE = 6


def validate_one_epoch(model, dataloader, criterion_seg, criterion_adv, device):
    model.eval()

    running_seg_loss = 0.0
    running_adv_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Val", unit="batch", leave=False)
    with torch.no_grad():
        for images, masks, tone_labels in progress_bar:
            batch_start = time.perf_counter()

            images = images.to(device)
            masks = masks.to(device)
            tone_labels = tone_labels.to(device)

            mask_pred, tone_logits = model(images)

            loss_seg = criterion_seg(mask_pred, masks)
            loss_adv = criterion_adv(tone_logits, tone_labels)

            running_seg_loss += loss_seg.item()
            running_adv_loss += loss_adv.item()
            num_batches += 1

            batch_time = time.perf_counter() - batch_start
            progress_bar.set_postfix(
                seg_loss=f"{loss_seg.item():.4f}",
                adv_loss=f"{loss_adv.item():.4f}",
                batch_time=f"{batch_time:.2f}s",
            )

    return running_seg_loss / num_batches, running_adv_loss / num_batches


def _limited(dataloader, max_batches):
    return itertools.islice(dataloader, max_batches) if max_batches else dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap the number of batches run per epoch (for quick smoke tests)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _test_loader = get_dataloaders(
        METADATA_PATH, IMAGES_DIR, MASKS_DIR, batch_size=BATCH_SIZE
    )

    model = DebiasedBurnUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_seg = nn.BCEWithLogitsLoss()
    criterion_adv = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_seg_loss, train_adv_loss = train_one_epoch(
            model, _limited(train_loader, args.max_batches), optimizer, criterion_seg, criterion_adv, device
        )
        val_seg_loss, val_adv_loss = validate_one_epoch(
            model, _limited(val_loader, args.max_batches), criterion_seg, criterion_adv, device
        )
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_seg: {train_seg_loss:.4f}, train_adv: {train_adv_loss:.4f}, "
            f"val_seg: {val_seg_loss:.4f}, val_adv: {val_adv_loss:.4f}"
        )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved model weights to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
