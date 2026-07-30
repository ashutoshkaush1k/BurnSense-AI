import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import BurnDataset
from model.model import DebiasedBurnUNet

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
METADATA_PATH = DATA_ROOT / "metadata.csv"
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "debiased_model.pth"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results_grid.png"
NUM_ROWS = 4


def load_samples(metadata_path):
    samples = []
    with open(metadata_path, newline="") as f:
        for row in csv.DictReader(f):
            samples.append((row["image_filename"], row["mask_filename"], int(row["skin_tone_label"])))
    return samples


def main():
    device = torch.device("cpu")

    samples = load_samples(METADATA_PATH)
    dataset = BurnDataset(samples, img_dir=DATA_ROOT / "images", mask_dir=DATA_ROOT / "masks", augment=False)
    dataloader = DataLoader(dataset, batch_size=NUM_ROWS, shuffle=True)

    images, masks, tones = next(iter(dataloader))

    model = DebiasedBurnUNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    with torch.no_grad():
        mask_logits, _ = model(images.to(device))
        mask_probs = torch.sigmoid(mask_logits)

    fig, axes = plt.subplots(NUM_ROWS, 3, figsize=(9, 3 * NUM_ROWS))

    for row in range(NUM_ROWS):
        image = images[row].permute(1, 2, 0).numpy()
        true_mask = masks[row, 0].numpy()
        pred_prob = mask_probs[row, 0].numpy()
        tone = int(tones[row].item())

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Original Image (Tone {tone})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(true_mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"True Mask (Tone {tone})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_prob, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Predicted Mask (Tone {tone})")
        axes[row, 2].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"Saved visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
