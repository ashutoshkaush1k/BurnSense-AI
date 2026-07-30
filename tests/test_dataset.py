import csv
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import BurnDataset, get_dataloaders


def _write_dummy_pair(img_dir, mask_dir, name, rng):
    rgb_array = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    mask_array = rng.choice([0, 255], size=(64, 64)).astype(np.uint8)

    image_file = f"{name}.png"
    mask_file = f"{name}_mask.png"

    Image.fromarray(rgb_array, mode="RGB").save(img_dir / image_file)
    Image.fromarray(mask_array, mode="L").save(mask_dir / mask_file)

    return image_file, mask_file


def test_getitem_returns_expected_shapes_and_types():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_dir = Path(tmp_dir) / "images"
        mask_dir = Path(tmp_dir) / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()

        image_file, mask_file = _write_dummy_pair(img_dir, mask_dir, "sample", rng)
        skin_tone = 3

        dataset = BurnDataset([(image_file, mask_file, skin_tone)], img_dir, mask_dir)
        image_tensor, mask_tensor, label = dataset[0]

        assert image_tensor.shape == (3, 256, 256)
        assert mask_tensor.shape == (1, 256, 256)
        assert label == skin_tone


def test_mask_is_strictly_binary():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_dir = Path(tmp_dir) / "images"
        mask_dir = Path(tmp_dir) / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()

        image_file, mask_file = _write_dummy_pair(img_dir, mask_dir, "sample", rng)

        dataset = BurnDataset([(image_file, mask_file, 0)], img_dir, mask_dir)
        _, mask_tensor, _ = dataset[0]

        unique_values = set(torch.unique(mask_tensor).tolist())
        assert unique_values <= {0.0, 1.0}


def test_augmented_flip_keeps_image_and_mask_in_sync():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_dir = Path(tmp_dir) / "images"
        mask_dir = Path(tmp_dir) / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()

        image_file, mask_file = _write_dummy_pair(img_dir, mask_dir, "sample", rng)

        unaugmented = BurnDataset([(image_file, mask_file, 0)], img_dir, mask_dir, augment=False)
        original_image, original_mask, _ = unaugmented[0]

        augmented = BurnDataset([(image_file, mask_file, 0)], img_dir, mask_dir, augment=True)
        with patch("data.dataset.random.random", return_value=0.0):
            flipped_image, flipped_mask, _ = augmented[0]

        assert torch.equal(flipped_image, TF.hflip(original_image))
        assert torch.equal(flipped_mask, TF.hflip(original_mask))


def test_get_dataloaders_split_sizes_and_augmentation_flags():
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        img_dir = tmp_dir / "images"
        mask_dir = tmp_dir / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()

        csv_path = tmp_dir / "metadata.csv"
        rows = []
        for i in range(20):
            image_file, mask_file = _write_dummy_pair(img_dir, mask_dir, f"sample_{i}", rng)
            rows.append({"image_file": image_file, "mask_file": mask_file, "skin_tone": i % 6})

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_file", "mask_file", "skin_tone"])
            writer.writeheader()
            writer.writerows(rows)

        train_loader, val_loader, test_loader = get_dataloaders(csv_path, img_dir, mask_dir, batch_size=4)

        assert len(train_loader.dataset) == 16
        assert len(val_loader.dataset) == 2
        assert len(test_loader.dataset) == 2

        assert train_loader.dataset.augment is True
        assert val_loader.dataset.augment is False
        assert test_loader.dataset.augment is False
