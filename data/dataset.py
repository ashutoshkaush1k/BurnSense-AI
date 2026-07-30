import csv
import random
from pathlib import Path

import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
SPLIT_SEED = 42
TARGET_SIZE = (256, 256)


def read_metadata(csv_path):
    """Reads a metadata CSV with columns image_file, mask_file, skin_tone into a list
    of (image_file, mask_file, skin_tone) tuples."""
    samples = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            samples.append((row["image_file"], row["mask_file"], int(row["skin_tone"])))
    return samples


class BurnDataset(Dataset):
    """Loads (image, burn mask, Fitzpatrick skin-tone label) triplets from disk, given
    a list of (image_file, mask_file, skin_tone) samples plus the directories they live
    in. Images are normalized to [0, 1]; masks are binarized to strictly {0, 1}. Both
    are resized to `TARGET_SIZE` (image via bilinear, mask via nearest, to avoid
    introducing fractional values at the mask's binary edges).

    When `augment=True`, a random horizontal flip (p=0.5) is applied identically to the
    image and mask tensors so they stay spatially aligned.
    """

    def __init__(self, samples, img_dir, mask_dir, augment=False):
        self.samples = samples
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_file, mask_file, skin_tone = self.samples[idx]

        image = Image.open(self.img_dir / image_file).convert("RGB")
        mask = Image.open(self.mask_dir / mask_file).convert("L")

        image_tensor = TF.to_tensor(image)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()

        image_tensor = TF.resize(image_tensor, TARGET_SIZE, interpolation=InterpolationMode.BILINEAR)
        mask_tensor = TF.resize(mask_tensor, TARGET_SIZE, interpolation=InterpolationMode.NEAREST)

        if self.augment and random.random() < 0.5:
            image_tensor = TF.hflip(image_tensor)
            mask_tensor = TF.hflip(mask_tensor)

        return image_tensor, mask_tensor, skin_tone


def get_dataloaders(csv_path, img_dir, mask_dir, batch_size=16):
    """Reads metadata_csv, splits samples 80/10/10 into train/val/test, and returns
    the corresponding DataLoaders. Only the training set is augmented (random flip)."""
    samples = read_metadata(csv_path)
    random.Random(SPLIT_SEED).shuffle(samples)

    n = len(samples)
    train_end = int(n * TRAIN_SPLIT)
    val_end = train_end + int(n * VAL_SPLIT)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    train_dataset = BurnDataset(train_samples, img_dir, mask_dir, augment=True)
    val_dataset = BurnDataset(val_samples, img_dir, mask_dir, augment=False)
    test_dataset = BurnDataset(test_samples, img_dir, mask_dir, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
