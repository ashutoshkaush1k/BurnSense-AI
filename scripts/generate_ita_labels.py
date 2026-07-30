import csv
import math
from pathlib import Path

import cv2
import numpy as np

REAL_ROOT = Path(__file__).resolve().parent.parent / "data" / "real"
IMAGES_DIR = REAL_ROOT / "images"
MASKS_DIR = REAL_ROOT / "masks"
METADATA_PATH = REAL_ROOT / "metadata.csv"

MIN_VALID_PIXELS = 500
B_EPS = 1e-6


def classify_ita(ita):
    if ita > 55:
        return 0
    elif ita > 41:
        return 1
    elif ita > 28:
        return 2
    elif ita > 10:
        return 3
    elif ita > -30:
        return 4
    else:
        return 5


def compute_ita(image_path, mask_path):
    """Returns (ita, skin_tone_class) or None if there aren't enough valid skin
    pixels (or the division would be unsafe) to score this pair."""
    image_bgr = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    healthy_skin = mask == 0
    pure_black = np.all(image_rgb == 0, axis=-1)
    pure_white = np.all(image_rgb == 255, axis=-1)
    valid_mask = healthy_skin & ~pure_black & ~pure_white

    if valid_mask.sum() < MIN_VALID_PIXELS:
        return None

    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    # OpenCV's 8-bit LAB encodes L as 0-255 (not 0-100) and a/b as 0-255 offset
    # by 128 (not -128..127) — convert to standard CIELAB units before using them.
    L_channel = lab_image[..., 0] * (100.0 / 255.0)
    b_channel = lab_image[..., 2] - 128.0

    mean_L = L_channel[valid_mask].mean()
    mean_b = b_channel[valid_mask].mean()

    if abs(mean_b) < B_EPS:
        return None

    ita = math.atan((mean_L - 50) / mean_b) * (180 / math.pi)
    return ita, classify_ita(ita)


def main():
    mask_paths = sorted(MASKS_DIR.glob("*.png"))
    total = len(mask_paths)

    labeled = 0
    skipped = 0
    tone_counts = {i: 0 for i in range(6)}
    rows = []

    for i, mask_path in enumerate(mask_paths, 1):
        matches = list(IMAGES_DIR.glob(f"{mask_path.stem}.*"))
        image_path = matches[0]

        result = compute_ita(image_path, mask_path)

        if result is None:
            image_path.unlink()
            mask_path.unlink()
            skipped += 1
        else:
            _, skin_tone = result
            rows.append({
                "image_file": image_path.name,
                "mask_file": mask_path.name,
                "skin_tone": skin_tone,
            })
            tone_counts[skin_tone] += 1
            labeled += 1

        if i % 500 == 0 or i == total:
            print(f"Processed {i}/{total} pairs (labeled: {labeled}, skipped: {skipped})...")

    with open(METADATA_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_file", "mask_file", "skin_tone"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Successfully labeled {labeled} images, skipped/deleted {skipped}.")
    print("Fitzpatrick class distribution:")
    for tone, count in tone_counts.items():
        print(f"  Class {tone}: {count}")


if __name__ == "__main__":
    main()
