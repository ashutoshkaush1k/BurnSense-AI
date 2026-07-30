import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

NUM_SAMPLES = 100
IMAGE_SIZE = 256
NUM_TONE_CLASSES = 6
SEED = 42

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
MASKS_DIR = RAW_DIR / "masks"
METADATA_PATH = RAW_DIR / "metadata.csv"


def make_sample(rng, np_rng):
    """Builds one (rgb_image_array, binary_mask_array) pair: a noisy dark skin-like
    background with 1-3 random bright blobs simulating burn wounds."""
    background = np_rng.integers(10, 45, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    mask_image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(mask_image)

    num_shapes = rng.randint(1, 3)
    for _ in range(num_shapes):
        radius = rng.randint(15, 45)
        cx = rng.randint(radius, IMAGE_SIZE - radius)
        cy = rng.randint(radius, IMAGE_SIZE - radius)
        stretch = rng.uniform(0.6, 1.4)
        bbox = [cx - radius, cy - radius * stretch, cx + radius, cy + radius * stretch]
        draw.ellipse(bbox, fill=255)

    mask_array = np.array(mask_image, dtype=np.uint8)
    burn_pixels = mask_array > 0

    bright_color = np_rng.integers(180, 256, size=3, dtype=np.uint8)
    image_array = background.copy()
    noise = np_rng.integers(-15, 16, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    blended = np.clip(bright_color.astype(int) + noise, 0, 255).astype(np.uint8)
    image_array[burn_pixels] = blended[burn_pixels]

    return image_array, mask_array


def main():
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(NUM_SAMPLES):
        image_array, mask_array = make_sample(rng, np_rng)

        image_filename = f"img_{i:03d}.png"
        mask_filename = f"mask_{i:03d}.png"

        Image.fromarray(image_array, mode="RGB").save(IMAGES_DIR / image_filename)
        Image.fromarray(mask_array, mode="L").save(MASKS_DIR / mask_filename)

        skin_tone_label = rng.randint(0, NUM_TONE_CLASSES - 1)
        rows.append({
            "image_filename": image_filename,
            "mask_filename": mask_filename,
            "skin_tone_label": skin_tone_label,
        })

    with open(METADATA_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_filename", "mask_filename", "skin_tone_label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {NUM_SAMPLES} synthetic samples in {RAW_DIR}")


if __name__ == "__main__":
    main()
