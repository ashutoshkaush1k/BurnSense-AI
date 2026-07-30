from pathlib import Path

import cv2

REAL_ROOT = Path(__file__).resolve().parent.parent / "data" / "real"
IMAGES_DIR = REAL_ROOT / "images"
MASKS_DIR = REAL_ROOT / "masks"
COVERAGE_THRESHOLD = 0.85


def main():
    mask_paths = sorted(MASKS_DIR.glob("*.png"))
    total = len(mask_paths)
    deleted = 0
    kept = 0

    for i, mask_path in enumerate(mask_paths, 1):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        coverage = (mask > 0).mean()

        if coverage > COVERAGE_THRESHOLD:
            for image_path in IMAGES_DIR.glob(f"{mask_path.stem}.*"):
                image_path.unlink()
            mask_path.unlink()
            deleted += 1
        else:
            kept += 1

        if i % 1000 == 0 or i == total:
            print(f"Processed {i}/{total} masks (deleted so far: {deleted})...")

    print(f"\nDone. Deleted {deleted} lazy bounding-box pairs (coverage > {COVERAGE_THRESHOLD:.0%}).")
    print(f"Remaining high-quality image-mask pairs: {kept}")


if __name__ == "__main__":
    main()
