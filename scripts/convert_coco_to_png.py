import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

RAW_COCO_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw_coco"
REAL_ROOT = Path(__file__).resolve().parent.parent / "data" / "real"
IMAGES_OUT_DIR = REAL_ROOT / "images"
MASKS_OUT_DIR = REAL_ROOT / "masks"
SPLITS = ["train", "valid", "test"]


def polygons_for_image(annotations, image_id):
    polygons = []
    for ann in annotations:
        if ann["image_id"] != image_id:
            continue
        for polygon in ann["segmentation"]:
            points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            polygons.append(points)
    return polygons


def convert_split(split):
    split_dir = RAW_COCO_ROOT / split
    annotations_path = split_dir / "_annotations.coco.json"

    with open(annotations_path) as f:
        coco = json.load(f)

    annotations_by_image = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    processed = 0
    failed = 0

    for image_info in tqdm(coco["images"], desc=f"{split:>5}", unit="img"):
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        height, width = image_info["height"], image_info["width"]

        src_image_path = split_dir / file_name
        if not src_image_path.exists():
            failed += 1
            continue

        try:
            dst_image_path = IMAGES_OUT_DIR / file_name
            shutil.copy2(src_image_path, dst_image_path)

            mask = np.zeros((height, width), dtype=np.uint8)
            for polygon in polygons_for_image(annotations_by_image.get(image_id, []), image_id):
                cv2.fillPoly(mask, [polygon], color=255)

            mask_path = MASKS_OUT_DIR / f"{Path(file_name).stem}.png"
            cv2.imwrite(str(mask_path), mask)

            processed += 1
        except Exception:
            failed += 1

    return processed, failed


def main():
    IMAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_failed = 0

    for split in SPLITS:
        processed, failed = convert_split(split)
        total_processed += processed
        total_failed += failed
        print(f"[{split}] processed: {processed}, failed: {failed}")

    print(
        f"\nDone. Total images/masks processed: {total_processed}, "
        f"failed: {total_failed}"
    )


if __name__ == "__main__":
    main()
