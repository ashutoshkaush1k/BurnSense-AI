import csv
import shutil
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent.parent / "data" / "degree_data"
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "degree_classification"
SPLITS = ["train", "valid", "test"]

# Roboflow's multi-label CSV has 4 columns; Forth-Degree (a typo for "Fourth") is
# folded into 3rd_degree since 4th-degree burns are clinically graded with 3rd-degree
# for severity purposes, and there are only ~40 such images total.
SEVERITY_ORDER = ["First-Degree", "Second-Degree", "Third-Degree", "Forth-Degree"]
FOLDER_MAP = {
    "First-Degree": "1st_degree",
    "Second-Degree": "2nd_degree",
    "Third-Degree": "3rd_degree",
    "Forth-Degree": "3rd_degree",
}


def pick_label(classes, values):
    """Roughly 15% of rows carry more than one positive label (mostly adjacent-degree
    pairs). Since ImageFolder needs exactly one label per image, we keep the more
    severe of the flagged degrees as a worst-case triage heuristic."""
    positive = [c for c, v in zip(classes, values) if v == 1]
    if not positive:
        return None
    return max(positive, key=SEVERITY_ORDER.index)


def process_split(split):
    split_dir = SOURCE_ROOT / split
    csv_path = split_dir / "_classes.csv"

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        classes = [c.strip() for c in next(reader)[1:]]
        rows = list(reader)

    copied = 0
    skipped = 0
    multi_label_resolved = 0

    for row in rows:
        filename = row[0].strip()
        values = [int(v.strip()) for v in row[1:]]

        label = pick_label(classes, values)
        if label is None:
            skipped += 1
            continue
        if sum(values) > 1:
            multi_label_resolved += 1

        src_path = split_dir / filename
        if not src_path.exists():
            skipped += 1
            continue

        dest_dir = OUTPUT_ROOT / FOLDER_MAP[label]
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / filename
        if dest_path.exists():
            dest_path = dest_dir / f"{split}_{filename}"

        shutil.copy2(src_path, dest_path)
        copied += 1

    return copied, skipped, multi_label_resolved


def main():
    total_copied = 0
    total_skipped = 0
    total_multi = 0

    for split in SPLITS:
        copied, skipped, multi = process_split(split)
        total_copied += copied
        total_skipped += skipped
        total_multi += multi
        print(f"[{split}] copied: {copied}, skipped: {skipped}, multi-label resolved: {multi}")

    print(f"\nDone. Total copied: {total_copied}, skipped: {total_skipped}, multi-label resolved: {total_multi}")

    print("\nClass distribution:")
    for folder in ["1st_degree", "2nd_degree", "3rd_degree"]:
        folder_path = OUTPUT_ROOT / folder
        count = len(list(folder_path.glob("*"))) if folder_path.exists() else 0
        print(f"  {folder}: {count}")


if __name__ == "__main__":
    main()
