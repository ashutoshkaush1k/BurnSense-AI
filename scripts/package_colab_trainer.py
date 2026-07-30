import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "colab_trainer.zip"

DATASET_SRC = PROJECT_ROOT / "data" / "degree_classification"
TRAIN_SCRIPT_SRC = PROJECT_ROOT / "train_classifier.py"

EXCLUDED_NAMES = {".venv", "__pycache__", ".pytest_cache"}


def is_excluded(path):
    return any(part.startswith(".") or part in EXCLUDED_NAMES for part in path.parts)


def main():
    if not DATASET_SRC.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_SRC}")
    if not TRAIN_SCRIPT_SRC.exists():
        raise FileNotFoundError(f"train_classifier.py not found: {TRAIN_SCRIPT_SRC}")

    dataset_files = [p for p in DATASET_SRC.rglob("*") if p.is_file() and not is_excluded(p.relative_to(DATASET_SRC))]

    with zipfile.ZipFile(OUTPUT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, file_path in enumerate(dataset_files, 1):
            arcname = Path("dataset") / file_path.relative_to(DATASET_SRC)
            zf.write(file_path, arcname)
            if i % 500 == 0 or i == len(dataset_files):
                print(f"Added {i}/{len(dataset_files)} dataset files...")

        zf.write(TRAIN_SCRIPT_SRC, "train_classifier.py")

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSuccess: created {OUTPUT_PATH} ({size_mb:.2f} MB)")
    print(f"  dataset/           {len(dataset_files)} files")
    print("  train_classifier.py")


if __name__ == "__main__":
    main()
