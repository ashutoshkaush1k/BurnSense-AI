import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "colab_export.zip"

TARGETS = [
    Path("data/real"),
    Path("data/dataset.py"),
    Path("model"),
    Path("scripts"),
    Path("training"),
    Path("tests"),
    Path("main.py"),
    Path("ARCHITECTURE.md"),
]

EXCLUDED_NAMES = {".venv", "__pycache__", ".pytest_cache"}
EXCLUDED_SUBPATHS = [Path("data/raw_coco")]


def is_excluded(path):
    rel_parts = path.relative_to(PROJECT_ROOT).parts

    if any(part.startswith(".") for part in rel_parts):
        return True
    if any(part in EXCLUDED_NAMES for part in rel_parts):
        return True
    for excluded in EXCLUDED_SUBPATHS:
        if rel_parts[: len(excluded.parts)] == excluded.parts:
            return True
    return False


def iter_files_to_zip():
    for target in TARGETS:
        target_path = PROJECT_ROOT / target

        if not target_path.exists():
            continue

        if target_path.is_file():
            if not is_excluded(target_path):
                yield target_path
            continue

        for root, dirs, files in os.walk(target_path):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if not is_excluded(root_path / d)]

            for file_name in files:
                file_path = root_path / file_name
                if not is_excluded(file_path):
                    yield file_path


def main():
    files = list(iter_files_to_zip())
    total = len(files)

    with zipfile.ZipFile(OUTPUT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, file_path in enumerate(files, 1):
            arcname = file_path.relative_to(PROJECT_ROOT)
            zf.write(file_path, arcname)

            if i % 500 == 0 or i == total:
                print(f"Added {i}/{total} files...")

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSuccess: created {OUTPUT_PATH.name} ({size_mb:.2f} MB) from {total} files.")


if __name__ == "__main__":
    main()
