import argparse
import csv
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

BURN_TRAINER_DIR = "/content/burn_trainer"
CONTENT_ROOT = "/content"
METRICS_CSV_PATH = os.path.join(CONTENT_ROOT, "evaluation_metrics.csv")
CONFUSION_MATRIX_PATH = os.path.join(CONTENT_ROOT, "confusion_matrix.png")
CHECKPOINT_BACKUP_PATH = os.path.join(CONTENT_ROOT, "burn_degree_classifier_backup.pth")

# train_classifier.py lives inside BURN_TRAINER_DIR, not next to this script,
# so it has to be added to sys.path explicitly before importing from it.
sys.path.insert(0, BURN_TRAINER_DIR)
from train_classifier import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    NUM_CLASSES,
    SPLIT_SEED,
    VAL_FRACTION,
    build_model,
    build_transforms,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BurnSense AI degree classifier, export metrics, and clean up /content/burn_trainer."
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the ImageFolder-style degree dataset.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained classifier checkpoint.")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def build_test_loader(data_dir, batch_size):
    """train_classifier.py only performs an 80/20 train/val split (no separate
    third test set exists in the current pipeline), so this reconstructs that
    SAME held-out split deterministically, via the identical SPLIT_SEED and
    VAL_FRACTION, and evaluates on it. This is the validation split, not a
    truly unseen test set — flagging that plainly rather than calling it
    something it isn't.
    """
    _, eval_transform = build_transforms()

    full_dataset = ImageFolder(data_dir, transform=eval_transform)
    imagefolder_class_names = full_dataset.classes

    val_size = int(VAL_FRACTION * len(full_dataset))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    _, test_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return test_loader, imagefolder_class_names


def load_trained_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(finetune_full=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint.get("class_names")


@torch.no_grad()
def run_inference(model, test_loader, device):
    all_preds = []
    all_labels = []
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        predictions = torch.argmax(logits, dim=1)
        all_preds.append(predictions.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


def save_metrics_csv(report_dict, output_path):
    fieldnames = ["class", "precision", "recall", "f1-score", "support"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, metrics in report_dict.items():
            if not isinstance(metrics, dict):
                continue
            writer.writerow({
                "class": class_name,
                "precision": metrics.get("precision", ""),
                "recall": metrics.get("recall", ""),
                "f1-score": metrics.get("f1-score", ""),
                "support": metrics.get("support", ""),
            })


def save_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Degree")
    plt.ylabel("True Degree")
    plt.title("BurnSense AI - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def backup_checkpoint_if_needed(checkpoint_path, cleanup_dir, backup_path):
    """The checkpoint often lives inside the directory this script is about to
    delete (e.g. /content/burn_trainer/checkpoints/...). If so, copy it out to
    /content/ first so cleanup never destroys the only copy of the trained
    weights."""
    checkpoint_abs = os.path.abspath(checkpoint_path)
    cleanup_abs = os.path.abspath(cleanup_dir)
    if checkpoint_abs.startswith(cleanup_abs + os.sep):
        shutil.copy2(checkpoint_abs, backup_path)
        print(f"Checkpoint lives inside {cleanup_dir} — backed it up to {backup_path} before cleanup.")
        return True
    return False


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader, imagefolder_class_names = build_test_loader(args.data_dir, args.batch_size)
    model, checkpoint_class_names = load_trained_model(args.checkpoint, device)
    class_names = checkpoint_class_names or imagefolder_class_names

    true_labels, predictions = run_inference(model, test_loader, device)

    accuracy = accuracy_score(true_labels, predictions)
    report_dict = classification_report(
        true_labels, predictions, target_names=class_names, digits=4, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        true_labels, predictions, target_names=class_names, digits=4, zero_division=0
    )

    print("=" * 60)
    print("BurnSense AI - Evaluation Report")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score:   {report_dict['macro avg']['f1-score']:.4f}")
    print("-" * 60)
    print(report_text)

    cm = confusion_matrix(true_labels, predictions)

    save_metrics_csv(report_dict, METRICS_CSV_PATH)
    save_confusion_matrix(cm, class_names, CONFUSION_MATRIX_PATH)

    csv_ok = os.path.exists(METRICS_CSV_PATH) and os.path.getsize(METRICS_CSV_PATH) > 0
    png_ok = os.path.exists(CONFUSION_MATRIX_PATH) and os.path.getsize(CONFUSION_MATRIX_PATH) > 0

    if not (csv_ok and png_ok):
        print("Export verification failed for one or both files — skipping cleanup so nothing is lost.")
        return

    print(f"Verified {METRICS_CSV_PATH} and {CONFUSION_MATRIX_PATH} were written successfully.")

    backup_checkpoint_if_needed(args.checkpoint, BURN_TRAINER_DIR, CHECKPOINT_BACKUP_PATH)

    if os.path.isdir(BURN_TRAINER_DIR):
        shutil.rmtree(BURN_TRAINER_DIR)
        print(f"Deleted {BURN_TRAINER_DIR}.")
    else:
        print(f"{BURN_TRAINER_DIR} not found — nothing to clean up.")


if __name__ == "__main__":
    main()
