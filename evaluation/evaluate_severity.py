import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DEGREE_CLASS_NAMES = ["1st-Degree", "2nd-Degree", "3rd-Degree"]


@torch.no_grad()
def evaluate_model(model, test_loader, device, class_names=DEGREE_CLASS_NAMES, cm_output_path="confusion_matrix.png"):
    """Evaluates severity-classification performance of the trained BurnSense AI
    pipeline over `test_loader`, printing an sklearn classification report and
    rendering a confusion matrix.

    Loader contract: each batch is (images, masks, severity_labels, tone_labels).
    `masks` and `tone_labels` are accepted for interface parity with the training
    dataloader — `tone_labels` in particular remains available here for a
    downstream per-tone fairness audit (verifying the GRL's debiasing effect on
    classification parity across skin tones) — but neither is consumed by the
    metrics computed in this function; only `severity_labels` is scored.

    Model contract: `model(images)` returns (mask_logits, severity_logits), i.e.
    the U-Net segmentation head and ResNet18 severity head composed behind a
    single callable. `mask_logits` is not scored here.

    Returns the classification_report as a dict for programmatic reuse (e.g.
    embedding metrics tables directly into the paper's LaTeX build).
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    for images, masks, severity_labels, tone_labels in test_loader:
        images = images.to(device)
        severity_labels = severity_labels.to(device)

        _, severity_logits = model(images)
        predictions = torch.argmax(severity_logits, dim=1)

        all_preds.append(predictions.cpu())
        all_labels.append(severity_labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, zero_division=0
    )

    print("=" * 60)
    print("BurnSense AI - Severity Classification Report")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score:   {report_dict['macro avg']['f1-score']:.4f}")
    print("-" * 60)
    print(report_text)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Severity")
    plt.ylabel("True Severity")
    plt.title("BurnSense AI - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_output_path, dpi=200)
    plt.show()

    return report_dict
