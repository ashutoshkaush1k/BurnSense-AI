import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

from model.model import DebiasedBurnUNet
from train_classifier import IMAGE_SIZE as CLASSIFIER_IMAGE_SIZE
from train_classifier import IMAGENET_MEAN, IMAGENET_STD, build_model

CHECKPOINT_PATH = Path("checkpoints/debiased_model.pth")
CLASSIFIER_CHECKPOINT_PATH = Path("checkpoints/burn_degree_classifier.pth")
TARGET_SIZE = (256, 256)
MASK_THRESHOLD = 0.5
CONTEXT_PADDING_FRACTION = 0.2

# Heuristic dashboard constants below (base healing days, severity weights, BSI
# thresholds) are placeholders for the frontend to visualize, not a clinically
# validated model.
BASE_HEALING_DAYS = {"1st_degree": 5, "2nd_degree": 21, "3rd_degree": 60}
SEVERITY_SCORE = {"1st_degree": 20, "2nd_degree": 60, "3rd_degree": 100}
ACTION_PLANS = {
    "1st_degree": "Cool the area, apply moisturizer, and monitor for 3-5 days. OTC pain relief as needed.",
    "2nd_degree": (
        "Clean gently, apply antibiotic ointment, cover with a sterile non-stick dressing, "
        "and seek medical evaluation within 24 hours."
    ),
    "3rd_degree": (
        "Seek emergency medical care immediately. Do not apply ointments; "
        "cover loosely with a clean cloth and treat for shock."
    ),
}
LOW_RISK_MAX_BSI = 35
MEDIUM_RISK_MAX_BSI = 65


def extract_padded_crop(image_tensor, binary_mask, padding_fraction=CONTEXT_PADDING_FRACTION):
    """Crops the bounding box of the segmented burn region directly out of the
    original (unmasked) image, expanded by `padding_fraction` on each side.

    Feeding the classifier a zeroed-out background (as before) throws away the
    surrounding skin context — redness spreading past the segmentation boundary,
    blister edges right at the mask's edge, etc. — that's actually useful signal for
    degree classification. Cropping a padded box around the burn keeps that context
    while still focusing the classifier on the relevant region.
    """
    mask_2d = binary_mask[0, 0]
    nonzero = torch.nonzero(mask_2d, as_tuple=False)

    if nonzero.numel() == 0:
        return image_tensor

    y_min, x_min = nonzero.min(dim=0).values.tolist()
    y_max, x_max = nonzero.max(dim=0).values.tolist()

    box_height = y_max - y_min + 1
    box_width = x_max - x_min + 1
    pad_y = round(box_height * padding_fraction)
    pad_x = round(box_width * padding_fraction)

    _, _, img_h, img_w = image_tensor.shape
    y_min = max(0, y_min - pad_y)
    y_max = min(img_h - 1, y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(img_w - 1, x_max + pad_x)

    return image_tensor[:, :, y_min:y_max + 1, x_min:x_max + 1]


def load_segmentation_model(checkpoint_path, device):
    model = DebiasedBurnUNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def load_classifier(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No burn-degree classifier checkpoint at {checkpoint_path}. "
            "Train one first with train_classifier.py."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(finetune_full=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["class_names"]


def classify_burn_degree(padded_crop, classifier_model, class_names, device):
    """Classifies burn severity from the padded burn-region crop using the ResNet18
    transfer-learning classifier trained by train_classifier.py. Returns the full
    softmax distribution as a {class_name: probability} dict (not just the argmax),
    so downstream logic can weight by confidence rather than snapping to one class."""
    classifier_input = TF.resize(
        padded_crop, [CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR
    )
    classifier_input = TF.normalize(classifier_input, mean=IMAGENET_MEAN, std=IMAGENET_STD).to(device)

    with torch.no_grad():
        logits = classifier_model(classifier_input)
        probs = torch.softmax(logits, dim=1)[0]

    return {class_name: probs[i].item() for i, class_name in enumerate(class_names)}


def predict_healing_time(mask_area_percentage, class_probs):
    """Hybrid healing-time / severity estimate driven by the full class-probability
    distribution rather than the single predicted class, so a borderline 1st/2nd
    -degree call nudges the estimate toward the more severe outcome instead of
    snapping entirely into one bucket.

    - estimated_days: probability-weighted base healing time, scaled up by burn area.
    - bsi_score: a 0-100 index blending weighted degree severity (70%) with burn
      area coverage (30%).
    - infection_risk: bucketed from bsi_score.
    - action_plan: a short directive keyed off the single most likely degree.
    """
    weighted_base_days = sum(class_probs[c] * BASE_HEALING_DAYS[c] for c in class_probs)
    area_multiplier = 1 + (mask_area_percentage / 50)
    estimated_days = round(weighted_base_days * area_multiplier, 1)

    weighted_severity = sum(class_probs[c] * SEVERITY_SCORE[c] for c in class_probs)
    area_component = min(mask_area_percentage, 100)
    bsi_score = int(round(max(0.0, min(100.0, 0.7 * weighted_severity + 0.3 * area_component))))

    if bsi_score < LOW_RISK_MAX_BSI:
        infection_risk = "Low"
    elif bsi_score < MEDIUM_RISK_MAX_BSI:
        infection_risk = "Medium"
    else:
        infection_risk = "High"

    predicted_degree = max(class_probs, key=class_probs.get)

    return {
        "estimated_days": estimated_days,
        "bsi_score": bsi_score,
        "infection_risk": infection_risk,
        "action_plan": ACTION_PLANS[predicted_degree],
    }


def preprocess_image(image, device):
    """PIL RGB image -> a [1, 3, *TARGET_SIZE] tensor on `device`."""
    image_tensor = TF.to_tensor(image)
    image_tensor = TF.resize(image_tensor, TARGET_SIZE, interpolation=InterpolationMode.BILINEAR)
    return image_tensor.unsqueeze(0).to(device)


def run_full_analysis(image, segmentation_model, classifier_model, class_names, device):
    """Runs the full segmentation -> padded-crop -> classification -> severity
    pipeline on a single PIL image. This is the single source of truth for the
    pipeline, shared by the CLI (main(), below) and the FastAPI backend (api.py) so
    the two never drift apart.
    """
    image_batched = preprocess_image(image, device)

    with torch.no_grad():
        mask_logits, _ = segmentation_model(image_batched)
        mask_probs = torch.sigmoid(mask_logits)

    binary_mask = (mask_probs > MASK_THRESHOLD).float()
    padded_crop = extract_padded_crop(image_batched, binary_mask)

    mask_area = binary_mask.sum().item()
    area_fraction = mask_area / binary_mask.numel()

    class_probs = classify_burn_degree(padded_crop, classifier_model, class_names, device)
    healing_info = predict_healing_time(area_fraction * 100, class_probs)

    return {
        "mask_area": mask_area,
        "area_fraction": area_fraction,
        "mask_probs": mask_probs,
        "class_probs": class_probs,
        "healing_info": healing_info,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run burn segmentation inference on a single image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH), help="Path to the model checkpoint.")
    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default=str(CLASSIFIER_CHECKPOINT_PATH),
        help="Path to the burn-degree classifier checkpoint (from train_classifier.py).",
    )
    parser.add_argument("--output", type=str, default="inference_result.png", help="Path to save the result plot.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(args.image_path).convert("RGB")

    segmentation_model = load_segmentation_model(Path(args.checkpoint), device)
    classifier_model, class_names = load_classifier(Path(args.classifier_checkpoint), device)

    result = run_full_analysis(image, segmentation_model, classifier_model, class_names, device)

    mask_area = result["mask_area"]
    area_fraction = result["area_fraction"]
    class_probs = result["class_probs"]
    healing_info = result["healing_info"]
    predicted_degree = max(class_probs, key=class_probs.get)

    print("=" * 40)
    print("Burn Analysis Summary Report")
    print("=" * 40)
    print(
        f"Predicted Burn Degree:   {predicted_degree.replace('_', ' ')} "
        f"({class_probs[predicted_degree]:.1%} confidence)"
    )
    print("Class Probabilities:")
    for class_name, prob in class_probs.items():
        print(f"  {class_name.replace('_', ' '):<12} {prob:.1%}")
    print(f"Estimated Healing Time:  {healing_info['estimated_days']} days")
    print(f"Burn Severity Index:     {healing_info['bsi_score']}/100")
    print(f"Infection Risk:          {healing_info['infection_risk']}")
    print(f"Action Plan:             {healing_info['action_plan']}")
    print(f"Burn Area:               {int(mask_area)} px ({area_fraction:.1%} of image)")
    print("=" * 40)

    display_image = preprocess_image(image, device)[0].permute(1, 2, 0).cpu().numpy()
    predicted_mask = result["mask_probs"][0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(display_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(predicted_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Predicted Burn Mask")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Saved inference result to {args.output}")


if __name__ == "__main__":
    main()
