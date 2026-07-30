import torch

NUM_TONE_CLASSES = 6


def calculate_dice(pred_mask, true_mask, threshold=0.5, eps=1e-7):
    """Standard Dice coefficient for a single image. `pred_mask` may be a probability
    map (e.g. sigmoid output) and is binarized at `threshold`; `true_mask` is assumed
    to already be a binary ground-truth mask."""
    pred_binary = (pred_mask > threshold).float()
    true_binary = (true_mask > threshold).float()

    intersection = (pred_binary * true_binary).sum()
    union = pred_binary.sum() + true_binary.sum()

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.item()


def evaluate_fairness(model, dataloader, device, num_tone_classes=NUM_TONE_CLASSES):
    """Runs the model over `dataloader` and returns per-Fitzpatrick-class Dice scores
    plus the fairness gap (max - min across classes with at least one sample)."""
    model.eval()

    dice_scores_by_tone = {tone: [] for tone in range(num_tone_classes)}

    with torch.no_grad():
        for images, masks, tone_labels in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            tone_labels = tone_labels.to(device)

            mask_logits, _ = model(images)
            mask_probs = torch.sigmoid(mask_logits)

            for i in range(images.size(0)):
                dice = calculate_dice(mask_probs[i], masks[i])
                tone = int(tone_labels[i].item())
                dice_scores_by_tone[tone].append(dice)

    dice_by_tone = {
        tone: (sum(scores) / len(scores) if scores else None)
        for tone, scores in dice_scores_by_tone.items()
    }

    observed_scores = [score for score in dice_by_tone.values() if score is not None]
    fairness_gap = max(observed_scores) - min(observed_scores) if observed_scores else None

    return {
        "dice_by_tone": dice_by_tone,
        "fairness_gap": fairness_gap,
    }
