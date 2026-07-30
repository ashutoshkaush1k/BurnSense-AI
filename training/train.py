import time

from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, criterion_seg, criterion_adv, device):
    """Runs one training epoch and returns (avg_seg_loss, avg_adv_loss).

    The GRL inside the model's adversarial head negates the encoder-bound gradient
    during loss.backward(), so a plain `loss_seg + loss_adv` sum is enough to train
    the segmentation and adversarial debiasing objectives simultaneously.
    """
    model.train()

    running_seg_loss = 0.0
    running_adv_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Train", unit="batch", leave=False)
    for images, masks, tone_labels in progress_bar:
        batch_start = time.perf_counter()

        images = images.to(device)
        masks = masks.to(device)
        tone_labels = tone_labels.to(device)

        mask_pred, tone_logits = model(images)

        loss_seg = criterion_seg(mask_pred, masks)
        loss_adv = criterion_adv(tone_logits, tone_labels)
        loss = loss_seg + loss_adv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_seg_loss += loss_seg.item()
        running_adv_loss += loss_adv.item()
        num_batches += 1

        batch_time = time.perf_counter() - batch_start
        progress_bar.set_postfix(
            seg_loss=f"{loss_seg.item():.4f}",
            adv_loss=f"{loss_adv.item():.4f}",
            batch_time=f"{batch_time:.2f}s",
        )

    avg_seg_loss = running_seg_loss / num_batches
    avg_adv_loss = running_adv_loss / num_batches
    return avg_seg_loss, avg_adv_loss
