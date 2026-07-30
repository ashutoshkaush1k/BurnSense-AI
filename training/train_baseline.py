def train_baseline_epoch(model, dataloader, optimizer, criterion_seg, device):
    """Runs one training epoch for the StandardBurnUNet baseline and returns the
    average segmentation loss. Skin tone labels from the dataloader are ignored —
    there is no adversarial head to consume them."""
    model.train()

    running_seg_loss = 0.0
    num_batches = 0

    for images, masks, _tone_labels in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        mask_logits = model(images)
        loss_seg = criterion_seg(mask_logits, masks)

        optimizer.zero_grad()
        loss_seg.backward()
        optimizer.step()

        running_seg_loss += loss_seg.item()
        num_batches += 1

    return running_seg_loss / num_batches
