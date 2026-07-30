import argparse
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

NUM_CLASSES = 3
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SPLIT_SEED = 42
VAL_FRACTION = 0.2


def build_model(finetune_full=False):
    """ResNet18 pretrained on ImageNet, with its final FC layer replaced for 3-way
    burn-degree classification.

    By default (finetune_full=False) everything is frozen except `layer4` (the last
    residual block) and the new `fc` head — `layer4` is unfrozen so the network can
    adapt its highest-level features to medical textures (blisters, charring, etc.)
    instead of relying purely on generic ImageNet features. Pass finetune_full=True
    to fine-tune the entire backbone instead.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if not finetune_full:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def compute_class_weights(full_dataset, indices, num_classes):
    """Inverse-frequency class weights (the standard `n_samples / (n_classes * count)`
    balanced formula) computed over the given subset of the ImageFolder, so
    under-represented degrees (e.g. 3rd-degree) get a proportionally larger loss
    penalty for misclassification instead of being dominated by common classes."""
    targets = [full_dataset.samples[i][1] for i in indices]
    class_counts = Counter(targets)
    total = len(targets)

    weights = [
        total / (num_classes * class_counts[c]) if class_counts.get(c, 0) > 0 else 0.0
        for c in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, dataloader, criterion, optimizer, device, train):
    model.train() if train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        progress_bar = tqdm(dataloader, desc="Train" if train else "Val", unit="batch", leave=False)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Transfer-learning trainer for burn-degree classification (1st/2nd/3rd degree). "
            "--data-dir must be laid out as an ImageFolder, one subdirectory per class, e.g. "
            "<data_dir>/1st_degree/*.jpg, <data_dir>/2nd_degree/*.jpg, <data_dir>/3rd_degree/*.jpg"
        )
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the ImageFolder-style degree dataset.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--finetune-full",
        action="store_true",
        help="Fine-tune the entire ResNet18 backbone instead of just layer4 + the classifier head.",
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/burn_degree_classifier.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform, eval_transform = build_transforms()

    full_dataset = ImageFolder(args.data_dir, transform=train_transform)
    class_names = full_dataset.classes
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} class subdirectories in {args.data_dir}, found {class_names}")

    val_size = int(VAL_FRACTION * len(full_dataset))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # random_split's two Subsets share one underlying ImageFolder; re-point the val
    # subset at a second instance (same root, so identical file ordering/indices) built
    # with eval_transform so validation isn't randomly augmented like training is.
    val_subset.dataset = ImageFolder(args.data_dir, transform=eval_transform)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    class_weights = compute_class_weights(full_dataset, train_subset.indices, NUM_CLASSES).to(device)
    print(f"Class weights (inverse frequency, {class_names}): {[round(w, 3) for w in class_weights.tolist()]}")

    model = build_model(args.finetune_full).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_trainable:,} / {num_total:,}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.2%}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.2%}"
        )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "class_names": class_names}, checkpoint_path)
    print(f"Saved burn-degree classifier to {checkpoint_path} (classes: {class_names})")


if __name__ == "__main__":
    main()
