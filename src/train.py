import argparse
from typing import Any, Callable, Dict, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm

from coco import CocoDataset, CocoSplit
from metrics import Metrics, MetricsEntry


def get_transforms(train: bool = True) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    if train == True:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


def build_model(num_classes: int):
    # Use pretrained weights for better convergence
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # Replace the classifier head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    # For multi-label classification: check if predicted class is in the set of true classes
    preds = outputs.argmax(dim=1)  # Shape: [batch_size]
    # targets shape: [batch_size, num_classes] - multi-label binary tensor
    # Use advanced indexing to check if predicted class is in true labels for each sample
    batch_indices = torch.arange(preds.size(0), device=targets.device)
    correct = targets[batch_indices, preds] > 0.5
    return correct.float().mean().item()


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_accu = 0.0
    total = 0

    for images, targets in tqdm(loader, desc=f"Train {epoch: 3d}"):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_accu += accuracy(outputs, targets) * batch_size
        total += batch_size

    return running_loss / total, running_accu / total


def evaluate(
    epoch: int,
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    running_loss = 0.0
    running_accu = 0.0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc=f" Eval {epoch: 3d}"):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_accu += accuracy(outputs, targets) * batch_size
            total += batch_size

    return running_loss / total, running_accu / total



def main(args: argparse.Namespace) -> None:
    # Training dataset
    train_split = CocoSplit.VALIDATE if args.dev else CocoSplit.TRAIN
    train_dataset = CocoDataset(split=train_split, transforms=get_transforms(train=True), max_samples=args.max_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Value dataset
    value_split = CocoSplit.VALIDATE
    value_dataset = CocoDataset(split=value_split, transforms=get_transforms(train=False), max_samples=args.max_samples)
    value_loader = DataLoader(value_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create model
    device = torch.device(args.device)
    model = build_model(train_dataset.num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training
    metrics = Metrics()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accu = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device)
        value_loss, value_accu = evaluate(epoch, model, value_loader, criterion, device)

        current = MetricsEntry(epoch, train_loss, train_accu, value_loss, value_accu)
        metrics.add_epoch(current)
        print(current)

    # Save metrics
    metrics.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")

    parser.add_argument(
        "--dev", action=argparse.BooleanOptionalAction, default=False, help="Use smaller dataset while developing.")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit dataset size.")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Dataset batch size.")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Worker count.")

    main(parser.parse_args())
