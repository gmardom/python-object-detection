import argparse
from typing import Any, Callable, Dict, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm

from coco import CocoDataset, CocoSplit
from metrics import Metrics, MetricsEntry, compute_found_rate


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Simple transforms for detecion.

    torchvision models expect images as float tensors in [0, 1].
    """
    t: List[Callable] = [transforms.ToTensor()]
    if train:
        t.insert(0, transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(t)


def build_model(num_classes: int):
    # Use pretrained weights for better convergence
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # Replace the classifier head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def make_model_filename(epoch: int, args: argparse.Namespace, ext: str = ".pt") -> str:
    """
    Generate model name from arguments.
    """
    return f"fasterrcnn_resnet50_fpn_coco_ep{epoch:03d}_lr{args.lr:.2e}_wd{args.weight_decay:.2e}_bs{args.batch_size:d}_ds{args.max_samples:d}{ext}"


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    running_loss = 0.0
    total = 0

    for images, targets in tqdm(loader, desc=f"Train {epoch: 3d}"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict: Dict[str, torch.Tensor] = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += 1

    total = max(total, 1)
    return running_loss / total


def evaluate(
    epoch: int,
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> float:
    model.eval()

    running_found_rate = 0.0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc=f" Eval {epoch: 3d}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            found_rate = compute_found_rate(outputs, targets, iou_threshold=0.5)
            running_found_rate += found_rate

            total += 1

    return running_found_rate / total


def main(args: argparse.Namespace) -> None:
    # Training dataset
    train_split = CocoSplit.VALIDATE if args.dev else CocoSplit.TRAIN
    train_dataset = CocoDataset(
        split=train_split,
        transforms=get_transforms(train=True),
        max_samples=args.max_samples
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Value dataset
    val_split = CocoSplit.VALIDATE
    val_dataset = CocoDataset(
        split=val_split,
        transforms=get_transforms(train=False),
        max_samples=args.max_samples
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Create model
    device = torch.device(args.device)
    model = build_model(train_dataset.num_classes + 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training
    metrics = Metrics()

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(epoch, model, train_loader, optimizer, device)
        accu = evaluate(epoch, model, val_loader, device)

        measure = MetricsEntry(epoch, loss, accu)
        metrics.add_epoch(measure)
        print(measure)

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
