import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch


class MetricsConfig:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def to_dict(self):
        return {
            "epochs": self.args.epochs,
            "max-samples": self.args.max_samples,
            "learning-rate": self.args.lr,
            "weight-decay": self.args.weight_decay,
            "batch_size": self.args.batch_size,
            "device": self.args.device,
            "num_workers": self.args.num_workers,
        }


class MetricsEntry:
    def __init__(self, epoch: int, loss: float, accu: float, train_times: List[int], eval_times: List[int]) -> None:
        self.epoch: int = epoch
        self.loss: float = loss
        self.accu: float = accu
        self.train_times: List[int] = train_times
        self.eval_times: List[int] = eval_times

    def __str__(self) -> str:
        return f"Epoch {self.epoch: 3d}: Loss = {self.loss:.4f}, Accu = {self.accu:.4f}"

    def to_dict(self):
        return {
            "epoch": self.epoch,
            "loss": self.loss,
            "accu": self.accu
            "train": self.train_times,
            "eval":  self.eval_times,
        }


class MetricsTestEntry:
    def __init__(self, loss: float, accu: float) -> None:
        self.loss: float = loss
        self.accu: float = accu

    def to_dict(self):
        return {
            "loss": self.loss,
            "accu": self.accu
        }


class Metrics:
    def __init__(self) -> None:
        self.conf: Optional[MetricsConfig] = None
        self.epoch: List[MetricsEntry] = []
        self.test: Optional[MetricsTestEntry] = None

    def add_conf(self, args: argparse.Namespace) -> None:
        self.conf = MetricsConfig(args)

    def add_epoch(self, entry: MetricsEntry) -> None:
        self.epoch.append(entry)

    def add_test(self, entry: MetricsTestEntry) -> None:
        self.test = entry

    def to_dict(self):
        return {
            "conf": self.conf.to_dict() if self.conf else None,
            "epoch": [e.to_dict() for e in self.epoch],
            "test": self.test.to_dict() if self.test else None
        }

    def save(self, path: Path = Path("data/train_metrics.json")) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_found_rate(predictions: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]], iou_threshold: float = 0.5):
    """
    predictions: list of dicts with keys 'boxes', 'labels', 'scores'
    targets:     list of dicts with keys 'boxes', 'labels'
    Returns: fraction of GT objects that were "found" (IoU >= thresh + correct class)
    """
    total_gt = 0
    found = 0

    for pred, target in zip(predictions, targets):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]

        if gt_boxes.numel() == 0:
            continue
        total_gt += gt_boxes.size(0)

        if pred_boxes.numel() == 0:
            continue  # no predictions → nothing found

        foreground_mask = (pred_labels != 0)
        if not foreground_mask.any():
            continue

        # Compute IoU between all pred and GT boxes
        # torchvision has a built-in function!
        from torchvision.ops import box_iou
        ious = box_iou(pred_boxes, gt_boxes)  # [num_pred, num_gt]

        # For each GT box, check if any pred matches (same class + IoU >= thresh)
        for gt_idx in range(gt_labels.size(0)):
            gt_label = gt_labels[gt_idx]
            # Find preds with same class
            match_class = (pred_labels == gt_label)
            if not match_class.any():
                continue
            # Max IoU among same-class preds for this GT
            max_iou = ious[match_class, gt_idx].max()
            if max_iou >= iou_threshold:
                found += 1

    return found / total_gt if total_gt > 0 else 1.0
