import json
from pathlib import Path
from typing import List, Optional


class MetricsEntry:
    def __init__(self, epoch: int, train_loss: float, train_accu: float, value_loss: float, value_accu: float) -> None:
        self.epoch: int = epoch
        self.train_loss: float = train_loss
        self.train_accu: float = train_accu
        self.value_loss: float = value_loss
        self.value_accu: float = value_accu

    def __str__(self) -> str:
        return f"Epoch {self.epoch: 3d}: tl = {self.train_loss:.4f}, ta = {self.train_accu:.4f}, vl = {self.value_loss:.4f}, va = {self.value_accu:.4f}"


class MetricsTestEntry:
    def __init__(self, loss: float, accu: float) -> None:
        self.loss: float = loss
        self.accu: float = accu


class Metrics:
    def __init__(self) -> None:
        self.epoch: List[MetricsEntry] = []
        self.test: Optional[MetricsTestEntry] = None

    def add_epoch(self, entry: MetricsEntry) -> None:
        self.epoch.append(entry)

    def add_test(self, entry: MetricsTestEntry) -> None:
        self.test = entry

    def save(self, path: Path = Path("data/train_metrics.json")) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self, f, indent=2)
