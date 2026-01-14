import json
from pathlib import Path
from typing import List, Optional

class MetricsEntry:
    def __init__(self, epoch: int, loss: float, accu: float) -> None:
        self.epoch: int = epoch
        self.loss: float = loss
        self.accu: float = accu

    def __str__(self) -> str:
        return f"Epoch {self.epoch: 3d}: Loss = {self.loss:.4f}, Accu = {self.accu:.4f}"

    def to_dict(self):
        return {
            "epoch": self.epoch,
            "loss": self.loss,
            "accu": self.accu
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
        self.epoch: List[MetricsEntry] = []
        self.test: Optional[MetricsTestEntry] = None

    def add_epoch(self, entry: MetricsEntry) -> None:
        self.epoch.append(entry)

    def add_test(self, entry: MetricsTestEntry) -> None:
        self.test = entry

    def to_dict(self):
        return {
            "epoch": [e.to_dict() for e in self.epoch],
            "test": self.test.to_dict() if self.test else None
        }

    def save(self, path: Path = Path("data/train_metrics.json")) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
