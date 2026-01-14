from enum import Enum
import os
from pathlib import Path
from typing import Callable, Optional
import zipfile

from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import requests
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class CocoSplit(Enum):
    TRAIN: str = "train"
    VAL: str = "val"
    TEST: str = "test"

    def get_transforms(self) -> transforms.Compose:
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        if self is self.TRAIN:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif self is self.VAL:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        elif self is self.TEST:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return None

    def get_images_root(self, root: Path) -> Path:
        path: Path = root / f"{self.value}2017"
        if not path.exists():
            down_url = f"http://images.cocodataset.org/zips/{self.value}2017.zip"
            down_zip = root / f"{self.value}2017.zip"
            root.mkdir(parents=True, exist_ok=True)
            self.__download_and_unzip_file(down_url, down_zip, root)
        return path

    def get_annotations(self, root: Path) -> Path:
        path: Path = root / "annotations" / f"instances_{self.value}2017.json"
        if not path.exists():
            down_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            down_zip = root / f"annotations_trainval2017.zip"
            root.mkdir(parents=True, exist_ok=True)
            self.__download_and_unzip_file(down_url, down_zip, root)
        return path

    def __download_and_unzip_file(self, url: str, down: Path, dest: Path):
        # Skip everything if down (zip) already exists
        if down.exists():
            return

        # Download the zip file
        response: requests.Response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size: int = int(response.headers.get('content-length', 0))
        block_size: int = 1024

        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {down.name}")
        with open(down, "wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    progress_bar.update(len(chunk))
                    f.write(chunk)
        progress_bar.close()

        # Unzip to destination directory
        dest_dir = dest if dest.suffix == "" else dest.parent
        print(f"Unzipping {down} to {dest_dir} ...")
        with zipfile.ZipFile(down, "r") as zip_ref:
            zip_ref.extractall(path=dest_dir)


class CocoDataset(Dataset):
    """Builds a multi-label dataset from COCO 'instances' annotations."""

    def __init__(
        self,
        root: Path = Path("data/coco"),
        split: CocoSplit = CocoSplit.TRAIN,
        transforms: Optional[Callable] = None,
        max_samples: Optional[int] = None
    ):
        # Basic information
        self.split: CocoSplit = split
        self.coco: COCO = COCO(split.get_annotations(root))
        self.root: Path = split.get_images_root(root)
        self.transforms: Optional[Callable] = transforms

        # Limit samples if desired
        self.img_ids: list[int] = self.coco.getImgIds()
        if max_samples:
            self.img_ids = self.img_ids[:max_samples]

        # Get list of category ids and names in a deterministic order
        self.cat_ids: list[int] = self.coco.getCatIds()
        self.cat_id_to_idx: dict[int, int] = { cid: i for i, cid in enumerate[int](self.cat_ids) }
        self.idx_to_cat: list[str] = [self.coco.loadCats([cid])[0]["name"] for cid in self.cat_ids]
        self.num_classes: int = len(self.cat_ids)

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> (Image, Tensor):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        label = np.zeros(self.num_classes, dtype=np.float32)
        for ann in anns:
            cid = ann["category_id"]
            if cid in self.cat_id_to_idx:
                label[self.cat_id_to_idx[cid]] = 1.0

        if self.transforms:
            image = self.transforms(image)

        return image, torch.from_numpy(label)
