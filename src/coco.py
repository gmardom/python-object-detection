from enum import Enum
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import zipfile

from PIL import Image
from pycocotools.coco import COCO
import requests
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm


class CocoSplit(Enum):
    TRAIN = "train"
    VALIDATE = "val"
    TEST = "test"

    def get_images_root(self, root: Path) -> Path:
        path: Path = root / f"{self.value}2017"
        if not path.exists():
            down_url = f"http://images.cocodataset.org/zips/{self.value}2017.zip"
            down_zip = root / f"{self.value}2017.zip"
            root.mkdir(parents=True, exist_ok=True)
            self.__download_and_unzip_file(down_url, down_zip, root)
        return path

    def get_annotations(self, root: Path) -> Path:
        path: Path = root / "annotations" / f"instances_{"val" if self is CocoSplit.TEST else self.value}2017.json"
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
    """
    COCO object detection dataset.

    Returns:
        image (Tensor): Image tensor (C, H, W).
        target (Dict[str, Tensor]): Dict containing:
            - boxes (FloatTensor[N, 4]): bounding boxes in (x1, y1, x2, y2) format
            - labels (Int64Tensor[N]): class labels in [1, num_classes]
            - image_id (Int64Tensor[1]): image id
            - area (Tensor[N]): area of the boxes
            - iscrowd (UInt8Tensor[N]): crowd flags
    """

    def __init__(
        self,
        root: Path = Path("data/coco"),
        split: CocoSplit = CocoSplit.TRAIN,
        transforms: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        seed: Any = None,
    ):
        # Basic information
        self.split: CocoSplit = split
        self.coco: COCO = COCO(split.get_annotations(root))
        self.root: Path = split.get_images_root(root)
        self.transforms: Optional[Callable] = transforms

        # Limit samples if desired
        self.img_ids: list[int] = self.coco.getImgIds()
        if max_samples:
            random.seed(seed)
            random.shuffle(self.img_ids)
            self.img_ids = self.img_ids[:max_samples]

        # Get list of category ids and names in a deterministic order
        self.cat_ids: list[int] = self.coco.getCatIds()
        self.cat_id_to_idx: dict[int, int] = { cid: i for i, cid in enumerate[int](self.cat_ids) }
        self.idx_to_cat: list[str] = [self.coco.loadCats([cid])[0]["name"] for cid in self.cat_ids]
        self.num_classes: int = len(self.cat_ids)

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Any]]:
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = self.root / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        if isinstance(image, Image.Image):
            image = torchvision.transforms.ToTensor()(image)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes:   List[List[float]] = []
        labels:  List[int] = []
        areas:   List[float] = []
        iscrowd: List[int] = []

        for ann in anns:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            # Convert to compatible bbox-es
            x1, y1, x2, y2 = x, y, x+w, y+h
            boxes.append([x1, y1, x2, y2])

            # Gather categories
            cid = ann["category_id"]
            if cid in self.cat_id_to_idx:
                labels.append(self.cat_id_to_idx[cid] + 1)
            else:
                continue

            # Gather rest of info
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if len(boxes) == 0:
            # Generate empty tensors if no boxes found
            boxes_tensor   = torch.zeros((0, 4),      dtype=torch.float32)
            labels_tensor  = torch.zeros((0,),        dtype=torch.int64)
            areas_tensor   = torch.zeros((0,),        dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,),        dtype=torch.uint8)
        else:
            boxes_tensor   = torch.as_tensor(boxes,   dtype=torch.float32)
            labels_tensor  = torch.as_tensor(labels,  dtype=torch.int64)
            areas_tensor   = torch.as_tensor(areas,   dtype=torch.float32)
            iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.uint8)

        target: Dict[str, Any] = {
            "boxes":    boxes_tensor,
            "labels":   labels_tensor,
            "area":     areas_tensor,
            "iscrowd":  iscrowd_tensor,
            "image_id": torch.as_tensor([img_id], dtype=torch.int64),
        }

        return image, target
