import random
from pathlib import Path
import argparse
import torch
from torch import nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm

from coco import CocoDataset, CocoSplit


def load_model(path: Path, device: torch.device, num_classes: int) -> nn.Module:
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    checkpoint = torch.load(path, map_location=device)
    try:
        model.load_state_dict(checkpoint)
    except Exception:
        try:
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        except Exception:
            model = checkpoint
    model.to(device)
    model.eval()
    return model


def draw_boxes_on_img(img_pil, boxes, labels, scores, categories, threshold=0.5):
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()
    for box, lbl, scr in zip(boxes, labels, scores):
        if scr < threshold:
            continue
        x1, y1, x2, y2 = box
        cls_name = categories[int(lbl)]
        caption = f"{cls_name}: {scr:.2f}"
        # rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text_size = font.getbbox(caption)
        # background for text
        text_bg = [x1, y1 - text_size[1], x1 + text_size[0], y1]
        draw.rectangle(text_bg, fill="red")
        draw.text((x1, y1 - text_size[1]), caption, fill="white", font=font)
    return img_pil


def main(args: argparse.Namespace) -> None:
    split = CocoSplit.TEST
    transform = transforms.Compose([ transforms.ToTensor() ])
    dataset = CocoDataset(split=split, transforms=transform, max_samples=0)

    categories: list[str] = ["__background__"]
    categories.extend(dataset.idx_to_cat)

    model = load_model(args.model, args.device, len(categories))

    results_path = Path("data/results")
    results_path.mkdir(parents=True, exist_ok=True)

    images = [f for f in os.listdir(dataset.root)]
    random.seed(args.seed)
    random.shuffle(images)
    images = images[:args.count]

    with torch.no_grad():
        for img_name in tqdm(images):
            img_path = dataset.root / img_name

            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).to(args.device)

            outputs = model([img_tensor])
            output = outputs[0]
            boxes  = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            annotated = img.copy()
            annotated = draw_boxes_on_img(annotated, boxes, labels, scores, categories, threshold=args.confidence)
            annotated.save(results_path / img_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test")

    parser.add_argument(
        "--model", type=str, default=None, help="Model state to be loaded.")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    parser.add_argument(
        "--seed", default=67, help="Seed for image selection.")
    parser.add_argument(
        "--count", default=10, help="Count of images to test on.")
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Boxes confidence threshold.")

    main(parser.parse_args())
