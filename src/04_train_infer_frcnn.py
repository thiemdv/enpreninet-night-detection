"""F05: Faster R-CNN ResNet-152 Training & Inference Script.

Builds a two-stage Faster R-CNN detector using torchvision's
`resnet_fpn_backbone(backbone_name="resnet152", weights=...)` wrapped in
`FasterRCNN(..., num_classes=4, min_size=640, max_size=640)`.

Label contract:
  - Training uses Faster R-CNN IDs `1..3` (`0` = background), encoded via
    `encode_frcnn_targets()`.
  - Inference removes background (`0`) and maps `1..3 -> 0..2` via
    `decode_frcnn_predictions()`, exporting the unified prediction schema:
      {image_id: {"boxes": FloatTensor[N, 4], "scores": FloatTensor[N], "labels": LongTensor[N]}}

Usage:
    python src/04_train_infer_frcnn.py --config configs/config.yaml --dummy --epochs 1 --device cpu
    python src/04_train_infer_frcnn.py --config configs/config.yaml --image-variant standard_local_he --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet152_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_IMAGE_VARIANTS = ("raw", "fig4_literal", "fig4_normalized", "standard_local_he")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def select_device(device_arg: str | None, cfg: dict[str, Any]) -> torch.device:
    req = (device_arg or str(cfg["project"]["device"])).lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def resolve_variant_image(data_root: Path, image_variant: str, split: str, file_name: str) -> Path:
    """Resolve image path under images/<variant>/<split>/ supporting both .png (GHE) and .jpg (raw)."""
    split_dir = data_root / "images" / image_variant / split
    stem = Path(file_name).stem
    for cand in (split_dir / file_name, split_dir / f"{stem}.png", split_dir / f"{stem}.jpg"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Image '{file_name}' not found under {split_dir}")


def empty_prediction() -> dict[str, torch.Tensor]:
    """Return an empty unified prediction entry with exact shapes and dtypes."""
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "scores": torch.empty((0,), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.long),
    }


def encode_frcnn_targets(
    annotations: list[dict[str, Any]],
    image_id: int,
    img_size: int = 640,
    num_target_classes: int = 3,
) -> dict[str, torch.Tensor]:
    """Encode COCO annotations (category_id 0..2, bbox xywh) into Faster R-CNN targets (labels 1..3, xyxy).

    Safely handles negative samples with 0 annotations by emitting empty (0, 4) boxes.
    """
    boxes_list: list[list[float]] = []
    labels_list: list[int] = []
    areas_list: list[float] = []
    iscrowd_list: list[int] = []

    for ann in annotations:
        cat_id = int(ann["category_id"])
        if cat_id < 0 or cat_id >= num_target_classes:
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        x1 = max(0.0, min(float(img_size), x))
        y1 = max(0.0, min(float(img_size), y))
        x2 = max(0.0, min(float(img_size), x + w))
        y2 = max(0.0, min(float(img_size), y + h))
        if x2 <= x1 or y2 <= y1:
            continue

        boxes_list.append([x1, y1, x2, y2])
        labels_list.append(cat_id + 1)  # 0..2 -> 1..3 (0 reserved for background)
        areas_list.append((x2 - x1) * (y2 - y1))
        iscrowd_list.append(int(ann.get("iscrowd", 0)))

    if not boxes_list:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
            "image_id": torch.tensor([int(image_id)], dtype=torch.int64),
            "area": torch.empty((0,), dtype=torch.float32),
            "iscrowd": torch.empty((0,), dtype=torch.int64),
        }

    return {
        "boxes": torch.tensor(boxes_list, dtype=torch.float32),
        "labels": torch.tensor(labels_list, dtype=torch.int64),
        "image_id": torch.tensor([int(image_id)], dtype=torch.int64),
        "area": torch.tensor(areas_list, dtype=torch.float32),
        "iscrowd": torch.tensor(iscrowd_list, dtype=torch.int64),
    }


def decode_frcnn_predictions(
    raw_output: dict[str, torch.Tensor] | None,
    img_size: int = 640,
    score_floor: float = 0.001,
    num_target_classes: int = 3,
) -> dict[str, torch.Tensor]:
    """Decode Faster R-CNN output (labels 1..3, 0=bg) into canonical unified schema (labels 0..2).

    Filters background (0), enforces score_floor, clamps xyxy boxes to [0, img_size],
    sorts descending by score, and supports empty (0, 4) tensors.
    """
    if raw_output is None:
        return empty_prediction()

    boxes = raw_output["boxes"].detach().cpu().to(torch.float32).clone()
    scores = raw_output["scores"].detach().cpu().to(torch.float32).clone().clamp(0.0, 1.0)
    labels = raw_output["labels"].detach().cpu().to(torch.int64).clone()

    if boxes.numel() == 0:
        return empty_prediction()

    boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(img_size))
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(img_size))

    valid_mask = (
        (labels >= 1)
        & (labels <= int(num_target_classes))
        & (scores >= float(score_floor))
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    if not bool(valid_mask.any()):
        return empty_prediction()

    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    labels = labels[valid_mask] - 1  # Map 1..3 -> 0..2

    order = torch.argsort(scores, descending=True)
    return {
        "boxes": boxes[order].contiguous().to(torch.float32),
        "scores": scores[order].contiguous().to(torch.float32),
        "labels": labels[order].contiguous().to(torch.long),
    }


def build_faster_rcnn_model(
    backbone_name: str = "resnet152",
    num_classes: int = 4,
    img_size: int = 640,
    pretrained: bool = False,
    score_floor: float = 0.001,
    nms_iou: float = 0.50,
) -> FasterRCNN:
    """Build Faster R-CNN with ResNet-152 FPN backbone and fixed 640x640 input scale."""
    weights = ResNet152_Weights.DEFAULT if (pretrained and backbone_name == "resnet152") else None
    backbone = resnet_fpn_backbone(backbone_name=backbone_name, weights=weights)
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        min_size=img_size,
        max_size=img_size,
        box_score_thresh=score_floor,
        box_nms_thresh=nms_iou,
    )
    return model


class CocoNightDataset(Dataset):
    """PyTorch Dataset reading F02 COCO JSON and images/<variant>/<split>/."""

    def __init__(
        self,
        data_root: Path,
        coco_json_path: Path,
        image_variant: str,
        split: str,
        img_size: int = 640,
        num_target_classes: int = 3,
    ) -> None:
        self.data_root = data_root
        self.image_variant = image_variant
        self.split = split
        self.img_size = img_size
        self.num_target_classes = num_target_classes

        if not coco_json_path.is_file():
            raise FileNotFoundError(f"COCO split file not found: {coco_json_path}")
        coco_data = json.loads(coco_json_path.read_text(encoding="utf-8"))
        self.images: list[dict[str, Any]] = list(coco_data.get("images", []))
        self.anns_by_img: dict[int, list[dict[str, Any]]] = {int(img["id"]): [] for img in self.images}
        for ann in coco_data.get("annotations", []):
            img_id = int(ann["image_id"])
            if img_id in self.anns_by_img:
                self.anns_by_img[img_id].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        entry = self.images[idx]
        img_id = int(entry["id"])
        img_path = resolve_variant_image(self.data_root, self.image_variant, self.split, entry["file_name"])
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        target = encode_frcnn_targets(
            annotations=self.anns_by_img[img_id],
            image_id=img_id,
            img_size=self.img_size,
            num_target_classes=self.num_target_classes,
        )
        return tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def train_and_infer_frcnn(
    cfg: dict[str, Any],
    data_root: Path,
    image_variant: str,
    eval_split: str,
    epochs: int,
    batch_size: int,
    device: torch.device,
    score_floor: float,
    dummy: bool,
) -> tuple[Path, Path, float]:
    """Train Faster R-CNN ResNet-152 and export unified predictions."""
    seed = int(cfg["project"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    frcnn_cfg = cfg["models"]["faster_rcnn"]
    img_size = int(frcnn_cfg["img_size"])
    num_classes = int(frcnn_cfg["num_classes"])
    num_target_classes = len(cfg["dataset"]["classes"])
    intra_nms_iou = float(cfg["fusion"]["intra_nms_iou"])
    coco_dir = data_root / cfg["paths"]["coco_subdir"]

    train_ds = CocoNightDataset(
        data_root=data_root,
        coco_json_path=coco_dir / "instances_train.json",
        image_variant=image_variant,
        split="train",
        img_size=img_size,
        num_target_classes=num_target_classes,
    )
    eval_ds = CocoNightDataset(
        data_root=data_root,
        coco_json_path=coco_dir / f"instances_{eval_split}.json",
        image_variant=image_variant,
        split=eval_split,
        img_size=img_size,
        num_target_classes=num_target_classes,
    )

    effective_bs = 1 if dummy else batch_size
    train_loader = DataLoader(train_ds, batch_size=effective_bs, shuffle=not dummy, collate_fn=collate_fn)

    # In --dummy mode, strictly set pretrained=False (weights=None) for offline execution
    use_pretrained = False if dummy else bool(frcnn_cfg["pretrained"])
    model = build_faster_rcnn_model(
        backbone_name=str(frcnn_cfg["backbone"]),
        num_classes=num_classes,
        img_size=img_size,
        pretrained=use_pretrained,
        score_floor=score_floor,
        nms_iou=intra_nms_iou,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=float(frcnn_cfg["learning_rate"]),
        momentum=float(frcnn_cfg["momentum"]),
        weight_decay=float(frcnn_cfg["weight_decay"]),
    )

    last_loss = 0.0
    model.train()
    for _ in range(max(1, epochs)):
        for step_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

            if dummy and step_idx >= 0:
                break  # 1 batch/step offline smoke test

    if dummy:
        run_dir = resolve_path(cfg["paths"]["runs_dir"]) / f"dummy_frcnn_{image_variant}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = run_dir / f"frcnn_resnet152_dummy_{image_variant}.pth"
    else:
        run_dir = resolve_path(cfg["paths"]["runs_dir"]) / f"frcnn_{image_variant}"
        run_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = resolve_path(cfg["paths"]["weights_dir"])
        weights_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = weights_dir / f"frcnn_resnet152_{image_variant}.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone": str(frcnn_cfg["backbone"]),
            "num_classes": num_classes,
            "img_size": img_size,
            "image_variant": image_variant,
            "dummy": dummy,
        },
        ckpt_path,
    )

    # Inference on eval_split and decode 1..3 -> 0..2 into unified schema
    model.eval()
    preds_by_image: dict[int, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for idx in range(len(eval_ds)):
            img_tensor, target = eval_ds[idx]
            img_id = int(target["image_id"].item())
            raw_out = model([img_tensor.to(device)])[0]
            preds_by_image[img_id] = decode_frcnn_predictions(
                raw_output=raw_out,
                img_size=img_size,
                score_floor=score_floor,
                num_target_classes=num_target_classes,
            )

    preds_path = run_dir / f"frcnn_{image_variant}_{eval_split}_preds.pt"
    torch.save(preds_by_image, preds_path)
    return ckpt_path, preds_path, last_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train & infer Faster R-CNN ResNet-152 and export unified predictions.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    parser.add_argument(
        "--image-variant",
        type=str,
        choices=VALID_IMAGE_VARIANTS,
        default=None,
        help="Input image variant (default: preprocessing.ghe_ensemble.method)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, or auto")
    parser.add_argument("--score-floor", type=float, default=0.001, help="Minimum confidence score kept before fusion")
    parser.add_argument("--eval-split", type=str, choices=("val", "test"), default="test", help="Split to infer on")
    parser.add_argument("--dummy", action="store_true", help="Run offline CPU smoke test (weights=None, 1 batch/step)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    cfg = load_config(config_path)

    image_variant = args.image_variant or str(cfg["preprocessing"]["ghe_ensemble"]["method"])
    epochs = args.epochs if args.epochs is not None else int(cfg["models"]["faster_rcnn"]["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(cfg["models"]["faster_rcnn"]["batch_size"])
    device = select_device(args.device, cfg)
    data_root = resolve_path(cfg["paths"]["data_dummy_processed"] if args.dummy else cfg["paths"]["data_processed"])

    ckpt_path, preds_path, last_loss = train_and_infer_frcnn(
        cfg=cfg,
        data_root=data_root,
        image_variant=image_variant,
        eval_split=args.eval_split,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        score_floor=args.score_floor,
        dummy=args.dummy,
    )

    preds = torch.load(preds_path, weights_only=True)
    summary = {
        "mode": "dummy" if args.dummy else "real",
        "backbone": cfg["models"]["faster_rcnn"]["backbone"],
        "image_variant": image_variant,
        "eval_split": args.eval_split,
        "device": str(device),
        "last_loss": round(last_loss, 4),
        "checkpoint": str(ckpt_path),
        "predictions": str(preds_path),
        "num_eval_images": len(preds),
        "total_detections": sum(int(v["boxes"].shape[0]) for v in preds.values()),
    }
    print("[04_train_infer_frcnn] Completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
