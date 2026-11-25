"""F06: EnPreNiNet Score-Level Fusion Module.

Merges unified predictions from YOLOv5l (Branch 1) and Faster R-CNN ResNet-152
(Branch 2) using:
  1. Optional intra-model NMS (disabled by default since both detectors run native NMS).
  2. Confidence thresholding (default score_floor=0.001 to preserve full PR curve for F07,
     plus support for paper thresholds tau in [0.25, 0.50, 0.75]).
  3. Concatenation + toggleable class-wise Global NMS via `torchvision.ops.batched_nms`
     (supports E4 vs E5 ablation via `--no-global-nms` / `--global-nms`).

Usage:
    python src/05_fusion.py --config configs/config.yaml --dummy --device cpu
    python src/05_fusion.py --config configs/config.yaml --yolo-preds ... --frcnn-preds ... --no-global-nms
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torchvision.ops import batched_nms
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


def empty_prediction() -> dict[str, torch.Tensor]:
    """Return an empty unified prediction entry with exact shapes and dtypes."""
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "scores": torch.empty((0,), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.long),
    }


def _prepare_branch(
    pred: dict[str, torch.Tensor] | None,
    conf_thres: float,
    apply_intra_nms: bool,
    intra_nms_iou: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred is None or pred["boxes"].numel() == 0:
        return (
            torch.empty((0, 4), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    boxes = pred["boxes"].detach().to(device=device, dtype=torch.float32)
    scores = pred["scores"].detach().to(device=device, dtype=torch.float32)
    labels = pred["labels"].detach().to(device=device, dtype=torch.long)

    if apply_intra_nms and boxes.shape[0] > 0:
        keep = batched_nms(boxes, scores, labels, float(intra_nms_iou))
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if boxes.shape[0] > 0:
        mask = scores >= float(conf_thres)
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    return boxes, scores, labels


def fuse_image_predictions(
    yolo_pred: dict[str, torch.Tensor] | None,
    frcnn_pred: dict[str, torch.Tensor] | None,
    conf_thres: float = 0.001,
    global_nms_iou: float = 0.50,
    global_nms_enabled: bool = True,
    apply_intra_nms: bool = False,
    intra_nms_iou: float = 0.50,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Fuse YOLOv5 and Faster R-CNN unified predictions for a single image.

    Uses class-wise `torchvision.ops.batched_nms` when `global_nms_enabled=True`,
    sorts results by descending score, and returns CPU tensors in the unified schema.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")

    y_boxes, y_scores, y_labels = _prepare_branch(
        yolo_pred, conf_thres=conf_thres, apply_intra_nms=apply_intra_nms, intra_nms_iou=intra_nms_iou, device=dev
    )
    f_boxes, f_scores, f_labels = _prepare_branch(
        frcnn_pred, conf_thres=conf_thres, apply_intra_nms=apply_intra_nms, intra_nms_iou=intra_nms_iou, device=dev
    )

    boxes = torch.cat([y_boxes, f_boxes], dim=0)
    scores = torch.cat([y_scores, f_scores], dim=0)
    labels = torch.cat([y_labels, f_labels], dim=0)

    if boxes.shape[0] == 0:
        return empty_prediction()

    if global_nms_enabled:
        keep = batched_nms(boxes, scores, labels, float(global_nms_iou))
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    order = torch.argsort(scores, descending=True)
    return {
        "boxes": boxes[order].detach().cpu().contiguous().to(torch.float32),
        "scores": scores[order].detach().cpu().contiguous().to(torch.float32),
        "labels": labels[order].detach().cpu().contiguous().to(torch.long),
    }


def fuse_prediction_dicts(
    yolo_preds: dict[int, dict[str, torch.Tensor]],
    frcnn_preds: dict[int, dict[str, torch.Tensor]],
    conf_thres: float = 0.001,
    global_nms_iou: float = 0.50,
    global_nms_enabled: bool = True,
    apply_intra_nms: bool = False,
    intra_nms_iou: float = 0.50,
    device: torch.device | str | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Verify exact image_id alignment across branches and fuse all images."""
    yolo_keys = set(yolo_preds.keys())
    frcnn_keys = set(frcnn_preds.keys())
    if yolo_keys != frcnn_keys:
        only_yolo = sorted(yolo_keys - frcnn_keys)
        only_frcnn = sorted(frcnn_keys - yolo_keys)
        raise ValueError(
            f"Mismatched image_id keys between YOLO and Faster R-CNN predictions: "
            f"only_in_yolo={only_yolo[:5]}, only_in_frcnn={only_frcnn[:5]}"
        )

    fused: dict[int, dict[str, torch.Tensor]] = {}
    for img_id in sorted(yolo_keys):
        fused[int(img_id)] = fuse_image_predictions(
            yolo_pred=yolo_preds[img_id],
            frcnn_pred=frcnn_preds[img_id],
            conf_thres=conf_thres,
            global_nms_iou=global_nms_iou,
            global_nms_enabled=global_nms_enabled,
            apply_intra_nms=apply_intra_nms,
            intra_nms_iou=intra_nms_iou,
            device=device,
        )
    return fused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EnPreNiNet Score-Level Fusion (YOLOv5 + Faster R-CNN).")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    parser.add_argument(
        "--image-variant",
        type=str,
        choices=VALID_IMAGE_VARIANTS,
        default=None,
        help="Input image variant (default: preprocessing.ghe_ensemble.method)",
    )
    parser.add_argument("--yolo-preds", type=Path, default=None, help="Path to YOLOv5 unified predictions .pt")
    parser.add_argument("--frcnn-preds", type=Path, default=None, help="Path to Faster R-CNN unified predictions .pt")
    parser.add_argument("--output", type=Path, default=None, help="Output path for fused predictions .pt")
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.001,
        help="Confidence threshold (default 0.001 preserves full spectrum for F07 AP computation)",
    )
    nms_group = parser.add_mutually_exclusive_group()
    nms_group.add_argument("--global-nms", dest="global_nms", action="store_true", help="Enable class-wise Global NMS")
    nms_group.add_argument(
        "--no-global-nms", dest="global_nms", action="store_false", help="Disable Global NMS (E4 ablation)"
    )
    parser.set_defaults(global_nms=None)
    parser.add_argument(
        "--apply-intra-nms",
        action="store_true",
        default=False,
        help="Apply extra intra-model NMS per branch (default False since detectors apply native NMS)",
    )
    parser.add_argument("--intra-nms-iou", type=float, default=None, help="Override fusion.intra_nms_iou")
    parser.add_argument("--global-nms-iou", type=float, default=None, help="Override fusion.global_nms_iou")
    parser.add_argument("--eval-split", type=str, choices=("val", "test"), default="test", help="Split name")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, or auto")
    parser.add_argument("--dummy", action="store_true", help="Run on isolated dummy predictions under runs/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    cfg = load_config(config_path)

    fusion_cfg = cfg["fusion"]
    image_variant = args.image_variant or str(cfg["preprocessing"]["ghe_ensemble"]["method"])
    global_nms_enabled = (
        bool(args.global_nms) if args.global_nms is not None else bool(fusion_cfg["global_nms_enabled"])
    )
    intra_nms_iou = (
        float(args.intra_nms_iou) if args.intra_nms_iou is not None else float(fusion_cfg["intra_nms_iou"])
    )
    global_nms_iou = (
        float(args.global_nms_iou) if args.global_nms_iou is not None else float(fusion_cfg["global_nms_iou"])
    )
    device = select_device(args.device, cfg)
    runs_dir = resolve_path(cfg["paths"]["runs_dir"])

    prefix = "dummy_" if args.dummy else ""
    yolo_path = args.yolo_preds or (
        runs_dir / f"{prefix}yolo_{image_variant}" / f"yolo_{image_variant}_{args.eval_split}_preds.pt"
    )
    frcnn_path = args.frcnn_preds or (
        runs_dir / f"{prefix}frcnn_{image_variant}" / f"frcnn_{image_variant}_{args.eval_split}_preds.pt"
    )
    output_path = args.output or (
        runs_dir / f"{prefix}fusion_{image_variant}" / f"fusion_{image_variant}_{args.eval_split}_preds.pt"
    )

    if not yolo_path.is_file():
        raise FileNotFoundError(f"YOLOv5 predictions not found: {yolo_path}")
    if not frcnn_path.is_file():
        raise FileNotFoundError(f"Faster R-CNN predictions not found: {frcnn_path}")

    yolo_preds = torch.load(yolo_path, weights_only=True)
    frcnn_preds = torch.load(frcnn_path, weights_only=True)

    fused_preds = fuse_prediction_dicts(
        yolo_preds=yolo_preds,
        frcnn_preds=frcnn_preds,
        conf_thres=args.conf_thres,
        global_nms_iou=global_nms_iou,
        global_nms_enabled=global_nms_enabled,
        apply_intra_nms=args.apply_intra_nms,
        intra_nms_iou=intra_nms_iou,
        device=device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fused_preds, output_path)

    # Also save thresholded snapshots for paper tau values [0.25, 0.50, 0.75]
    tau_counts: dict[str, int] = {}
    for tau in fusion_cfg["confidence_thresholds"]:
        tau_val = float(tau)
        tau_preds = fuse_prediction_dicts(
            yolo_preds=yolo_preds,
            frcnn_preds=frcnn_preds,
            conf_thres=tau_val,
            global_nms_iou=global_nms_iou,
            global_nms_enabled=global_nms_enabled,
            apply_intra_nms=args.apply_intra_nms,
            intra_nms_iou=intra_nms_iou,
            device=device,
        )
        tau_tag = int(round(tau_val * 100))
        tau_path = output_path.parent / f"fusion_{image_variant}_{args.eval_split}_tau{tau_tag:02d}_preds.pt"
        torch.save(tau_preds, tau_path)
        tau_counts[f"tau_{tau_val:.2f}"] = sum(int(v["boxes"].shape[0]) for v in tau_preds.values())

    summary = {
        "mode": "dummy" if args.dummy else "real",
        "image_variant": image_variant,
        "eval_split": args.eval_split,
        "global_nms_enabled": global_nms_enabled,
        "global_nms_iou": global_nms_iou,
        "apply_intra_nms": args.apply_intra_nms,
        "conf_thres": args.conf_thres,
        "num_eval_images": len(fused_preds),
        "total_detections": sum(int(v["boxes"].shape[0]) for v in fused_preds.values()),
        "thresholded_detection_counts": tau_counts,
        "output": str(output_path),
    }
    print("[05_fusion] Completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
