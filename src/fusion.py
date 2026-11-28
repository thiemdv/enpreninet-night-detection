"""F06: EnPreNiNet Score-Level Fusion Module.

Merges unified predictions from YOLOv5l (Branch 1) and Faster R-CNN ResNet-152
(Branch 2) using:
  1. Fail-closed sidecar metadata verification (dataset_fingerprint, image_variant,
     eval_split, preprocessing_fingerprint; opt-in `--allow-legacy-predictions`).
  2. Strict image_id key alignment across both branches.
  3. Optional intra-model NMS + confidence thresholding + toggleable class-wise Global NMS.
  4. Collision-free confidence snapshot filenames + companion `.meta.json` for every `.pt`.

Usage:
    python src/fusion.py --config configs/config.yaml --dummy --device cpu
    python src/fusion.py --config configs/config.yaml --run-id pilot --global-nms --device cuda
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
VALID_IMAGE_VARIANTS = (
    "raw",
    "fig4_literal",
    "fig4_normalized",
    "standard_local_he",
    "tiled_ghe_gray",
    "tiled_ghe_lab",
)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def resolve_fusion_paths(
    cfg: dict[str, Any],
    run_id: str,
    image_variant: str,
    eval_split: str,
) -> tuple[Path, Path, Path]:
    """Return isolated (yolo_path, frcnn_path, fusion_output_path) under runs/<run_id>/."""
    run_root = resolve_path(cfg["paths"]["runs_dir"]) / run_id
    yolo_path = run_root / f"yolo_{image_variant}" / f"yolo_{image_variant}_{eval_split}_preds.pt"
    frcnn_path = run_root / f"frcnn_{image_variant}" / f"frcnn_{image_variant}_{eval_split}_preds.pt"
    out_path = run_root / f"fusion_{image_variant}" / f"fusion_{image_variant}_{eval_split}_preds.pt"
    return yolo_path, frcnn_path, out_path


def select_device(device_arg: str | None, cfg: dict[str, Any]) -> torch.device:
    req = (device_arg or str(cfg["project"]["device"])).lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def read_prediction_meta(preds_path: Path) -> dict[str, Any] | None:
    meta_path = preds_path.with_name(f"{preds_path.name}.meta.json")
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def save_prediction_meta(preds_path: Path, meta: dict[str, Any]) -> Path:
    meta_path = preds_path.with_name(f"{preds_path.name}.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def snapshot_path_for_tau(output_path: Path, tau: float) -> Path:
    """Derive a collision-free confidence-threshold snapshot path from output_path.stem."""
    tau_tag = int(round(float(tau) * 100))
    return output_path.with_name(f"{output_path.stem}_tau{tau_tag:02d}.pt")


def empty_prediction() -> dict[str, torch.Tensor]:
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
    """Fuse YOLOv5 and Faster R-CNN unified predictions for a single image."""
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


def verify_and_resolve_branch_metadata(
    yolo_path: Path,
    frcnn_path: Path,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """A02 fail-closed check: require matching dataset_fingerprint, image_variant, and eval_split."""
    yolo_meta = read_prediction_meta(yolo_path)
    frcnn_meta = read_prediction_meta(frcnn_path)

    if not allow_legacy:
        if yolo_meta is None or frcnn_meta is None:
            raise ValueError(
                f"Missing prediction sidecar metadata (.meta.json) for YOLO ({yolo_path}) or "
                f"Faster R-CNN ({frcnn_path}). Pass --allow-legacy-predictions to override."
            )

    ym = yolo_meta or {}
    fm = frcnn_meta or {}
    fp_yolo = ym.get("dataset_fingerprint")
    fp_frcnn = fm.get("dataset_fingerprint")

    if not allow_legacy and (not fp_yolo or not fp_frcnn):
        raise ValueError(
            f"Missing dataset_fingerprint: yolo={fp_yolo}, frcnn={fp_frcnn}. "
            f"Pass --allow-legacy-predictions to override."
        )
    if fp_yolo and fp_frcnn and str(fp_yolo) != str(fp_frcnn):
        raise ValueError(
            f"Dataset fingerprint mismatch between YOLO ({fp_yolo}) and Faster R-CNN ({fp_frcnn})."
        )

    var_yolo = ym.get("image_variant")
    var_frcnn = fm.get("image_variant")
    if var_yolo and var_frcnn and str(var_yolo) != str(var_frcnn):
        raise ValueError(
            f"Image variant mismatch between YOLO ({var_yolo}) and Faster R-CNN ({var_frcnn})."
        )

    split_yolo = ym.get("eval_split")
    split_frcnn = fm.get("eval_split")
    if split_yolo and split_frcnn and str(split_yolo) != str(split_frcnn):
        raise ValueError(
            f"Eval split mismatch between YOLO ({split_yolo}) and Faster R-CNN ({split_frcnn})."
        )

    pfp_yolo = ym.get("preprocessing_fingerprint")
    pfp_frcnn = fm.get("preprocessing_fingerprint")
    if pfp_yolo and pfp_frcnn and str(pfp_yolo) != str(pfp_frcnn):
        raise ValueError(
            f"Preprocessing fingerprint mismatch between YOLO ({pfp_yolo}) and Faster R-CNN ({pfp_frcnn})."
        )

    return {
        "dataset_fingerprint": str(fp_yolo) if (fp_yolo and fp_frcnn) else (str(fp_yolo or fp_frcnn) if allow_legacy and (fp_yolo or fp_frcnn) else None),
        "preprocessing_fingerprint": str(pfp_yolo or pfp_frcnn) if (pfp_yolo or pfp_frcnn) else None,
        "image_variant": str(var_yolo or var_frcnn) if (var_yolo or var_frcnn) else None,
        "eval_split": str(split_yolo or split_frcnn) if (split_yolo or split_frcnn) else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EnPreNiNet Score-Level Fusion (YOLOv5 + Faster R-CNN).")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    parser.add_argument("--run-id", type=str, default=None, help="Run namespace isolating outputs (dummy, pilot, full)")
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
    parser.add_argument(
        "--allow-legacy-predictions",
        action="store_true",
        default=False,
        help="Allow fusing prediction .pt files lacking .meta.json fingerprints",
    )
    parser.add_argument("--dummy", action="store_true", help="Run on isolated runs/dummy/ predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    cfg = load_config(config_path)

    fusion_cfg = cfg["fusion"]
    run_id = args.run_id or ("dummy" if args.dummy else "full")
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

    default_yolo, default_frcnn, default_out = resolve_fusion_paths(
        cfg, run_id=run_id, image_variant=image_variant, eval_split=args.eval_split
    )
    yolo_path = args.yolo_preds or default_yolo
    frcnn_path = args.frcnn_preds or default_frcnn
    output_path = args.output or default_out

    if not yolo_path.is_file():
        raise FileNotFoundError(f"YOLOv5 predictions not found: {yolo_path}")
    if not frcnn_path.is_file():
        raise FileNotFoundError(f"Faster R-CNN predictions not found: {frcnn_path}")

    branch_meta = verify_and_resolve_branch_metadata(
        yolo_path, frcnn_path, allow_legacy=args.allow_legacy_predictions
    )
    dataset_fp = branch_meta["dataset_fingerprint"]
    preproc_fp = branch_meta["preprocessing_fingerprint"]

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
    base_meta = {
        "model": "enpreninet_fusion",
        "run_id": run_id,
        "image_variant": image_variant,
        "eval_split": args.eval_split,
        "global_nms_enabled": global_nms_enabled,
        "global_nms_iou": global_nms_iou,
        "apply_intra_nms": args.apply_intra_nms,
        "conf_thres": args.conf_thres,
        "dataset_fingerprint": dataset_fp,
        "preprocessing_fingerprint": preproc_fp,
        "num_eval_images": len(fused_preds),
        "dummy": args.dummy,
    }
    save_prediction_meta(output_path, base_meta)

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
        tau_path = snapshot_path_for_tau(output_path, tau_val)
        torch.save(tau_preds, tau_path)
        # A08 fix: emit companion .meta.json for every confidence snapshot too
        save_prediction_meta(tau_path, {**base_meta, "conf_thres": tau_val, "snapshot": True})
        tau_counts[f"tau_{tau_val:.2f}"] = sum(int(v["boxes"].shape[0]) for v in tau_preds.values())

    summary = {
        "mode": "dummy" if args.dummy else "real",
        "run_id": run_id,
        "image_variant": image_variant,
        "eval_split": args.eval_split,
        "global_nms_enabled": global_nms_enabled,
        "global_nms_iou": global_nms_iou,
        "apply_intra_nms": args.apply_intra_nms,
        "conf_thres": args.conf_thres,
        "dataset_fingerprint": dataset_fp,
        "num_eval_images": len(fused_preds),
        "total_detections": sum(int(v["boxes"].shape[0]) for v in fused_preds.values()),
        "thresholded_detection_counts": tau_counts,
        "output": str(output_path),
    }
    print("[fusion] Completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
