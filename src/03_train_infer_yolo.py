"""F04: YOLOv5l Training & Standard Inference Wrapper.

Wraps YOLOv5l (CSPDarknet + PANet) training and inference, exporting test/val
predictions in the EnPreNiNet unified prediction schema:
    {
        image_id: {
            "boxes": FloatTensor[N, 4],  # xyxy in [0, 640]
            "scores": FloatTensor[N],    # detector confidence score in [0, 1]
            "labels": LongTensor[N],     # canonical class IDs in {0, 1, 2}
        }
    }

Isolates all outputs under `--run-id` (`runs/<run_id>/yolo_<variant>/`,
`weights/<run_id>/yolov5l_<variant>.pt`), supports `--work-dir` for local SSD
YOLO dataset views, and validates `ghe_meta.json["dataset_fingerprint"]` before
training/inferring on any GHE variant.

Usage:
    python src/03_train_infer_yolo.py --config configs/config.yaml --dummy --epochs 1 --device cpu
    python src/03_train_infer_yolo.py --config configs/config.yaml --run-id pilot --yolo-repo /content/yolov5 --work-dir /content/work/yolo_views/pilot --image-variant standard_local_he
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_IMAGE_VARIANTS = ("raw", "fig4_literal", "fig4_normalized", "standard_local_he")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def resolve_yolo_output_paths(
    cfg: dict[str, Any],
    run_id: str,
    image_variant: str,
    eval_split: str,
    dummy: bool = False,
) -> tuple[Path, Path, Path]:
    """Return isolated (run_dir, ckpt_path, preds_path) under runs/<run_id>/ and weights/<run_id>/."""
    run_dir = resolve_path(cfg["paths"]["runs_dir"]) / run_id / f"yolo_{image_variant}"
    preds_path = run_dir / f"yolo_{image_variant}_{eval_split}_preds.pt"
    if dummy:
        ckpt_path = run_dir / f"yolov5l_dummy_{image_variant}.pt"
    else:
        variant_name = str(cfg["models"]["yolov5"]["variant"])
        ckpt_path = resolve_path(cfg["paths"]["weights_dir"]) / run_id / f"{variant_name}_{image_variant}.pt"
    return run_dir, ckpt_path, preds_path


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


def load_coco_split_images(coco_json_path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not coco_json_path.is_file():
        raise FileNotFoundError(f"COCO split file not found: {coco_json_path}")
    data = json.loads(coco_json_path.read_text(encoding="utf-8"))
    fp = (data.get("info") or {}).get("dataset_fingerprint")
    return list(data.get("images", [])), (str(fp) if fp else None)


def verify_ghe_metadata(
    data_root: Path,
    image_variant: str,
    expected_dataset_fp: str | None,
) -> dict[str, Any] | None:
    """A03 fix: verify GHE images match the current dataset_fingerprint before training/inference."""
    if image_variant == "raw":
        return None
    ghe_meta_path = data_root / "images" / image_variant / "ghe_meta.json"
    if not ghe_meta_path.is_file():
        raise FileNotFoundError(
            f"Missing GHE metadata file: {ghe_meta_path}. Re-run src/02_ghe_ensemble.py for '{image_variant}'."
        )
    ghe_meta = json.loads(ghe_meta_path.read_text(encoding="utf-8"))
    ghe_fp = ghe_meta.get("dataset_fingerprint")
    if not expected_dataset_fp or not ghe_fp:
        raise ValueError(
            f"Missing dataset_fingerprint in COCO ({expected_dataset_fp}) or {ghe_meta_path} ({ghe_fp})."
        )
    if str(ghe_fp) != str(expected_dataset_fp):
        raise ValueError(
            f"Stale GHE images detected for '{image_variant}': "
            f"ghe_meta.dataset_fingerprint={ghe_fp} != coco.dataset_fingerprint={expected_dataset_fp}. "
            f"Re-run src/02_ghe_ensemble.py before training."
        )
    return ghe_meta


def _read_git_commit(repo_dir: Path) -> str | None:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip() or None
    except Exception:
        return None


def save_prediction_meta(preds_path: Path, meta: dict[str, Any]) -> Path:
    meta_path = preds_path.with_name(f"{preds_path.name}.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def empty_prediction() -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "scores": torch.empty((0,), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.long),
    }


def standardize_yolo_predictions(
    raw_dets: torch.Tensor | np.ndarray | None,
    img_size: int = 640,
    score_floor: float = 0.001,
    num_classes: int = 3,
) -> dict[str, torch.Tensor]:
    """Convert raw YOLO detections [N, 6] (x1, y1, x2, y2, score, cls) into unified schema."""
    if raw_dets is None:
        return empty_prediction()

    det_tensor = torch.as_tensor(raw_dets, dtype=torch.float32).cpu()
    if det_tensor.numel() == 0:
        return empty_prediction()
    if det_tensor.ndim != 2 or det_tensor.shape[1] != 6:
        raise ValueError(f"Expected raw_dets of shape (N, 6), got {tuple(det_tensor.shape)}")

    boxes = det_tensor[:, :4].clone()
    scores = det_tensor[:, 4].clone().clamp(0.0, 1.0)
    labels = det_tensor[:, 5].round().to(torch.long)

    boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(img_size))
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(img_size))

    valid_mask = (
        (scores >= float(score_floor))
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
        & (labels >= 0)
        & (labels < int(num_classes))
    )
    if not bool(valid_mask.any()):
        return empty_prediction()

    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    labels = labels[valid_mask]

    order = torch.argsort(scores, descending=True)
    return {
        "boxes": boxes[order].contiguous().to(torch.float32),
        "scores": scores[order].contiguous().to(torch.float32),
        "labels": labels[order].contiguous().to(torch.long),
    }


class DummyYoloDetector(nn.Module):
    """Lightweight offline PyTorch detector used exclusively in --dummy --device cpu smoke tests."""

    def __init__(self, num_classes: int = 3, max_dets: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.max_dets = max_dets
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(16, max_dets * (5 + num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)
        return self.head(feat).view(x.shape[0], self.max_dets, 5 + self.num_classes)


def _read_image_tensor(img_path: Path, img_size: int) -> torch.Tensor:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    if img.shape[:2] != (img_size, img_size):
        raise ValueError(f"Expected image shape ({img_size}, {img_size}), got {img.shape[:2]} for {img_path}")
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def run_dummy_yolo(
    cfg: dict[str, Any],
    data_root: Path,
    run_id: str,
    image_variant: str,
    eval_split: str,
    epochs: int,
    device: torch.device,
    score_floor: float,
) -> tuple[Path, Path]:
    """Offline 1-batch/step CPU smoke test for F04."""
    seed = int(cfg["project"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    img_size = int(cfg["models"]["yolov5"]["img_size"])
    num_classes = len(cfg["dataset"]["classes"])
    coco_dir = data_root / cfg["paths"]["coco_subdir"]

    train_images, _ = load_coco_split_images(coco_dir / "instances_train.json")
    eval_images, dataset_fp = load_coco_split_images(coco_dir / f"instances_{eval_split}.json")
    ghe_meta = verify_ghe_metadata(data_root, image_variant, dataset_fp)
    if not train_images:
        raise RuntimeError("No training images found in dummy dataset. Run F02/F03 first.")

    batch_tensors = [
        _read_image_tensor(resolve_variant_image(data_root, image_variant, "train", entry["file_name"]), img_size)
        for entry in train_images[:2]
    ]
    batch_x = torch.stack(batch_tensors, dim=0).to(device)

    model = DummyYoloDetector(num_classes=num_classes, max_dets=3).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    model.train()
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        out = model(batch_x)
        target = torch.zeros_like(out)
        target[..., 0:2] = 0.5
        target[..., 2:4] = 0.2
        target[..., 4] = 0.7
        loss = nn.functional.mse_loss(torch.sigmoid(out[..., :5]), target[..., :5])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite dummy YOLO loss: {loss.item()}")
        loss.backward()
        optimizer.step()

    run_dir, ckpt_path, preds_path = resolve_yolo_output_paths(
        cfg, run_id=run_id, image_variant=image_variant, eval_split=eval_split, dummy=True
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "variant": cfg["models"]["yolov5"]["variant"],
            "run_id": run_id,
            "image_variant": image_variant,
            "img_size": img_size,
            "num_classes": num_classes,
            "dataset_fingerprint": dataset_fp,
            "preprocessing_fingerprint": (ghe_meta or {}).get("preprocessing_fingerprint"),
            "dummy": True,
        },
        ckpt_path,
    )

    model.eval()
    preds_by_image: dict[int, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for entry in eval_images:
            img_id = int(entry["id"])
            img_path = resolve_variant_image(data_root, image_variant, eval_split, entry["file_name"])
            x = _read_image_tensor(img_path, img_size).unsqueeze(0).to(device)
            raw_out = model(x)[0]

            cx = torch.sigmoid(raw_out[:, 0]) * float(img_size)
            cy = torch.sigmoid(raw_out[:, 1]) * float(img_size)
            w = torch.sigmoid(raw_out[:, 2]).clamp(min=0.05, max=0.5) * float(img_size)
            h = torch.sigmoid(raw_out[:, 3]).clamp(min=0.05, max=0.5) * float(img_size)
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            obj_conf = torch.sigmoid(raw_out[:, 4])
            cls_probs = torch.softmax(raw_out[:, 5:], dim=-1)
            cls_conf, cls_ids = cls_probs.max(dim=-1)
            scores = obj_conf * cls_conf

            raw_dets = torch.stack([x1, y1, x2, y2, scores, cls_ids.float()], dim=-1)
            preds_by_image[img_id] = standardize_yolo_predictions(
                raw_dets=raw_dets,
                img_size=img_size,
                score_floor=score_floor,
                num_classes=num_classes,
            )

    torch.save(preds_by_image, preds_path)
    save_prediction_meta(
        preds_path,
        {
            "model": "dummy_yolo",
            "run_id": run_id,
            "image_variant": image_variant,
            "eval_split": eval_split,
            "score_floor": score_floor,
            "dataset_fingerprint": dataset_fp,
            "preprocessing_fingerprint": (ghe_meta or {}).get("preprocessing_fingerprint"),
            "num_eval_images": len(preds_by_image),
            "dummy": True,
        },
    )
    return ckpt_path, preds_path


def _prepare_yolo_view(
    cfg: dict[str, Any],
    data_root: Path,
    image_variant: str,
    view_root: Path,
) -> Path:
    """Create a fresh YOLOv5-compatible dataset view strictly from current split manifests."""
    if view_root.exists():
        shutil.rmtree(view_root)

    splits_root = data_root / cfg["paths"]["splits_subdir"]
    yolo_labels_root = data_root / cfg["paths"]["yolo_subdir"]

    for split in ("train", "val", "test"):
        manifest_path = splits_root / f"{split}.txt"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing split manifest: {manifest_path}")

        dst_img_dir = view_root / "images" / split
        dst_lbl_dir = view_root / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        entries = [ln.strip() for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for rel_entry in entries:
            file_name = Path(rel_entry).name
            stem = Path(file_name).stem
            src_img = resolve_variant_image(data_root, image_variant, split, file_name)
            src_lbl = yolo_labels_root / split / f"{stem}.txt"
            if not src_lbl.is_file():
                raise FileNotFoundError(f"Missing YOLO label for {file_name}: {src_lbl}")
            shutil.copy2(src_img, dst_img_dir / src_img.name)
            shutil.copy2(src_lbl, dst_lbl_dir / src_lbl.name)

    classes = {int(k): str(v) for k, v in cfg["dataset"]["classes"].items()}
    yaml_payload = {
        "path": str(view_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes,
    }
    dataset_yaml = view_root / f"dataset_{image_variant}.yaml"
    with dataset_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_payload, f, sort_keys=False)
    return dataset_yaml


def _parse_yolo_txt_to_xyxy(txt_path: Path, img_size: int) -> np.ndarray | None:
    if not txt_path.is_file():
        return None
    rows: list[list[float]] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        cls_id = float(parts[0])
        cx, cy, w, h, conf = (float(v) for v in parts[1:6])
        x1 = (cx - w / 2.0) * img_size
        y1 = (cy - h / 2.0) * img_size
        x2 = (cx + w / 2.0) * img_size
        y2 = (cy + h / 2.0) * img_size
        rows.append([x1, y1, x2, y2, conf, cls_id])
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float32)


def run_real_yolo(
    cfg: dict[str, Any],
    data_root: Path,
    run_id: str,
    yolo_repo: Path,
    image_variant: str,
    eval_split: str,
    epochs: int,
    batch_size: int,
    device_str: str,
    score_floor: float,
    work_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Train YOLOv5l via local ultralytics/yolov5 repo and export unified predictions."""
    train_py = yolo_repo / "train.py"
    detect_py = yolo_repo / "detect.py"
    if not train_py.is_file() or not detect_py.is_file():
        raise FileNotFoundError(f"Expected train.py and detect.py inside --yolo-repo: {yolo_repo}")

    coco_dir = data_root / cfg["paths"]["coco_subdir"]
    eval_images, dataset_fp = load_coco_split_images(coco_dir / f"instances_{eval_split}.json")
    ghe_meta = verify_ghe_metadata(data_root, image_variant, dataset_fp)

    seed = int(cfg["project"]["seed"])
    yolo_cfg = cfg["models"]["yolov5"]
    img_size = int(yolo_cfg["img_size"])
    num_classes = len(cfg["dataset"]["classes"])
    intra_nms_iou = float(cfg["fusion"]["intra_nms_iou"])

    run_dir, exported_weights, preds_path = resolve_yolo_output_paths(
        cfg, run_id=run_id, image_variant=image_variant, eval_split=eval_split, dummy=False
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    exported_weights.parent.mkdir(parents=True, exist_ok=True)

    # A05 fix: allow local SSD work_dir for yolo_view so YOLOv5 does not read images from Google Drive
    view_root = (work_dir / f"yolo_view_{image_variant}") if work_dir is not None else (run_dir / "yolo_view")
    dataset_yaml = _prepare_yolo_view(cfg, data_root, image_variant, view_root)

    train_out_dir = run_dir / "train"
    if train_out_dir.exists():
        shutil.rmtree(train_out_dir)

    train_cmd = [
        sys.executable,
        str(train_py),
        "--img",
        str(img_size),
        "--batch",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--data",
        str(dataset_yaml),
        "--weights",
        f"{yolo_cfg['variant']}.pt",
        "--project",
        str(run_dir),
        "--name",
        "train",
        "--exist-ok",
        "--device",
        device_str,
    ]
    subprocess.run(train_cmd, check=True)

    best_pt = run_dir / "train" / "weights" / "best.pt"
    if best_pt.is_file():
        shutil.copy2(best_pt, exported_weights)
    else:
        raise FileNotFoundError(f"YOLOv5 training finished without producing {best_pt}")

    infer_dir = run_dir / f"infer_{eval_split}"
    if infer_dir.exists():
        shutil.rmtree(infer_dir)

    eval_img_dir = view_root / "images" / eval_split
    detect_cmd = [
        sys.executable,
        str(detect_py),
        "--weights",
        str(exported_weights),
        "--source",
        str(eval_img_dir),
        "--imgsz",
        str(img_size),
        "--conf-thres",
        str(score_floor),
        "--iou-thres",
        str(intra_nms_iou),
        "--save-txt",
        "--save-conf",
        "--nosave",
        "--project",
        str(run_dir),
        "--name",
        f"infer_{eval_split}",
        "--exist-ok",
        "--device",
        device_str,
    ]
    subprocess.run(detect_cmd, check=True)

    labels_out_dir = infer_dir / "labels"
    preds_by_image: dict[int, dict[str, torch.Tensor]] = {}
    for entry in eval_images:
        img_id = int(entry["id"])
        stem = Path(entry["file_name"]).stem
        raw_dets = _parse_yolo_txt_to_xyxy(labels_out_dir / f"{stem}.txt", img_size=img_size)
        preds_by_image[img_id] = standardize_yolo_predictions(
            raw_dets=raw_dets,
            img_size=img_size,
            score_floor=score_floor,
            num_classes=num_classes,
        )

    torch.save(preds_by_image, preds_path)
    save_prediction_meta(
        preds_path,
        {
            "model": str(yolo_cfg["variant"]),
            "run_id": run_id,
            "image_variant": image_variant,
            "eval_split": eval_split,
            "score_floor": score_floor,
            "seed": seed,
            "dataset_fingerprint": dataset_fp,
            "preprocessing_fingerprint": (ghe_meta or {}).get("preprocessing_fingerprint"),
            "yolo_repo_commit": _read_git_commit(yolo_repo),
            "num_eval_images": len(preds_by_image),
            "dummy": False,
        },
    )
    return exported_weights, preds_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train & infer YOLOv5l and export unified predictions.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    parser.add_argument("--run-id", type=str, default=None, help="Run namespace isolating outputs (dummy, pilot, full)")
    parser.add_argument(
        "--image-variant",
        type=str,
        choices=VALID_IMAGE_VARIANTS,
        default=None,
        help="Input image variant (default: preprocessing.ghe_ensemble.method)",
    )
    parser.add_argument("--yolo-repo", type=Path, default=None, help="Path to local ultralytics/yolov5 repository")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Local SSD directory for temporary YOLO dataset view (avoids Google Drive I/O)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, or auto")
    parser.add_argument("--score-floor", type=float, default=0.001, help="Minimum confidence score kept before fusion")
    parser.add_argument("--eval-split", type=str, choices=("val", "test"), default="test", help="Split to infer on")
    parser.add_argument("--data-root", type=Path, default=None, help="Override processed data root directory")
    parser.add_argument("--dummy", action="store_true", help="Run offline CPU smoke test on data/dummy_processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    cfg = load_config(config_path)

    run_id = args.run_id or ("dummy" if args.dummy else "full")
    image_variant = args.image_variant or str(cfg["preprocessing"]["ghe_ensemble"]["method"])
    epochs = args.epochs if args.epochs is not None else int(cfg["models"]["yolov5"]["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(cfg["models"]["yolov5"]["batch_size"])
    device = select_device(args.device, cfg)

    if args.dummy:
        data_root = args.data_root or resolve_path(cfg["paths"]["data_dummy_processed"])
        ckpt_path, preds_path = run_dummy_yolo(
            cfg=cfg,
            data_root=data_root,
            run_id=run_id,
            image_variant=image_variant,
            eval_split=args.eval_split,
            epochs=epochs,
            device=device,
            score_floor=args.score_floor,
        )
    else:
        if args.yolo_repo is None:
            raise ValueError("--yolo-repo is required when running without --dummy")
        data_root = args.data_root or resolve_path(cfg["paths"]["data_processed"])
        yolo_device = "cpu" if device.type == "cpu" else "0"
        ckpt_path, preds_path = run_real_yolo(
            cfg=cfg,
            data_root=data_root,
            run_id=run_id,
            yolo_repo=args.yolo_repo,
            image_variant=image_variant,
            eval_split=args.eval_split,
            epochs=epochs,
            batch_size=batch_size,
            device_str=yolo_device,
            score_floor=args.score_floor,
            work_dir=args.work_dir,
        )

    preds = torch.load(preds_path, weights_only=True)
    summary = {
        "mode": "dummy" if args.dummy else "real",
        "run_id": run_id,
        "image_variant": image_variant,
        "eval_split": args.eval_split,
        "device": str(device),
        "checkpoint": str(ckpt_path),
        "predictions": str(preds_path),
        "num_eval_images": len(preds),
        "total_detections": sum(int(v["boxes"].shape[0]) for v in preds.values()),
    }
    print("[03_train_infer_yolo] Completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
