"""F02: BDD100K Night Filter & Dual-Format Exporter (YOLO + COCO).

Filters BDD100K frames where timeofday == 'night' (retaining negative samples),
maps target classes {car: 0, person: 1, traffic light: 2}, splits 70/20/10
deterministically, resizes images to 640x640 into images/raw/{train,val,test}/,
and exports split manifests, YOLO .txt labels, and COCO instances_*.json.

Usage:
    python src/01_prepare_bdd.py --config configs/config.yaml --dummy
    python src/01_prepare_bdd.py --config configs/config.yaml --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def generate_dummy_bdd_dataset(dummy_dir: Path, seed: int = 42) -> Path:
    """Create a realistic 12-frame BDD100K-schema dummy dataset under data/dummy/.

    Includes:
    - 10 night frames (1280x720): 8 with target box2d + poly2d + out-of-scope classes,
      1 with labels=None (negative sample), 1 with only out-of-scope/poly2d (negative sample).
    - 2 daytime frames (1280x720) that must be filtered out.
    """
    images_dir = dummy_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = dummy_dir / "labels_dummy.json"

    rng = np.random.default_rng(seed)
    width, height = 1280, 720
    frames: list[dict[str, Any]] = []

    def _make_image(name: str, is_night: bool) -> None:
        img_path = images_dir / name
        if is_night:
            # Low-light synthetic night frame with slight road/headlight structure
            img = rng.integers(5, 45, size=(height, width, 3), dtype=np.uint8)
            cv2.rectangle(img, (200, 320), (420, 480), (90, 95, 110), -1)
            cv2.circle(img, (900, 140), 18, (60, 190, 220), -1)
        else:
            img = rng.integers(120, 220, size=(height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

    # 8 positive night frames (with target classes + poly2d + out-of-scope classes)
    for idx in range(8):
        name = f"night_{idx:02d}.jpg"
        _make_image(name, is_night=True)
        labels: list[dict[str, Any]] = [
            {
                "id": idx * 10 + 1,
                "category": "car",
                "box2d": {
                    "x1": 0.0 if idx == 0 else float(120 + idx * 40),
                    "y1": float(300 + (idx % 3) * 20),
                    "x2": float(360 + idx * 40),
                    "y2": float(500 + (idx % 3) * 20),
                },
            },
            {
                "id": idx * 10 + 2,
                "category": "person",
                "box2d": {
                    "x1": float(520 + idx * 30),
                    "y1": 280.0,
                    "x2": float(590 + idx * 30),
                    "y2": 490.0,
                },
            },
            {
                "id": idx * 10 + 3,
                "category": "traffic light",
                "box2d": {
                    "x1": float(880 + idx * 15),
                    "y1": 80.0,
                    "x2": float(930 + idx * 15),
                    "y2": 180.0,
                },
            },
            # Out-of-scope box2d class (must be ignored)
            {
                "id": idx * 10 + 4,
                "category": "truck",
                "box2d": {"x1": 60.0, "y1": 220.0, "x2": 260.0, "y2": 460.0},
            },
            # poly2d annotation without box2d (must be ignored)
            {
                "id": idx * 10 + 5,
                "category": "drivable area",
                "poly2d": [
                    {
                        "vertices": [[100.0, 700.0], [500.0, 420.0], [900.0, 700.0]],
                        "types": "LLL",
                        "closed": True,
                    }
                ],
            },
        ]
        frames.append(
            {
                "name": name,
                "attributes": {"weather": "clear", "scene": "city street", "timeofday": "night"},
                "labels": labels,
            }
        )

    # Night frame 8: labels=None (negative sample)
    _make_image("night_08.jpg", is_night=True)
    frames.append(
        {
            "name": "night_08.jpg",
            "attributes": {"weather": "clear", "scene": "highway", "timeofday": "night"},
            "labels": None,
        }
    )

    # Night frame 9: only out-of-scope class + poly2d (0 target objects -> negative sample)
    _make_image("night_09.jpg", is_night=True)
    frames.append(
        {
            "name": "night_09.jpg",
            "attributes": {"weather": "clear", "scene": "city street", "timeofday": "night"},
            "labels": [
                {
                    "id": 901,
                    "category": "traffic sign",
                    "box2d": {"x1": 1000.0, "y1": 150.0, "x2": 1100.0, "y2": 240.0},
                },
                {
                    "id": 902,
                    "category": "lane",
                    "poly2d": [{"vertices": [[200.0, 720.0], [600.0, 400.0]], "types": "LL", "closed": False}],
                },
            ],
        }
    )

    # 2 daytime frames (must be filtered out)
    for idx in range(2):
        name = f"day_{idx:02d}.jpg"
        _make_image(name, is_night=False)
        frames.append(
            {
                "name": name,
                "attributes": {"weather": "clear", "scene": "city street", "timeofday": "daytime"},
                "labels": [
                    {
                        "id": 1000 + idx,
                        "category": "car",
                        "box2d": {"x1": 150.0, "y1": 300.0, "x2": 450.0, "y2": 520.0},
                    }
                ],
            }
        )

    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(frames, f, indent=2)

    return labels_path


def find_image_file(raw_dir: Path, image_name: str) -> Path:
    """Locate an image inside raw_dir (supports flat or nested images/ subdirectories)."""
    candidates = [
        raw_dir / "images" / image_name,
        raw_dir / image_name,
        raw_dir / "images" / "100k" / "train" / image_name,
        raw_dir / "images" / "100k" / "val" / image_name,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Could not locate image '{image_name}' under {raw_dir}")


def split_frames(
    frames: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically shuffle and split frames into train / val / test."""
    shuffled = list(frames)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def prepare_bdd_dataset(
    cfg: dict[str, Any],
    raw_dir: Path,
    labels_json: Path,
    out_dir: Path,
    seed: int,
    img_size: int,
) -> dict[str, Any]:
    """Filter night frames, split 70/20/10, resize to 640x640, and export YOLO + COCO."""
    target_timeofday: str = cfg["dataset"]["timeofday"]
    class_mapping: dict[str, int] = cfg["dataset"]["bdd_class_mapping"]
    classes_dict: dict[int, str] = {int(k): str(v) for k, v in cfg["dataset"]["classes"].items()}
    splits_cfg: dict[str, float] = cfg["dataset"]["splits"]

    images_raw_subdir = Path(cfg["paths"]["images_raw_subdir"])
    splits_subdir = Path(cfg["paths"]["splits_subdir"])
    yolo_subdir = Path(cfg["paths"]["yolo_subdir"])
    coco_subdir = Path(cfg["paths"]["coco_subdir"])

    with labels_json.open("r", encoding="utf-8") as f:
        all_frames: list[dict[str, Any]] = json.load(f)

    night_frames = [
        fr for fr in all_frames if (fr.get("attributes") or {}).get("timeofday") == target_timeofday
    ]
    filtered_out = len(all_frames) - len(night_frames)

    split_map = split_frames(
        night_frames,
        train_ratio=float(splits_cfg["train"]),
        val_ratio=float(splits_cfg["val"]),
        seed=seed,
    )

    coco_categories = [
        {"id": cls_id, "name": cls_name, "supercategory": "object"}
        for cls_id, cls_name in sorted(classes_dict.items())
    ]

    global_image_id = 1
    global_ann_id = 1
    stats: dict[str, Any] = {
        "total_input_frames": len(all_frames),
        "night_frames": len(night_frames),
        "filtered_out_frames": filtered_out,
        "splits": {},
        "negative_samples": 0,
        "class_counts": {cls_id: 0 for cls_id in sorted(classes_dict.keys())},
    }

    (out_dir / splits_subdir).mkdir(parents=True, exist_ok=True)
    (out_dir / coco_subdir).mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        split_items = split_map[split_name]
        split_img_dir = out_dir / images_raw_subdir / split_name
        split_yolo_dir = out_dir / yolo_subdir / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_yolo_dir.mkdir(parents=True, exist_ok=True)

        manifest_lines: list[str] = []
        coco_images: list[dict[str, Any]] = []
        coco_annotations: list[dict[str, Any]] = []

        for frame in split_items:
            img_name = frame["name"]
            src_img_path = find_image_file(raw_dir, img_name)
            img = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {src_img_path}")

            orig_h, orig_w = img.shape[:2]
            resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

            dst_img_path = split_img_dir / img_name
            cv2.imwrite(str(dst_img_path), resized)

            rel_manifest_path = (images_raw_subdir / split_name / img_name).as_posix()
            manifest_lines.append(rel_manifest_path)

            img_id = global_image_id
            global_image_id += 1
            coco_images.append(
                {
                    "id": img_id,
                    "file_name": img_name,
                    "width": img_size,
                    "height": img_size,
                }
            )

            sx = float(img_size) / float(orig_w)
            sy = float(img_size) / float(orig_h)

            yolo_lines: list[str] = []
            raw_labels = frame.get("labels") or []
            for obj in raw_labels:
                category = obj.get("category")
                if category not in class_mapping:
                    continue
                box2d = obj.get("box2d")
                if not isinstance(box2d, dict):
                    continue

                x1 = max(0.0, min(float(orig_w), float(box2d["x1"])))
                y1 = max(0.0, min(float(orig_h), float(box2d["y1"])))
                x2 = max(0.0, min(float(orig_w), float(box2d["x2"])))
                y2 = max(0.0, min(float(orig_h), float(box2d["y2"])))

                x1_s = max(0.0, min(float(img_size), x1 * sx))
                y1_s = max(0.0, min(float(img_size), y1 * sy))
                x2_s = max(0.0, min(float(img_size), x2 * sx))
                y2_s = max(0.0, min(float(img_size), y2 * sy))

                w_s = x2_s - x1_s
                h_s = y2_s - y1_s
                if w_s <= 0.0 or h_s <= 0.0:
                    continue

                cls_id = int(class_mapping[category])
                stats["class_counts"][cls_id] += 1

                # Compute exact-bounded normalized coordinates so cx +/- wn/2 stays inside [0, 1]
                x1_n = round(x1_s / img_size, 6)
                y1_n = round(y1_s / img_size, 6)
                x2_n = round(x2_s / img_size, 6)
                y2_n = round(y2_s / img_size, 6)
                wn = x2_n - x1_n
                hn = y2_n - y1_n
                if wn <= 0.0 or hn <= 0.0:
                    continue
                cx = (x1_n + x2_n) / 2.0
                cy = (y1_n + y2_n) / 2.0
                yolo_lines.append(f"{cls_id} {cx:.8f} {cy:.8f} {wn:.8f} {hn:.8f}")

                coco_annotations.append(
                    {
                        "id": global_ann_id,
                        "image_id": img_id,
                        "category_id": cls_id,
                        "bbox": [round(x1_s, 4), round(y1_s, 4), round(w_s, 4), round(h_s, 4)],
                        "area": round(w_s * h_s, 4),
                        "iscrowd": 0,
                    }
                )
                global_ann_id += 1

            if not yolo_lines:
                stats["negative_samples"] += 1

            yolo_txt_path = split_yolo_dir / f"{Path(img_name).stem}.txt"
            yolo_txt_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        manifest_path = out_dir / splits_subdir / f"{split_name}.txt"
        manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")

        coco_payload = {
            "info": {"description": f"EnPreNiNet BDD100K Night ({split_name})", "version": "2.1"},
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": coco_categories,
        }
        coco_path = out_dir / coco_subdir / f"instances_{split_name}.json"
        with coco_path.open("w", encoding="utf-8") as f:
            json.dump(coco_payload, f, indent=2)

        stats["splits"][split_name] = {
            "images": len(coco_images),
            "annotations": len(coco_annotations),
        }

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BDD100K night-vision dataset (YOLO + COCO).")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    parser.add_argument("--dummy", action="store_true", help="Run on isolated data/dummy -> data/dummy_processed")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Override raw BDD100K directory")
    parser.add_argument("--labels-json", type=Path, default=None, help="Override BDD100K labels JSON path")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--img-size", type=int, default=None, help="Override target image size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    cfg = load_config(config_path)

    seed = args.seed if args.seed is not None else int(cfg["project"]["seed"])
    img_size = args.img_size if args.img_size is not None else int(cfg["dataset"]["image_size"])

    if args.dummy:
        raw_dir = args.raw_dir or resolve_path(cfg["paths"]["data_dummy"])
        out_dir = args.out_dir or resolve_path(cfg["paths"]["data_dummy_processed"])
        labels_json = args.labels_json or (raw_dir / "labels_dummy.json")
        if not labels_json.is_file():
            generate_dummy_bdd_dataset(raw_dir, seed=seed)
    else:
        raw_dir = args.raw_dir or resolve_path(cfg["paths"]["data_raw"])
        out_dir = args.out_dir or resolve_path(cfg["paths"]["data_processed"])
        labels_json = args.labels_json or (raw_dir / "labels.json")
        if not labels_json.is_file():
            raise FileNotFoundError(f"BDD100K labels JSON not found: {labels_json}")

    stats = prepare_bdd_dataset(
        cfg=cfg,
        raw_dir=raw_dir,
        labels_json=labels_json,
        out_dir=out_dir,
        seed=seed,
        img_size=img_size,
    )
    print(f"[01_prepare_bdd] Completed ({'dummy' if args.dummy else 'full'}) -> {out_dir}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
