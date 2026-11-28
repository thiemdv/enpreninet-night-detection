"""F02: BDD100K Night Filter & Dual-Format Exporter (YOLO + COCO).

Filters BDD100K frames where timeofday == 'night', distinguishes unannotated
frames (labels is None, skipped by default) from valid negative samples
(labels == [] or 0 target objects, retained), maps target classes
{car: 0, person: 1, traffic light: 2}, deterministically sorts & splits 70/20/10,
cleans stale output directories before writing, resizes images to 640x640 into
images/raw/{train,val,test}/, and exports split manifests, dataset_meta.json
(with SHA-256 dataset_fingerprint hashing manifests + class mapping + annotations),
YOLO .txt labels, and COCO instances_*.json.

Usage:
    python src/prepare_bdd.py --config configs/config.yaml --dummy
    python src/prepare_bdd.py --config configs/config.yaml --raw-dir /content/raw_bdd/bdd100k
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
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


def _write_image_checked(dst_path: Path, img: np.ndarray) -> None:
    ok = cv2.imwrite(str(dst_path), img)
    if not ok:
        raise IOError(f"cv2.imwrite failed for {dst_path}")


def compute_dataset_fingerprint(
    manifest_by_split: dict[str, list[str]],
    seed: int,
    img_size: int,
    class_mapping: dict[str, int] | None = None,
    annotations_by_split: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Compute a deterministic SHA-256 fingerprint over manifests, class mapping, and annotations."""
    hasher = hashlib.sha256()
    hasher.update(f"seed={seed}|img_size={img_size}\n".encode("utf-8"))
    if class_mapping is not None:
        cm_str = json.dumps(sorted((str(k), int(v)) for k, v in class_mapping.items()))
        hasher.update(f"class_mapping={cm_str}\n".encode("utf-8"))
    for split_name in ("train", "val", "test"):
        hasher.update(f"[{split_name}]\n".encode("utf-8"))
        for entry in manifest_by_split.get(split_name, []):
            hasher.update(f"img:{entry}\n".encode("utf-8"))
        if annotations_by_split is not None:
            for ann in annotations_by_split.get(split_name, []):
                hasher.update(
                    f"ann:{ann['image_id']}:{ann['category_id']}:{ann['bbox']}\n".encode("utf-8")
                )
    return hasher.hexdigest()


def generate_dummy_bdd_dataset(dummy_dir: Path, seed: int = 42) -> Path:
    """Create a 13-frame BDD100K-schema dummy dataset under data/dummy/."""
    images_dir = dummy_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = dummy_dir / "labels_dummy.json"

    rng = np.random.default_rng(seed)
    width, height = 1280, 720
    frames: list[dict[str, Any]] = []

    def _make_image(name: str, is_night: bool) -> None:
        img_path = images_dir / name
        if is_night:
            img = rng.integers(5, 45, size=(height, width, 3), dtype=np.uint8)
            cv2.rectangle(img, (200, 320), (420, 480), (90, 95, 110), -1)
            cv2.circle(img, (900, 140), 18, (60, 190, 220), -1)
        else:
            img = rng.integers(120, 220, size=(height, width, 3), dtype=np.uint8)
        _write_image_checked(img_path, img)

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
            {
                "id": idx * 10 + 4,
                "category": "truck",
                "box2d": {"x1": 60.0, "y1": 220.0, "x2": 260.0, "y2": 460.0},
            },
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

    _make_image("night_08.jpg", is_night=True)
    frames.append(
        {
            "name": "night_08.jpg",
            "attributes": {"weather": "clear", "scene": "highway", "timeofday": "night"},
            "labels": [],
        }
    )

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

    _make_image("night_unannotated_10.jpg", is_night=True)
    frames.append(
        {
            "name": "night_unannotated_10.jpg",
            "attributes": {"weather": "clear", "scene": "highway", "timeofday": "night"},
            "labels": None,
        }
    )

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


def build_image_index(raw_dir: Path) -> dict[str, Path]:
    """Recursively index image filenames under raw_dir once for O(1) lookup (handles trainA/trainB/testA/testB)."""
    index: dict[str, Path] = {}
    search_roots = []
    for sub in (raw_dir / "images" / "100k", raw_dir / "bdd100k" / "images" / "100k"):
        if sub.is_dir():
            search_roots.append(sub)
    if not search_roots:
        search_roots = [raw_dir]

    valid_exts = {".jpg", ".jpeg", ".png"}
    for root_dir in search_roots:
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            dp = Path(dirpath)
            for fname in filenames:
                if Path(fname).suffix.lower() in valid_exts and fname not in index:
                    index[fname] = dp / fname
    return index


def find_image_file(raw_dir: Path, image_name: str, image_index: dict[str, Path] | None = None) -> Path:
    """Locate an image inside raw_dir using prebuilt recursive index or standard BDD100K paths."""
    if image_index is not None and image_name in image_index:
        return image_index[image_name]

    candidates = [
        raw_dir / "images" / image_name,
        raw_dir / image_name,
        raw_dir / "images" / "100k" / "train" / image_name,
        raw_dir / "images" / "100k" / "val" / image_name,
        raw_dir / "images" / "100k" / "test" / image_name,
        raw_dir / "bdd100k" / "images" / "100k" / "train" / image_name,
        raw_dir / "bdd100k" / "images" / "100k" / "val" / image_name,
        raw_dir / "bdd100k" / "images" / "100k" / "test" / image_name,
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
    """Validate unique frame names, sort deterministically, then shuffle & split train/val/test."""
    names = [str(fr["name"]) for fr in frames]
    if len(names) != len(set(names)):
        seen: set[str] = set()
        dups = sorted({nm for nm in names if nm in seen or seen.add(nm)})
        raise ValueError(f"Duplicate frame names detected in BDD labels: {dups[:5]}")

    ordered = sorted(frames, key=lambda fr: str(fr["name"]))
    random.Random(seed).shuffle(ordered)

    n = len(ordered)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


def _clean_output_subdirs(out_dir: Path, subdirs: tuple[Path, ...]) -> None:
    for sub in subdirs:
        target = out_dir / sub
        if target.exists():
            shutil.rmtree(target)


def prepare_bdd_dataset(
    cfg: dict[str, Any],
    raw_dir: Path,
    labels_json: Path,
    out_dir: Path,
    seed: int,
    img_size: int,
    include_null_labels: bool = False,
    max_images: int | None = None,
) -> dict[str, Any]:
    """Filter night frames, split 70/20/10, resize to 640x640, and export YOLO + COCO cleanly."""
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

    night_candidates = [
        fr for fr in all_frames if (fr.get("attributes") or {}).get("timeofday") == target_timeofday
    ]
    filtered_out_non_night = len(all_frames) - len(night_candidates)

    skipped_null_labels = 0
    valid_night_frames: list[dict[str, Any]] = []
    for fr in night_candidates:
        if fr.get("labels") is None and not include_null_labels:
            skipped_null_labels += 1
            continue
        valid_night_frames.append(fr)

    if max_images is not None and max_images > 0 and len(valid_night_frames) > max_images:
        valid_night_frames = sorted(valid_night_frames, key=lambda fr: str(fr["name"]))[:max_images]

    split_map = split_frames(
        valid_night_frames,
        train_ratio=float(splits_cfg["train"]),
        val_ratio=float(splits_cfg["val"]),
        seed=seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_output_subdirs(out_dir, (images_raw_subdir, splits_subdir, yolo_subdir, coco_subdir))

    coco_categories = [
        {"id": cls_id, "name": cls_name, "supercategory": "object"}
        for cls_id, cls_name in sorted(classes_dict.items())
    ]
    manifest_by_split: dict[str, list[str]] = {
        split_name: [(images_raw_subdir / split_name / fr["name"]).as_posix() for fr in split_map[split_name]]
        for split_name in ("train", "val", "test")
    }

    global_image_id = 1
    global_ann_id = 1
    coco_images_by_split: dict[str, list[dict[str, Any]]] = {}
    coco_anns_by_split: dict[str, list[dict[str, Any]]] = {}
    negative_samples = 0
    class_counts: dict[int, int] = {cls_id: 0 for cls_id in sorted(classes_dict.keys())}

    (out_dir / splits_subdir).mkdir(parents=True, exist_ok=True)
    (out_dir / coco_subdir).mkdir(parents=True, exist_ok=True)
    image_index = build_image_index(raw_dir)

    for split_name in ("train", "val", "test"):
        split_items = split_map[split_name]
        split_img_dir = out_dir / images_raw_subdir / split_name
        split_yolo_dir = out_dir / yolo_subdir / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_yolo_dir.mkdir(parents=True, exist_ok=True)

        coco_images: list[dict[str, Any]] = []
        coco_annotations: list[dict[str, Any]] = []

        for frame in split_items:
            img_name = frame["name"]
            src_img_path = find_image_file(raw_dir, img_name, image_index=image_index)
            img = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {src_img_path}")

            orig_h, orig_w = img.shape[:2]
            resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            _write_image_checked(split_img_dir / img_name, resized)

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
                class_counts[cls_id] += 1

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
                negative_samples += 1

            yolo_txt_path = split_yolo_dir / f"{Path(img_name).stem}.txt"
            yolo_txt_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        manifest_lines = manifest_by_split[split_name]
        manifest_path = out_dir / splits_subdir / f"{split_name}.txt"
        manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")

        coco_images_by_split[split_name] = coco_images
        coco_anns_by_split[split_name] = coco_annotations

    # A03 fix: include class_mapping and all annotation contents in dataset_fingerprint
    dataset_fingerprint = compute_dataset_fingerprint(
        manifest_by_split=manifest_by_split,
        seed=seed,
        img_size=img_size,
        class_mapping=class_mapping,
        annotations_by_split=coco_anns_by_split,
    )

    stats: dict[str, Any] = {
        "total_input_frames": len(all_frames),
        "night_frames": len(valid_night_frames),
        "filtered_out_frames": filtered_out_non_night,
        "skipped_null_labels": skipped_null_labels,
        "include_null_labels": include_null_labels,
        "seed": seed,
        "img_size": img_size,
        "dataset_fingerprint": dataset_fingerprint,
        "splits": {},
        "negative_samples": negative_samples,
        "class_counts": class_counts,
    }

    for split_name in ("train", "val", "test"):
        coco_payload = {
            "info": {
                "description": f"EnPreNiNet BDD100K Night ({split_name})",
                "version": "2.1",
                "dataset_fingerprint": dataset_fingerprint,
                "seed": seed,
            },
            "images": coco_images_by_split[split_name],
            "annotations": coco_anns_by_split[split_name],
            "categories": coco_categories,
        }
        coco_path = out_dir / coco_subdir / f"instances_{split_name}.json"
        with coco_path.open("w", encoding="utf-8") as f:
            json.dump(coco_payload, f, indent=2)
        stats["splits"][split_name] = {
            "images": len(coco_images_by_split[split_name]),
            "annotations": len(coco_anns_by_split[split_name]),
        }

    (out_dir / "dataset_meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
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
    parser.add_argument(
        "--include-null-labels",
        action="store_true",
        default=False,
        help="Include frames where labels is None (default False: skip unannotated frames)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on valid night frames (useful for GPU pilot runs)",
    )
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
        include_null_labels=args.include_null_labels,
        max_images=args.max_images,
    )
    print(f"[prepare_bdd] Completed ({'dummy' if args.dummy else 'full'}) -> {out_dir}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
