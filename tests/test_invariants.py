"""Deterministic structural, config, and pipeline I/O invariant tests for EnPreNiNet (Harness v2.1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

VALID_GHE_METHODS = {"fig4_literal", "fig4_normalized", "standard_local_he"}
VALID_BORDER_MODES = {"reflect", "replicate", "constant"}
VALID_ZERO_DIV_POLICIES = {"identity", "zero"}
VALID_UNIFORM_WINDOW_POLICIES = {"identity", "stretch"}


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# F01: Project Scaffolding & Config Invariants
# ---------------------------------------------------------------------------


def test_config_exists() -> None:
    """Verify configs/config.yaml exists, is valid YAML, and has required sections."""
    assert CONFIG_PATH.is_file(), f"Missing config file: {CONFIG_PATH}"
    cfg = load_config()
    assert isinstance(cfg, dict)
    required_sections = {
        "project",
        "paths",
        "dataset",
        "preprocessing",
        "models",
        "fusion",
        "evaluation",
    }
    missing = required_sections - set(cfg.keys())
    assert not missing, f"Missing config sections: {missing}"


def test_config_no_hardcoded_paths() -> None:
    """Verify all configured paths and weights are relative (cross-platform)."""
    cfg = load_config()
    path_values = list(cfg["paths"].values()) + [
        cfg["models"]["yolov5"]["weights"],
        cfg["models"]["faster_rcnn"]["weights"],
    ]
    for raw_path in path_values:
        assert isinstance(raw_path, str) and raw_path.strip(), f"Empty path in config: {raw_path!r}"
        p = Path(raw_path)
        assert not p.is_absolute(), f"Hardcoded absolute path not allowed: {raw_path}"
        assert ":" not in raw_path, f"Drive-letter path not allowed: {raw_path}"


def test_config_dataset_invariants() -> None:
    """Verify dataset parameters match EnPreNiNet paper & project specs."""
    ds = load_config()["dataset"]
    assert ds["timeofday"] == "night"
    assert ds["image_size"] == 640
    assert ds["channels"] == 3
    assert ds["classes"] == {0: "car", 1: "person", 2: "traffic light"}

    splits = ds["splits"]
    assert pytest.approx(splits["train"], abs=1e-6) == 0.70
    assert pytest.approx(splits["val"], abs=1e-6) == 0.20
    assert pytest.approx(splits["test"], abs=1e-6) == 0.10
    assert pytest.approx(sum(splits.values()), abs=1e-6) == 1.0


def test_config_ghe_invariants() -> None:
    """Verify GHE Ensemble preprocessing parameters and v2.1 mode/edge-case policies."""
    ghe = load_config()["preprocessing"]["ghe_ensemble"]
    assert ghe["image_size"] == 640
    assert ghe["gaussian_kernel_size"] == 3
    assert pytest.approx(ghe["gaussian_sigma"], abs=1e-6) == 2.0
    assert isinstance(ghe["window_size"], int)
    assert ghe["window_size"] >= 3 and ghe["window_size"] % 2 == 1
    assert ghe["output_channels"] == 3
    assert ghe["method"] in VALID_GHE_METHODS, f"Invalid GHE method: {ghe.get('method')}"
    assert ghe["border_mode"] in VALID_BORDER_MODES, f"Invalid border_mode: {ghe.get('border_mode')}"
    assert ghe["zero_division_policy"] in VALID_ZERO_DIV_POLICIES, (
        f"Invalid zero_division_policy: {ghe.get('zero_division_policy')}"
    )
    assert ghe["uniform_window_policy"] in VALID_UNIFORM_WINDOW_POLICIES, (
        f"Invalid uniform_window_policy: {ghe.get('uniform_window_policy')}"
    )


def test_config_models_and_fusion_invariants() -> None:
    """Verify detector architectures, score-level fusion toggles, and evaluation thresholds."""
    cfg = load_config()
    yolo = cfg["models"]["yolov5"]
    frcnn = cfg["models"]["faster_rcnn"]
    fusion = cfg["fusion"]
    evaluation = cfg["evaluation"]

    assert yolo["variant"] == "yolov5l"
    assert yolo["img_size"] == 640

    assert frcnn["backbone"] == "resnet152"
    assert frcnn["num_classes"] == 4  # 3 target classes (1..3) + 1 background (0)
    assert frcnn["img_size"] == 640

    assert fusion["confidence_thresholds"] == [0.25, 0.50, 0.75]
    assert 0.0 < fusion["intra_nms_iou"] <= 1.0
    assert 0.0 < fusion["global_nms_iou"] <= 1.0
    assert isinstance(fusion["global_nms_enabled"], bool)

    assert evaluation["iou_thresholds"] == [0.50, 0.75]


def test_requirements_exists() -> None:
    """Verify requirements.txt exists and declares core dependencies."""
    assert REQUIREMENTS_PATH.is_file(), f"Missing requirements.txt: {REQUIREMENTS_PATH}"
    content = REQUIREMENTS_PATH.read_text(encoding="utf-8").lower()
    for pkg in ("numpy", "opencv-python", "scikit-image", "pyyaml", "torch", "torchvision", "pytest"):
        assert pkg in content, f"Missing {pkg} in requirements.txt"


def test_directory_structure() -> None:
    """Verify core repository directories exist."""
    for rel in ("configs", "docs", "src", "tests", "notebooks"):
        assert (REPO_ROOT / rel).is_dir(), f"Missing directory: {rel}"


# ---------------------------------------------------------------------------
# Pipeline Mathematical, Label Mapping & Tensor-Shape Contract Tests
# ---------------------------------------------------------------------------


def test_pipeline_gaussian_and_channel_stacking_contract() -> None:
    """Verify normalized 3x3 Gaussian kernel (sigma=2) and 3-channel stacking contract."""
    ghe = load_config()["preprocessing"]["ghe_ensemble"]
    k = ghe["gaussian_kernel_size"]
    sigma = ghe["gaussian_sigma"]
    half = k // 2

    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()

    assert kernel.shape == (3, 3)
    assert np.isclose(kernel.sum(), 1.0, atol=1e-7)
    assert np.allclose(kernel, kernel.T)

    gray = np.zeros((ghe["image_size"], ghe["image_size"]), dtype=np.uint8)
    stacked = np.stack([gray, gray, gray], axis=-1)
    assert stacked.shape == (640, 640, 3)
    assert stacked.dtype == np.uint8
    assert np.array_equal(stacked[..., 0], stacked[..., 1])
    assert np.array_equal(stacked[..., 0], stacked[..., 2])


def test_pipeline_fusion_threshold_contract() -> None:
    """Verify confidence thresholding across [0.25, 0.50, 0.75] behaves monotonically."""
    thresholds = load_config()["fusion"]["confidence_thresholds"]
    scores = np.array([0.10, 0.30, 0.60, 0.80], dtype=np.float32)
    counts = [int((scores >= tau).sum()) for tau in thresholds]
    assert counts == [3, 2, 1]


def test_pipeline_unified_prediction_schema_and_label_mapping() -> None:
    """Verify image_id-keyed unified prediction schema, FRCNN 1..3->0..2 mapping, and empty (0, 4) support."""
    import torch

    raw_frcnn = {
        "boxes": torch.tensor([[10.0, 20.0, 100.0, 120.0], [5.0, 5.0, 50.0, 50.0], [200.0, 100.0, 300.0, 250.0]]),
        "scores": torch.tensor([0.90, 0.80, 0.65]),
        "labels": torch.tensor([1, 0, 3], dtype=torch.long),  # 0=bg, 1->0 (car), 3->2 (traffic light)
    }
    fg_mask = raw_frcnn["labels"] > 0
    preds_by_image = {
        1: {
            "boxes": raw_frcnn["boxes"][fg_mask].to(torch.float32),
            "scores": raw_frcnn["scores"][fg_mask].to(torch.float32),
            "labels": (raw_frcnn["labels"][fg_mask] - 1).to(torch.long),
        },
        2: {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.long),
        },
    }

    assert set(preds_by_image[1].keys()) == {"boxes", "scores", "labels"}
    assert preds_by_image[1]["boxes"].shape == (2, 4)
    assert preds_by_image[1]["scores"].shape == (2,)
    assert preds_by_image[1]["labels"].tolist() == [0, 2]
    assert preds_by_image[1]["labels"].dtype == torch.long

    assert preds_by_image[2]["boxes"].shape == (0, 4)
    assert preds_by_image[2]["scores"].shape == (0,)
    assert preds_by_image[2]["labels"].shape == (0,)


# ---------------------------------------------------------------------------
# F02–F08 Integration Invariants (auto-activated once modules exist)
# ---------------------------------------------------------------------------


def test_labels_format_and_bounds() -> None:
    """F02: Verify isolated dummy outputs, split disjointness, YOLO bounds, negative samples, and YOLO<->COCO round-trip."""
    script_path = REPO_ROOT / "src" / "01_prepare_bdd.py"
    if not script_path.is_file():
        pytest.skip("Waiting for F02 (src/01_prepare_bdd.py)")

    import cv2

    cfg = load_config()
    img_size = int(cfg["dataset"]["image_size"])
    valid_classes = set(int(k) for k in cfg["dataset"]["classes"].keys())
    out_root = REPO_ROOT / cfg["paths"]["data_dummy_processed"]

    images_raw_dir = out_root / cfg["paths"]["images_raw_subdir"]
    splits_dir = out_root / cfg["paths"]["splits_subdir"]
    yolo_dir = out_root / cfg["paths"]["yolo_subdir"]
    coco_dir = out_root / cfg["paths"]["coco_subdir"]

    expected_dummy_counts = cfg["dataset"]["dummy_samples"]
    seen_image_ids: set[int] = set()
    seen_file_names: set[str] = set()
    negative_sample_count = 0

    for split_name in ("train", "val", "test"):
        manifest_path = splits_dir / f"{split_name}.txt"
        coco_path = coco_dir / f"instances_{split_name}.json"
        assert manifest_path.is_file(), f"Missing split manifest: {manifest_path}"
        assert coco_path.is_file(), f"Missing COCO JSON: {coco_path}"

        manifest_entries = [ln.strip() for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(manifest_entries) == expected_dummy_counts[split_name], (
            f"Expected {expected_dummy_counts[split_name]} images in {split_name}, got {len(manifest_entries)}"
        )

        coco_data = json.loads(coco_path.read_text(encoding="utf-8"))
        assert {"images", "annotations", "categories"}.issubset(coco_data.keys())
        assert {c["id"] for c in coco_data["categories"]} == valid_classes
        assert len(coco_data["images"]) == expected_dummy_counts[split_name]

        split_image_id_to_name: dict[int, str] = {}
        anns_by_image_id: dict[int, list[dict]] = {}

        for img_entry in coco_data["images"]:
            img_id = int(img_entry["id"])
            file_name = str(img_entry["file_name"])
            assert img_id not in seen_image_ids, f"Duplicate image_id {img_id} across splits"
            assert file_name not in seen_file_names, f"Duplicate file_name {file_name} across splits"
            assert "day" not in file_name.lower(), f"Daytime image leaked into night split: {file_name}"
            assert img_entry["width"] == img_size and img_entry["height"] == img_size

            seen_image_ids.add(img_id)
            seen_file_names.add(file_name)
            split_image_id_to_name[img_id] = file_name
            anns_by_image_id[img_id] = []

            resized_img_path = images_raw_dir / split_name / file_name
            assert resized_img_path.is_file(), f"Missing resized raw image: {resized_img_path}"
            img = cv2.imread(str(resized_img_path), cv2.IMREAD_COLOR)
            assert img is not None and img.shape == (img_size, img_size, 3)

        for ann in coco_data["annotations"]:
            ann_img_id = int(ann["image_id"])
            assert ann_img_id in split_image_id_to_name, f"Orphan annotation image_id={ann_img_id} in {coco_path}"
            assert int(ann["category_id"]) in valid_classes
            x, y, w, h = (float(v) for v in ann["bbox"])
            assert 0.0 <= x <= img_size and 0.0 <= y <= img_size
            assert w > 0.0 and h > 0.0
            assert x + w <= img_size + 1e-3 and y + h <= img_size + 1e-3
            anns_by_image_id[ann_img_id].append(ann)

        # Verify YOLO label bounds and exact YOLO <-> COCO coordinate round-trip
        for img_id, file_name in split_image_id_to_name.items():
            yolo_txt = yolo_dir / split_name / f"{Path(file_name).stem}.txt"
            assert yolo_txt.is_file(), f"Missing YOLO label file: {yolo_txt}"

            yolo_boxes: list[tuple[int, float, float, float, float]] = []
            for line in yolo_txt.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                assert len(parts) == 5, f"Invalid YOLO line in {yolo_txt}: {line}"
                cls_id = int(parts[0])
                cx, cy, wn, hn = (float(v) for v in parts[1:])
                assert cls_id in valid_classes
                assert 0.0 < wn <= 1.0 and 0.0 < hn <= 1.0
                assert cx - wn / 2.0 >= 0.0, f"YOLO box left edge < 0 in {yolo_txt}: {line}"
                assert cx + wn / 2.0 <= 1.0, f"YOLO box right edge > 1 in {yolo_txt}: {line}"
                assert cy - hn / 2.0 >= 0.0, f"YOLO box top edge < 0 in {yolo_txt}: {line}"
                assert cy + hn / 2.0 <= 1.0, f"YOLO box bottom edge > 1 in {yolo_txt}: {line}"
                yolo_boxes.append((cls_id, cx, cy, wn, hn))

            coco_anns = anns_by_image_id[img_id]
            assert len(yolo_boxes) == len(coco_anns), (
                f"YOLO ({len(yolo_boxes)}) vs COCO ({len(coco_anns)}) box count mismatch for {file_name}"
            )
            if len(yolo_boxes) == 0:
                negative_sample_count += 1

            for (y_cls, y_cx, y_cy, y_w, y_h), c_ann in zip(yolo_boxes, coco_anns):
                assert y_cls == int(c_ann["category_id"])
                cx_px, cy_px, w_px, h_px = (float(v) for v in c_ann["bbox"])
                expected_cx = (cx_px + w_px / 2.0) / img_size
                expected_cy = (cy_px + h_px / 2.0) / img_size
                expected_w = w_px / img_size
                expected_h = h_px / img_size
                assert abs(y_cx - expected_cx) < 1e-4
                assert abs(y_cy - expected_cy) < 1e-4
                assert abs(y_w - expected_w) < 1e-4
                assert abs(y_h - expected_h) < 1e-4

    assert len(seen_image_ids) == sum(expected_dummy_counts.values())
    assert negative_sample_count >= 1, "Expected at least 1 night negative sample (0 target objects) to be retained"


def test_ghe_output_shape_and_channels() -> None:
    """F03: Verify on-disk lossless PNG GHE outputs, comparison artifact, and all 3 GHE formulas on edge-case inputs."""
    script_path = REPO_ROOT / "src" / "02_ghe_ensemble.py"
    if not script_path.is_file():
        pytest.skip("Waiting for F03 (src/02_ghe_ensemble.py)")

    import importlib.util

    import cv2

    # 1. Verify on-disk dummy outputs and visual comparison artifact
    cfg = load_config()
    ghe_cfg = cfg["preprocessing"]["ghe_ensemble"]
    method = ghe_cfg["method"]
    expected_size = int(ghe_cfg["image_size"])
    expected_total = sum(cfg["dataset"]["dummy_samples"].values())

    ghe_dir = REPO_ROOT / cfg["paths"]["data_dummy_processed"] / cfg["paths"]["ghe_images_root"] / method
    png_files = sorted(ghe_dir.rglob("*.png"))
    assert len(png_files) == expected_total, (
        f"Expected {expected_total} lossless PNG GHE outputs under {ghe_dir}, found {len(png_files)}"
    )

    for img_path in png_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        assert img is not None, f"Failed to read {img_path}"
        assert img.dtype == np.uint8, f"Expected uint8 for {img_path}, got {img.dtype}"
        assert img.shape == (expected_size, expected_size, 3), f"Unexpected shape {img.shape} for {img_path}"
        assert np.array_equal(img[..., 0], img[..., 1])
        assert np.array_equal(img[..., 0], img[..., 2])

    artifact_path = REPO_ROOT / cfg["paths"]["artifacts_dir"] / "ghe_comparison_dummy.png"
    assert artifact_path.is_file() and artifact_path.stat().st_size > 0, (
        f"Missing or empty GHE comparison artifact: {artifact_path}"
    )

    # 2. Directly import ghe_ensemble and verify all 3 modes on synthetic edge-case images
    spec = importlib.util.spec_from_file_location("ghe_ensemble_mod", script_path)
    assert spec is not None and spec.loader is not None
    ghe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ghe_mod)

    # 2a. All-black image (H == 0): must not divide by zero, must return all 0s in all 3 modes
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    for mode in ("fig4_literal", "fig4_normalized", "standard_local_he"):
        out = ghe_mod.ghe_ensemble(black, image_size=None, window_size=5, method=mode)
        assert out.shape == (32, 32, 3) and out.dtype == np.uint8
        assert np.all(out == 0), f"All-black image failed for mode={mode}"
        assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 0], out[..., 2])

    # 2b. Uniform non-zero image (v = 30)
    uniform = np.full((32, 32, 3), 30, dtype=np.uint8)
    # fig4_literal: W=5 -> W^2=25 -> round(30 * (25/30)) = 25; W=17 -> W^2=289 -> clipped to 255
    out_lit_w5 = ghe_mod.ghe_ensemble(uniform, image_size=None, window_size=5, method="fig4_literal")
    assert np.all(out_lit_w5 == 25), f"Expected 25 for fig4_literal W=5 uniform, got unique {np.unique(out_lit_w5)}"
    out_lit_w17 = ghe_mod.ghe_ensemble(uniform, image_size=None, window_size=17, method="fig4_literal")
    assert np.all(out_lit_w17 == 255), f"Expected 255 (clipped) for fig4_literal W=17 uniform, got {np.unique(out_lit_w17)}"

    # fig4_normalized: C(v)=1.0 -> round(H * 1.0) = 30 (identity)
    out_norm = ghe_mod.ghe_ensemble(uniform, image_size=None, window_size=5, method="fig4_normalized")
    assert np.all(out_norm == 30), f"Expected 30 for fig4_normalized uniform, got {np.unique(out_norm)}"

    # standard_local_he: identity policy preserves 30; stretch policy maps to 255
    out_std_id = ghe_mod.ghe_ensemble(
        uniform, image_size=None, window_size=5, method="standard_local_he", uniform_window_policy="identity"
    )
    assert np.all(out_std_id == 30), f"Expected 30 for standard_local_he identity, got {np.unique(out_std_id)}"
    out_std_st = ghe_mod.ghe_ensemble(
        uniform, image_size=None, window_size=5, method="standard_local_he", uniform_window_policy="stretch"
    )
    assert np.all(out_std_st == 255), f"Expected 255 for standard_local_he stretch, got {np.unique(out_std_st)}"

    # 2c. Synthetic low-light gradient image (0..100)
    ramp = np.tile(np.linspace(0, 100, 64, dtype=np.uint8), (64, 1))
    grad_bgr = np.stack([ramp, ramp, ramp], axis=-1)
    out_grad_lit = ghe_mod.ghe_ensemble(grad_bgr, image_size=None, window_size=7, method="fig4_literal")
    out_grad_norm = ghe_mod.ghe_ensemble(grad_bgr, image_size=None, window_size=7, method="fig4_normalized")
    out_grad_std = ghe_mod.ghe_ensemble(grad_bgr, image_size=None, window_size=7, method="standard_local_he")

    for arr in (out_grad_lit, out_grad_norm, out_grad_std):
        assert arr.shape == (64, 64, 3) and arr.dtype == np.uint8
        assert np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2])
    # fig4_normalized stays bounded by local max H <= 100, while standard_local_he stretches toward 255
    assert int(out_grad_norm.max()) <= 100
    assert int(out_grad_std.max()) >= 200
    # fig4_literal max is bounded by W^2 = 49
    assert int(out_grad_lit.max()) <= 49


def test_yolo_output() -> None:
    """F04: Verify YOLOv5l wrapper unified predictions (.pt), schema, bounds, and 0-detection handling."""
    script_path = REPO_ROOT / "src" / "03_train_infer_yolo.py"
    if not script_path.is_file():
        pytest.skip("Waiting for F04 (src/03_train_infer_yolo.py)")

    import importlib.util

    import torch

    cfg = load_config()
    variant = str(cfg["preprocessing"]["ghe_ensemble"]["method"])
    img_size = int(cfg["models"]["yolov5"]["img_size"])
    valid_classes = set(int(k) for k in cfg["dataset"]["classes"].keys())

    run_dir = REPO_ROOT / cfg["paths"]["runs_dir"] / f"dummy_yolo_{variant}"
    preds_path = run_dir / f"yolo_{variant}_test_preds.pt"
    assert preds_path.is_file(), f"Missing YOLOv5 unified prediction file: {preds_path}"

    coco_test_path = (
        REPO_ROOT / cfg["paths"]["data_dummy_processed"] / cfg["paths"]["coco_subdir"] / "instances_test.json"
    )
    coco_test = json.loads(coco_test_path.read_text(encoding="utf-8"))
    expected_image_ids = {int(img["id"]) for img in coco_test["images"]}

    preds = torch.load(preds_path, weights_only=True)
    assert isinstance(preds, dict)
    assert set(preds.keys()) == expected_image_ids, (
        f"Prediction image_ids {set(preds.keys())} != test split image_ids {expected_image_ids}"
    )

    for img_id, det in preds.items():
        assert isinstance(det, dict)
        assert set(det.keys()) == {"boxes", "scores", "labels"}
        boxes = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]

        assert isinstance(boxes, torch.Tensor) and boxes.dtype == torch.float32
        assert isinstance(scores, torch.Tensor) and scores.dtype == torch.float32
        assert isinstance(labels, torch.Tensor) and labels.dtype == torch.int64
        assert boxes.ndim == 2 and boxes.shape[1] == 4
        n = boxes.shape[0]
        assert scores.shape == (n,)
        assert labels.shape == (n,)

        if n > 0:
            assert bool((boxes >= 0.0).all() and (boxes <= float(img_size)).all()), (
                f"Boxes out of [0, {img_size}] for image_id={img_id}"
            )
            assert bool((boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all())
            assert bool((scores >= 0.0).all() and (scores <= 1.0).all())
            assert set(labels.tolist()).issubset(valid_classes)

    # Verify standardize_yolo_predictions handles 0 detections cleanly -> empty (0, 4) tensors
    spec = importlib.util.spec_from_file_location("yolo_wrapper_mod", script_path)
    assert spec is not None and spec.loader is not None
    yolo_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(yolo_mod)

    for empty_in in (None, torch.empty((0, 6), dtype=torch.float32)):
        empty_out = yolo_mod.standardize_yolo_predictions(empty_in, img_size=img_size, score_floor=0.001)
        assert empty_out["boxes"].shape == (0, 4) and empty_out["boxes"].dtype == torch.float32
        assert empty_out["scores"].shape == (0,) and empty_out["scores"].dtype == torch.float32
        assert empty_out["labels"].shape == (0,) and empty_out["labels"].dtype == torch.int64

    below_floor = torch.tensor([[10.0, 20.0, 50.0, 60.0, 0.0001, 0.0]], dtype=torch.float32)
    filtered_out = yolo_mod.standardize_yolo_predictions(below_floor, img_size=img_size, score_floor=0.001)
    assert filtered_out["boxes"].shape == (0, 4)
    assert filtered_out["scores"].shape == (0,)
    assert filtered_out["labels"].shape == (0,)


def test_frcnn_output() -> None:
    """F05: Verify Faster R-CNN ResNet-152 unified predictions (.pt), target encoding (0..2->1..3), and decoding (1..3->0..2)."""
    script_path = REPO_ROOT / "src" / "04_train_infer_frcnn.py"
    if not script_path.is_file():
        pytest.skip("Waiting for F05 (src/04_train_infer_frcnn.py)")

    import importlib.util

    import torch

    cfg = load_config()
    variant = str(cfg["preprocessing"]["ghe_ensemble"]["method"])
    img_size = int(cfg["models"]["faster_rcnn"]["img_size"])
    valid_classes = set(int(k) for k in cfg["dataset"]["classes"].keys())

    run_dir = REPO_ROOT / cfg["paths"]["runs_dir"] / f"dummy_frcnn_{variant}"
    preds_path = run_dir / f"frcnn_{variant}_test_preds.pt"
    assert preds_path.is_file(), f"Missing Faster R-CNN unified prediction file: {preds_path}"

    coco_test_path = (
        REPO_ROOT / cfg["paths"]["data_dummy_processed"] / cfg["paths"]["coco_subdir"] / "instances_test.json"
    )
    coco_test = json.loads(coco_test_path.read_text(encoding="utf-8"))
    expected_image_ids = {int(img["id"]) for img in coco_test["images"]}

    preds = torch.load(preds_path, weights_only=True)
    assert isinstance(preds, dict)
    assert set(preds.keys()) == expected_image_ids, (
        f"Prediction image_ids {set(preds.keys())} != test split image_ids {expected_image_ids}"
    )

    for img_id, det in preds.items():
        assert isinstance(det, dict)
        assert set(det.keys()) == {"boxes", "scores", "labels"}
        boxes = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]

        assert isinstance(boxes, torch.Tensor) and boxes.dtype == torch.float32
        assert isinstance(scores, torch.Tensor) and scores.dtype == torch.float32
        assert isinstance(labels, torch.Tensor) and labels.dtype == torch.int64
        assert boxes.ndim == 2 and boxes.shape[1] == 4
        n = boxes.shape[0]
        assert scores.shape == (n,)
        assert labels.shape == (n,)

        if n > 0:
            assert bool((boxes >= 0.0).all() and (boxes <= float(img_size)).all()), (
                f"Boxes out of [0, {img_size}] for image_id={img_id}"
            )
            assert bool((boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all())
            assert bool((scores >= 0.0).all() and (scores <= 1.0).all())
            assert set(labels.tolist()).issubset(valid_classes)

    # Directly test encode_frcnn_targets() and decode_frcnn_predictions()
    spec = importlib.util.spec_from_file_location("frcnn_mod", script_path)
    assert spec is not None and spec.loader is not None
    frcnn_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frcnn_mod)

    # 1. Target encoding: 0..2 -> 1..3 and xywh -> xyxy
    sample_anns = [
        {"category_id": 0, "bbox": [10.0, 20.0, 100.0, 80.0], "iscrowd": 0},
        {"category_id": 1, "bbox": [50.0, 60.0, 40.0, 90.0], "iscrowd": 0},
        {"category_id": 2, "bbox": [300.0, 10.0, 30.0, 50.0], "iscrowd": 0},
    ]
    enc = frcnn_mod.encode_frcnn_targets(sample_anns, image_id=7, img_size=img_size)
    assert enc["boxes"].shape == (3, 4) and enc["boxes"].dtype == torch.float32
    assert enc["labels"].tolist() == [1, 2, 3] and enc["labels"].dtype == torch.int64
    assert enc["boxes"][0].tolist() == [10.0, 20.0, 110.0, 100.0]

    # Negative sample (0 annotations) -> empty (0, 4)
    enc_empty = frcnn_mod.encode_frcnn_targets([], image_id=8, img_size=img_size)
    assert enc_empty["boxes"].shape == (0, 4) and enc_empty["boxes"].dtype == torch.float32
    assert enc_empty["labels"].shape == (0,) and enc_empty["labels"].dtype == torch.int64

    # 2. Prediction decoding: drop background (0), map 1..3 -> 0..2, filter score_floor, clamp [0, 640], sort desc
    raw_out = {
        "boxes": torch.tensor(
            [
                [-5.0, 10.0, 100.0, 120.0],   # label 1 -> 0, score 0.40
                [20.0, 20.0, 80.0, 80.0],     # label 0 -> background (drop!)
                [200.0, 100.0, 700.0, 300.0], # label 3 -> 2, score 0.85
                [30.0, 30.0, 60.0, 60.0],     # label 2 -> 1, score 0.0001 (below floor -> drop!)
            ],
            dtype=torch.float32,
        ),
        "scores": torch.tensor([0.40, 0.99, 0.85, 0.0001], dtype=torch.float32),
        "labels": torch.tensor([1, 0, 3, 2], dtype=torch.int64),
    }
    dec = frcnn_mod.decode_frcnn_predictions(raw_out, img_size=img_size, score_floor=0.001)
    assert dec["boxes"].shape == (2, 4)
    assert dec["scores"].tolist() == pytest.approx([0.85, 0.40], abs=1e-6)
    assert dec["labels"].tolist() == [2, 0]
    assert dec["boxes"][0, 2].item() == float(img_size)  # clamped from 700 -> 640
    assert dec["boxes"][1, 0].item() == 0.0              # clamped from -5 -> 0

    # 0-detection / all-background decoding -> empty (0, 4)
    bg_only = {
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32),
        "scores": torch.tensor([0.95], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.int64),
    }
    for empty_in in (None, bg_only):
        dec_empty = frcnn_mod.decode_frcnn_predictions(empty_in, img_size=img_size, score_floor=0.001)
        assert dec_empty["boxes"].shape == (0, 4) and dec_empty["boxes"].dtype == torch.float32
        assert dec_empty["scores"].shape == (0,) and dec_empty["scores"].dtype == torch.float32
        assert dec_empty["labels"].shape == (0,) and dec_empty["labels"].dtype == torch.int64


def test_fusion_output() -> None:
    """F06: Verify on-disk fused predictions (.pt) and all 4 fusion behavioral scenarios via fuse_image_predictions()."""
    script_path = REPO_ROOT / "src" / "05_fusion.py"
    if not script_path.is_file():
        pytest.skip("Waiting for F06 (src/05_fusion.py)")

    import importlib.util

    import torch

    cfg = load_config()
    variant = str(cfg["preprocessing"]["ghe_ensemble"]["method"])
    img_size = int(cfg["dataset"]["image_size"])
    valid_classes = set(int(k) for k in cfg["dataset"]["classes"].keys())
    thresholds = [float(t) for t in cfg["fusion"]["confidence_thresholds"]]

    preds_path = (
        REPO_ROOT / cfg["paths"]["runs_dir"] / f"dummy_fusion_{variant}" / f"fusion_{variant}_test_preds.pt"
    )
    assert preds_path.is_file(), f"Missing fused prediction file: {preds_path}"

    coco_test_path = (
        REPO_ROOT / cfg["paths"]["data_dummy_processed"] / cfg["paths"]["coco_subdir"] / "instances_test.json"
    )
    coco_test = json.loads(coco_test_path.read_text(encoding="utf-8"))
    expected_image_ids = {int(img["id"]) for img in coco_test["images"]}

    preds = torch.load(preds_path, weights_only=True)
    assert isinstance(preds, dict)
    assert set(preds.keys()) == expected_image_ids

    for img_id, det in preds.items():
        assert set(det.keys()) == {"boxes", "scores", "labels"}
        boxes, scores, labels = det["boxes"], det["scores"], det["labels"]
        assert boxes.dtype == torch.float32 and scores.dtype == torch.float32 and labels.dtype == torch.int64
        assert boxes.ndim == 2 and boxes.shape[1] == 4
        n = boxes.shape[0]
        assert scores.shape == (n,) and labels.shape == (n,)
        if n > 0:
            assert bool((boxes >= 0.0).all() and (boxes <= float(img_size)).all())
            assert bool((boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all())
            assert bool((scores >= 0.0).all() and (scores <= 1.0).all())
            assert set(labels.tolist()).issubset(valid_classes)
            # Scores must be sorted in descending order
            if n > 1:
                assert bool((scores[:-1] >= scores[1:]).all()), f"Scores not sorted desc for image_id={img_id}"

    # Directly import fuse_image_predictions() and test 4 behavioral scenarios
    spec = importlib.util.spec_from_file_location("fusion_mod", script_path)
    assert spec is not None and spec.loader is not None
    fusion_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fusion_mod)

    # Scenario 1: 2 overlapping boxes (IoU > 0.5) of the SAME class (0.90 vs 0.70)
    yolo_s1 = {
        "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32),
        "scores": torch.tensor([0.90], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.int64),
    }
    frcnn_s1 = {
        "boxes": torch.tensor([[105.0, 105.0, 205.0, 205.0]], dtype=torch.float32),
        "scores": torch.tensor([0.70], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.int64),
    }
    out_nms_on = fusion_mod.fuse_image_predictions(
        yolo_s1, frcnn_s1, conf_thres=0.001, global_nms_iou=0.50, global_nms_enabled=True
    )
    assert out_nms_on["boxes"].shape == (1, 4)
    assert out_nms_on["scores"].tolist() == pytest.approx([0.90], abs=1e-6)
    assert out_nms_on["labels"].tolist() == [0]

    out_nms_off = fusion_mod.fuse_image_predictions(
        yolo_s1, frcnn_s1, conf_thres=0.001, global_nms_iou=0.50, global_nms_enabled=False
    )
    assert out_nms_off["boxes"].shape == (2, 4)
    assert out_nms_off["scores"].tolist() == pytest.approx([0.90, 0.70], abs=1e-6)

    # Scenario 2: 2 overlapping boxes (IoU > 0.5) of DIFFERENT classes (car=0 vs person=1) -> keep both
    frcnn_s2 = {
        "boxes": torch.tensor([[105.0, 105.0, 205.0, 205.0]], dtype=torch.float32),
        "scores": torch.tensor([0.70], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    out_diff_cls = fusion_mod.fuse_image_predictions(
        yolo_s1, frcnn_s2, conf_thres=0.001, global_nms_iou=0.50, global_nms_enabled=True
    )
    assert out_diff_cls["boxes"].shape == (2, 4)
    assert out_diff_cls["labels"].tolist() == [0, 1]
    assert out_diff_cls["scores"].tolist() == pytest.approx([0.90, 0.70], abs=1e-6)

    # Scenario 3: Confidence thresholding across [0.25, 0.50, 0.75]
    yolo_s3 = {
        "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0], [100.0, 10.0, 140.0, 40.0]], dtype=torch.float32),
        "scores": torch.tensor([0.10, 0.60], dtype=torch.float32),
        "labels": torch.tensor([0, 1], dtype=torch.int64),
    }
    frcnn_s3 = {
        "boxes": torch.tensor([[200.0, 10.0, 240.0, 40.0], [300.0, 10.0, 340.0, 40.0]], dtype=torch.float32),
        "scores": torch.tensor([0.30, 0.85], dtype=torch.float32),
        "labels": torch.tensor([2, 0], dtype=torch.int64),
    }
    counts_by_tau = [
        fusion_mod.fuse_image_predictions(
            yolo_s3, frcnn_s3, conf_thres=tau, global_nms_iou=0.50, global_nms_enabled=True
        )["boxes"].shape[0]
        for tau in thresholds
    ]
    assert counts_by_tau == [3, 2, 1]

    # Scenario 4: Empty (0, 4) input in one branch or both branches
    empty_b = fusion_mod.empty_prediction()
    out_one_empty_a = fusion_mod.fuse_image_predictions(empty_b, yolo_s1, conf_thres=0.001, global_nms_enabled=True)
    assert out_one_empty_a["boxes"].shape == (1, 4)
    out_one_empty_b = fusion_mod.fuse_image_predictions(yolo_s1, empty_b, conf_thres=0.001, global_nms_enabled=True)
    assert out_one_empty_b["boxes"].shape == (1, 4)
    out_both_empty = fusion_mod.fuse_image_predictions(empty_b, empty_b, conf_thres=0.001, global_nms_enabled=True)
    assert out_both_empty["boxes"].shape == (0, 4) and out_both_empty["boxes"].dtype == torch.float32
    assert out_both_empty["scores"].shape == (0,) and out_both_empty["scores"].dtype == torch.float32
    assert out_both_empty["labels"].shape == (0,) and out_both_empty["labels"].dtype == torch.int64


def test_eval_output() -> None:
    """F07: Verify on-disk dummy metrics report/table and 4 deterministic toy evaluation scenarios."""
    script_path = REPO_ROOT / "src" / "06_evaluate.py"
    if not script_path.is_file():
        pytest.skip("Waiting for F07 (src/06_evaluate.py)")

    import importlib.util
    import math

    import torch

    cfg = load_config()
    artifacts_dir = REPO_ROOT / cfg["paths"]["artifacts_dir"]
    report_path = artifacts_dir / "dummy_metrics_report.json"
    table_path = artifacts_dir / "dummy_ablation_table.md"

    assert report_path.is_file(), f"Missing evaluation JSON report: {report_path}"
    assert table_path.is_file() and table_path.stat().st_size > 0, f"Missing ablation table: {table_path}"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "models" in report
    for model_key in ("yolo", "frcnn", "fusion"):
        assert model_key in report["models"], f"Missing '{model_key}' in {report_path}"
        m = report["models"][model_key]
        for metric_key in ("precision", "recall", "f1", "map50", "map75"):
            assert metric_key in m, f"Missing '{metric_key}' for {model_key}"
            val = float(m[metric_key])
            assert math.isfinite(val) and 0.0 <= val <= 1.0, f"Invalid {metric_key}={val} for {model_key}"

    # Directly import evaluate_predictions() and test 4 deterministic toy scenarios
    spec = importlib.util.spec_from_file_location("eval_mod", script_path)
    assert spec is not None and spec.loader is not None
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)

    class_names = {0: "car", 1: "person", 2: "traffic light"}

    # Scenario 1: Perfect prediction -> AP@50 == 1.0, AP@75 == 1.0, F1 == 1.0
    gt_s1 = {
        1: {
            "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }
    }
    pred_s1 = {
        1: {
            "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32),
            "scores": torch.tensor([0.95], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }
    }
    res_s1 = eval_mod.evaluate_predictions(pred_s1, gt_s1, class_names)
    assert res_s1["map50"] == pytest.approx(1.0, abs=1e-6)
    assert res_s1["map75"] == pytest.approx(1.0, abs=1e-6)
    assert res_s1["precision"] == pytest.approx(1.0, abs=1e-6)
    assert res_s1["recall"] == pytest.approx(1.0, abs=1e-6)
    assert res_s1["f1"] == pytest.approx(1.0, abs=1e-6)

    # Scenario 2: 2 duplicate predictions on same GT (scores 0.9 & 0.7) -> 2nd counts as FP -> Precision = 0.5
    pred_s2 = {
        1: {
            "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0], [100.0, 100.0, 200.0, 200.0]], dtype=torch.float32),
            "scores": torch.tensor([0.90, 0.70], dtype=torch.float32),
            "labels": torch.tensor([0, 0], dtype=torch.long),
        }
    }
    res_s2 = eval_mod.evaluate_predictions(pred_s2, gt_s1, class_names)
    assert res_s2["precision"] == pytest.approx(0.5, abs=1e-6)
    assert res_s2["recall"] == pytest.approx(1.0, abs=1e-6)
    assert res_s2["f1"] == pytest.approx(2.0 / 3.0, abs=1e-6)

    # Scenario 3: Missed 1 of 2 GTs -> Recall = 0.5, Precision = 1.0, AP@50 = 0.5
    gt_s3 = {
        1: {
            "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0], [300.0, 300.0, 400.0, 400.0]], dtype=torch.float32),
            "labels": torch.tensor([0, 0], dtype=torch.long),
        }
    }
    res_s3 = eval_mod.evaluate_predictions(pred_s1, gt_s3, class_names)
    assert res_s3["precision"] == pytest.approx(1.0, abs=1e-6)
    assert res_s3["recall"] == pytest.approx(0.5, abs=1e-6)
    assert res_s3["map50"] == pytest.approx(0.5, abs=1e-6)

    # Scenario 4: Empty (0, 4) predictions & 0-GT classes -> finite 0.0 metrics, no NaN/ZeroDivisionError
    empty_pred = {
        1: {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.long),
        }
    }
    res_s4 = eval_mod.evaluate_predictions(empty_pred, gt_s1, class_names)
    for k in ("precision", "recall", "f1", "map50", "map75"):
        assert math.isfinite(res_s4[k]) and res_s4[k] == 0.0
    assert res_s4["ap_by_iou"]["0.50"]["per_class"]["person"] is None
    assert res_s4["ap_by_iou"]["0.50"]["per_class"]["traffic light"] is None

    empty_gt = {
        1: {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.long),
        }
    }
    res_s4_no_gt = eval_mod.evaluate_predictions(empty_pred, empty_gt, class_names)
    for k in ("precision", "recall", "f1", "map50", "map75"):
        assert math.isfinite(res_s4_no_gt[k]) and res_s4_no_gt[k] == 0.0


def test_notebook_clean() -> None:
    """F08: Verify notebooks/Colab_Runner.ipynb is thin (shell/magic + Drive mount only) and covers all scripts & E0–E5."""
    nb_path = REPO_ROOT / "notebooks" / "Colab_Runner.ipynb"
    if not nb_path.is_file():
        pytest.skip("Waiting for F08 (notebooks/Colab_Runner.ipynb)")

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4, f"Expected nbformat 4, got {nb.get('nbformat')}"

    allowed_python_lines = {
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        'drive.mount("/content/drive")',
    }
    all_code_lines: list[str] = []

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for raw_line in cell.get("source", []):
            line = raw_line.strip()
            all_code_lines.append(line)
            if (
                not line
                or line.startswith("#")
                or line.startswith("!")
                or line.startswith("%")
                or line in allowed_python_lines
            ):
                continue
            raise AssertionError(f"Non-shell Python logic found in Colab_Runner.ipynb: {line!r}")

    joined_code = "\n".join(all_code_lines)
    required_scripts = (
        "01_prepare_bdd.py",
        "02_ghe_ensemble.py",
        "03_train_infer_yolo.py",
        "04_train_infer_frcnn.py",
        "05_fusion.py",
        "06_evaluate.py",
    )
    for script_name in required_scripts:
        assert script_name in joined_code, f"Colab_Runner.ipynb missing invocation of {script_name}"

    for exp_tag in ("E0", "E1", "E2", "E3", "E4", "E5"):
        assert exp_tag in joined_code, f"Colab_Runner.ipynb missing experiment {exp_tag}"
