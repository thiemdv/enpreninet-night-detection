"""F07: Evaluation & Ablation Benchmark Module.

Computes Precision, Recall, F1-Score (per-class, micro, macro) across
confidence thresholds [0.25, 0.50, 0.75] and AP@50/mAP@50, AP@75/mAP@75
using Pascal VOC greedy best-IoU 1-to-1 matching with all-point PR interpolation
(excluding 0-GT classes from mAP mean).

Enforces fail-closed validation of `dataset_fingerprint` and `eval_split`
between every prediction `.meta.json` sidecar and COCO GT (opt-in override
via `--allow-legacy-predictions`), plus strict `set(preds) == set(gt)` equality.

Usage:
    python src/06_evaluate.py --config configs/config.yaml --dummy --device cpu
    python src/06_evaluate.py --config configs/config.yaml --run-id pilot --data-root /content/work/pilot_processed --preds ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_IMAGE_VARIANTS = ("raw", "fig4_literal", "fig4_normalized", "standard_local_he")
MATCHING_PROTOCOL = "pascal_voc_greedy_best_iou_all_point_ap"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def resolve_eval_output_paths(cfg: dict[str, Any], run_id: str) -> tuple[Path, Path]:
    """Return isolated (json_report_path, md_table_path) under artifacts/<run_id>/."""
    out_dir = resolve_path(cfg["paths"]["artifacts_dir"]) / run_id
    return out_dir / "metrics_report.json", out_dir / "ablation_table.md"


def read_prediction_meta(preds_path: Path) -> dict[str, Any] | None:
    meta_path = preds_path.with_name(f"{preds_path.name}.meta.json")
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def verify_prediction_against_gt(
    model_name: str,
    pred_path: Path,
    gt_fingerprint: str | None,
    eval_split: str,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """A02 fail-closed check: verify prediction sidecar matches GT dataset_fingerprint & eval_split."""
    pred_meta = read_prediction_meta(pred_path)
    if not allow_legacy and pred_meta is None:
        raise ValueError(
            f"Missing sidecar metadata (.meta.json) for '{model_name}' ({pred_path}). "
            f"Pass --allow-legacy-predictions to override."
        )
    pm = pred_meta or {}
    pred_fp = pm.get("dataset_fingerprint")

    if not allow_legacy and (not gt_fingerprint or not pred_fp):
        raise ValueError(
            f"Missing dataset_fingerprint for '{model_name}': pred={pred_fp}, gt={gt_fingerprint}. "
            f"Pass --allow-legacy-predictions to override."
        )
    if gt_fingerprint and pred_fp and str(pred_fp) != str(gt_fingerprint):
        raise ValueError(
            f"Dataset fingerprint mismatch for '{model_name}': "
            f"pred={pred_fp} != gt={gt_fingerprint}. Refusing to evaluate mismatched dataset/split."
        )

    pred_split = pm.get("eval_split")
    if pred_split and str(pred_split) != str(eval_split):
        raise ValueError(
            f"Eval split mismatch for '{model_name}': pred={pred_split} != requested eval_split={eval_split}."
        )
    return pm


def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU matrix of shape (M, K) for xyxy boxes1 (M, 4) and boxes2 (K, 4)."""
    b1 = torch.as_tensor(boxes1, dtype=torch.float32)
    b2 = torch.as_tensor(boxes2, dtype=torch.float32)
    m = b1.shape[0] if b1.ndim == 2 else 0
    k = b2.shape[0] if b2.ndim == 2 else 0
    if m == 0 or k == 0:
        return torch.zeros((m, k), dtype=torch.float32)

    lt = torch.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = torch.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0.0) * (b1[:, 3] - b1[:, 1]).clamp(min=0.0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0.0) * (b2[:, 3] - b2[:, 1]).clamp(min=0.0)
    union = area1[:, None] + area2[None, :] - inter
    return torch.where(union > 0.0, inter / union, torch.zeros_like(inter))


def _compute_ap_all_point(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        if mpre[idx - 1] < mpre[idx]:
            mpre[idx - 1] = mpre[idx]
    change_idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[change_idx + 1] - mrec[change_idx]) * mpre[change_idx + 1]))
    return max(0.0, min(1.0, ap))


def _match_class_detections(
    preds_by_image: dict[int, dict[str, torch.Tensor]],
    gt_by_image: dict[int, dict[str, torch.Tensor]],
    class_id: int,
    iou_threshold: float,
    conf_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pascal VOC greedy best-IoU 1-to-1 matching for a single class at given IoU & confidence thresholds."""
    all_image_ids = sorted(set(gt_by_image.keys()) | set(preds_by_image.keys()))

    gt_boxes_map: dict[int, torch.Tensor] = {}
    gt_matched_map: dict[int, list[bool]] = {}
    total_gt = 0
    for img_id in all_image_ids:
        gt = gt_by_image.get(img_id)
        if gt is not None and gt["boxes"].numel() > 0:
            mask = gt["labels"].to(torch.long) == class_id
            cls_boxes = gt["boxes"][mask].detach().cpu().to(torch.float32)
        else:
            cls_boxes = torch.empty((0, 4), dtype=torch.float32)
        gt_boxes_map[img_id] = cls_boxes
        gt_matched_map[img_id] = [False] * int(cls_boxes.shape[0])
        total_gt += int(cls_boxes.shape[0])

    det_records: list[tuple[float, int, torch.Tensor]] = []
    for img_id in all_image_ids:
        # A08 fix: safe .get() so strict_image_ids=False never triggers KeyError
        pred = preds_by_image.get(img_id)
        if pred is None or pred["boxes"].numel() == 0:
            continue
        boxes = pred["boxes"].detach().cpu().to(torch.float32)
        scores = pred["scores"].detach().cpu().to(torch.float32)
        labels = pred["labels"].detach().cpu().to(torch.long)
        mask = (labels == class_id) & (scores >= float(conf_threshold))
        if not bool(mask.any()):
            continue
        for b, s in zip(boxes[mask], scores[mask]):
            det_records.append((float(s.item()), img_id, b))

    det_records.sort(key=lambda item: item[0], reverse=True)

    tp = np.zeros(len(det_records), dtype=np.float64)
    fp = np.zeros(len(det_records), dtype=np.float64)

    for idx, (_score, img_id, pred_box) in enumerate(det_records):
        gt_boxes = gt_boxes_map[img_id]
        if gt_boxes.shape[0] == 0:
            fp[idx] = 1.0
            continue
        ious = compute_iou_matrix(pred_box.unsqueeze(0), gt_boxes)[0]
        best_iou_val, best_gt_idx_t = ious.max(dim=0)
        best_iou = float(best_iou_val.item())
        best_gt_idx = int(best_gt_idx_t.item())
        if best_iou >= float(iou_threshold) and not gt_matched_map[img_id][best_gt_idx]:
            gt_matched_map[img_id][best_gt_idx] = True
            tp[idx] = 1.0
        else:
            fp[idx] = 1.0

    return tp, fp, total_gt


def _safe_prf(tp_count: float, fp_count: float, fn_count: float) -> tuple[float, float, float]:
    precision = float(tp_count / (tp_count + fp_count)) if (tp_count + fp_count) > 0.0 else 0.0
    recall = float(tp_count / (tp_count + fn_count)) if (tp_count + fn_count) > 0.0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    return precision, recall, f1


def evaluate_predictions(
    preds_by_image: dict[int, dict[str, torch.Tensor]],
    gt_by_image: dict[int, dict[str, torch.Tensor]],
    class_names: dict[int, str],
    iou_thresholds: list[float] | tuple[float, ...] = (0.50, 0.75),
    conf_thresholds: list[float] | tuple[float, ...] = (0.25, 0.50, 0.75),
    strict_image_ids: bool = True,
) -> dict[str, Any]:
    """Evaluate unified predictions against ground truth across IoU and confidence thresholds."""
    pred_ids = set(preds_by_image.keys())
    gt_ids = set(gt_by_image.keys())
    if strict_image_ids and pred_ids != gt_ids:
        missing_in_preds = sorted(gt_ids - pred_ids)
        extra_in_preds = sorted(pred_ids - gt_ids)
        raise ValueError(
            f"Prediction image_ids do not match GT image_ids! "
            f"missing_in_preds={missing_in_preds[:5]}, extra_in_preds={extra_in_preds[:5]}"
        )

    sorted_classes = sorted((int(cid), str(cname)) for cid, cname in class_names.items())
    ap_by_iou: dict[str, Any] = {}
    operating_points: dict[str, Any] = {}

    for iou_thr in iou_thresholds:
        iou_key = f"{float(iou_thr):.2f}"
        per_class_ap: dict[str, float | None] = {}
        valid_aps: list[float] = []

        for cid, cname in sorted_classes:
            tp, fp, total_gt = _match_class_detections(
                preds_by_image=preds_by_image,
                gt_by_image=gt_by_image,
                class_id=cid,
                iou_threshold=float(iou_thr),
                conf_threshold=0.0,
            )
            if total_gt == 0:
                per_class_ap[cname] = None
                continue
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / float(total_gt)
            precisions = np.divide(cum_tp, cum_tp + cum_fp, out=np.zeros_like(cum_tp), where=(cum_tp + cum_fp) > 0)
            ap_val = _compute_ap_all_point(recalls, precisions)
            per_class_ap[cname] = ap_val
            valid_aps.append(ap_val)

        map_val = float(np.mean(valid_aps)) if valid_aps else 0.0
        ap_by_iou[iou_key] = {"map": map_val, "per_class": per_class_ap}

        op_for_iou: dict[str, Any] = {}
        for conf_thr in conf_thresholds:
            conf_key = f"{float(conf_thr):.2f}"
            per_class_prf: dict[str, dict[str, float | int]] = {}
            micro_tp = 0.0
            micro_fp = 0.0
            micro_fn = 0.0
            macro_ps: list[float] = []
            macro_rs: list[float] = []
            macro_f1s: list[float] = []

            for cid, cname in sorted_classes:
                tp_arr, fp_arr, total_gt = _match_class_detections(
                    preds_by_image=preds_by_image,
                    gt_by_image=gt_by_image,
                    class_id=cid,
                    iou_threshold=float(iou_thr),
                    conf_threshold=float(conf_thr),
                )
                tp_sum = float(tp_arr.sum())
                fp_sum = float(fp_arr.sum())
                fn_sum = float(total_gt) - tp_sum
                p, r, f1 = _safe_prf(tp_sum, fp_sum, fn_sum)

                per_class_prf[cname] = {
                    "tp": int(tp_sum),
                    "fp": int(fp_sum),
                    "fn": int(fn_sum),
                    "gt": int(total_gt),
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                }
                micro_tp += tp_sum
                micro_fp += fp_sum
                micro_fn += fn_sum
                if total_gt > 0:
                    macro_ps.append(p)
                    macro_rs.append(r)
                    macro_f1s.append(f1)

            micro_p, micro_r, micro_f1 = _safe_prf(micro_tp, micro_fp, micro_fn)
            macro_p = float(np.mean(macro_ps)) if macro_ps else 0.0
            macro_r = float(np.mean(macro_rs)) if macro_rs else 0.0
            macro_f1 = float(np.mean(macro_f1s)) if macro_f1s else 0.0

            op_for_iou[conf_key] = {
                "micro": {
                    "tp": int(micro_tp),
                    "fp": int(micro_fp),
                    "fn": int(micro_fn),
                    "precision": micro_p,
                    "recall": micro_r,
                    "f1": micro_f1,
                },
                "macro": {
                    "precision": macro_p,
                    "recall": macro_r,
                    "f1": macro_f1,
                },
                "per_class": per_class_prf,
            }
        operating_points[iou_key] = op_for_iou

    primary_iou = "0.50" if "0.50" in operating_points else next(iter(operating_points.keys()))
    primary_conf = "0.25" if "0.25" in operating_points[primary_iou] else next(iter(operating_points[primary_iou].keys()))
    primary_micro = operating_points[primary_iou][primary_conf]["micro"]

    return {
        "matching_protocol": MATCHING_PROTOCOL,
        "precision": float(primary_micro["precision"]),
        "recall": float(primary_micro["recall"]),
        "f1": float(primary_micro["f1"]),
        "map50": float(ap_by_iou["0.50"]["map"]) if "0.50" in ap_by_iou else 0.0,
        "map75": float(ap_by_iou["0.75"]["map"]) if "0.75" in ap_by_iou else 0.0,
        "ap_by_iou": ap_by_iou,
        "operating_points": operating_points,
    }


def load_coco_ground_truth(coco_json_path: Path) -> tuple[dict[int, dict[str, torch.Tensor]], str | None]:
    if not coco_json_path.is_file():
        raise FileNotFoundError(f"Ground truth COCO JSON not found: {coco_json_path}")
    data = json.loads(coco_json_path.read_text(encoding="utf-8"))
    fp = (data.get("info") or {}).get("dataset_fingerprint")

    boxes_by_img: dict[int, list[list[float]]] = {int(img["id"]): [] for img in data.get("images", [])}
    labels_by_img: dict[int, list[int]] = {int(img["id"]): [] for img in data.get("images", [])}

    for ann in data.get("annotations", []):
        img_id = int(ann["image_id"])
        if img_id not in boxes_by_img:
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        if w <= 0.0 or h <= 0.0:
            continue
        boxes_by_img[img_id].append([x, y, x + w, y + h])
        labels_by_img[img_id].append(int(ann["category_id"]))

    gt_by_image: dict[int, dict[str, torch.Tensor]] = {}
    for img_id in sorted(boxes_by_img.keys()):
        if boxes_by_img[img_id]:
            gt_by_image[img_id] = {
                "boxes": torch.tensor(boxes_by_img[img_id], dtype=torch.float32),
                "labels": torch.tensor(labels_by_img[img_id], dtype=torch.long),
            }
        else:
            gt_by_image[img_id] = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.long),
            }
    return gt_by_image, (str(fp) if fp else None)


def format_ablation_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# EnPreNiNet Evaluation & Ablation Summary (run_id={report['run_id']} | split={report['eval_split']})",
        "",
        f"> **Protocol:** `{report['matching_protocol']}` (Pascal VOC greedy best-IoU 1-to-1 matching + all-point PR interpolation; 0-GT classes excluded from mAP mean).",
        f"> **Dataset Fingerprint:** `{report.get('dataset_fingerprint')}`",
        "",
        "| Model | mAP@50 | mAP@75 | Precision@0.25 | Recall@0.25 | F1@0.25 | F1@0.50 | F1@0.75 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for model_name, metrics in report["models"].items():
        op50 = metrics["operating_points"].get("0.50", {})
        f1_25 = op50.get("0.25", {}).get("micro", {}).get("f1", 0.0)
        f1_50 = op50.get("0.50", {}).get("micro", {}).get("f1", 0.0)
        f1_75 = op50.get("0.75", {}).get("micro", {}).get("f1", 0.0)
        lines.append(
            f"| `{model_name}` | {metrics['map50']:.4f} | {metrics['map75']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{f1_25:.4f} | {f1_50:.4f} | {f1_75:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_pred_specs(
    pred_args: list[str] | None,
    cfg: dict[str, Any],
    run_id: str,
    image_variant: str,
    eval_split: str,
) -> dict[str, Path]:
    if pred_args:
        parsed: dict[str, Path] = {}
        for item in pred_args:
            if "=" not in item:
                raise ValueError(f"Expected --preds entry in 'name=path' format, got {item!r}")
            name, raw_path = item.split("=", 1)
            parsed[name.strip()] = resolve_path(raw_path.strip())
        return parsed

    run_root = resolve_path(cfg["paths"]["runs_dir"]) / run_id
    return {
        "yolo": run_root / f"yolo_{image_variant}" / f"yolo_{image_variant}_{eval_split}_preds.pt",
        "frcnn": run_root / f"frcnn_{image_variant}" / f"frcnn_{image_variant}_{eval_split}_preds.pt",
        "fusion": run_root / f"fusion_{image_variant}" / f"fusion_{image_variant}_{eval_split}_preds.pt",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EnPreNiNet detectors and fusion outputs.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    parser.add_argument("--run-id", type=str, default=None, help="Run namespace isolating outputs (dummy, pilot, full)")
    parser.add_argument(
        "--preds",
        nargs="+",
        default=None,
        help="List of name=path/to/preds.pt entries to evaluate (defaults to yolo, frcnn, fusion in runs/<run_id>/)",
    )
    parser.add_argument(
        "--image-variant",
        type=str,
        choices=VALID_IMAGE_VARIANTS,
        default=None,
        help="Image variant used for default prediction discovery",
    )
    parser.add_argument("--gt-coco", type=Path, default=None, help="Override COCO ground-truth JSON path")
    parser.add_argument("--data-root", type=Path, default=None, help="Override processed data root directory")
    parser.add_argument("--eval-split", type=str, choices=("val", "test"), default="test", help="Evaluation split")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override artifacts output directory")
    parser.add_argument("--device", type=str, default=None, help="Device flag (evaluation runs on CPU tensors)")
    parser.add_argument(
        "--allow-legacy-predictions",
        action="store_true",
        default=False,
        help="Allow evaluating prediction .pt files lacking .meta.json fingerprints",
    )
    parser.add_argument("--dummy", action="store_true", help="Evaluate runs/dummy/ against data/dummy_processed GT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    cfg = load_config(config_path)

    run_id = args.run_id or ("dummy" if args.dummy else "full")
    image_variant = args.image_variant or str(cfg["preprocessing"]["ghe_ensemble"]["method"])
    default_root = resolve_path(cfg["paths"]["data_dummy_processed"] if args.dummy else cfg["paths"]["data_processed"])
    data_root = args.data_root or default_root
    gt_coco_path = args.gt_coco or (data_root / cfg["paths"]["coco_subdir"] / f"instances_{args.eval_split}.json")

    if args.output_dir is not None:
        json_path = args.output_dir / "metrics_report.json"
        md_path = args.output_dir / "ablation_table.md"
    else:
        json_path, md_path = resolve_eval_output_paths(cfg, run_id=run_id)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    gt_by_image, gt_fingerprint = load_coco_ground_truth(gt_coco_path)
    class_names = {int(k): str(v) for k, v in cfg["dataset"]["classes"].items()}
    iou_thresholds = [float(t) for t in cfg["evaluation"]["iou_thresholds"]]
    conf_thresholds = [float(t) for t in cfg["fusion"]["confidence_thresholds"]]

    pred_specs = parse_pred_specs(
        pred_args=args.preds,
        cfg=cfg,
        run_id=run_id,
        image_variant=image_variant,
        eval_split=args.eval_split,
    )

    models_report: dict[str, Any] = {}
    for model_name, pred_path in pred_specs.items():
        if not pred_path.is_file():
            raise FileNotFoundError(f"Prediction file for '{model_name}' not found: {pred_path}")

        verify_prediction_against_gt(
            model_name=model_name,
            pred_path=pred_path,
            gt_fingerprint=gt_fingerprint,
            eval_split=args.eval_split,
            allow_legacy=args.allow_legacy_predictions,
        )

        preds_by_image = torch.load(pred_path, weights_only=True)
        models_report[model_name] = evaluate_predictions(
            preds_by_image=preds_by_image,
            gt_by_image=gt_by_image,
            class_names=class_names,
            iou_thresholds=iou_thresholds,
            conf_thresholds=conf_thresholds,
            strict_image_ids=True,
        )

    report = {
        "mode": "dummy" if args.dummy else "real",
        "run_id": run_id,
        "eval_split": args.eval_split,
        "image_variant": image_variant,
        "gt_coco": str(gt_coco_path),
        "dataset_fingerprint": gt_fingerprint,
        "matching_protocol": MATCHING_PROTOCOL,
        "iou_thresholds": iou_thresholds,
        "conf_thresholds": conf_thresholds,
        "models": models_report,
    }

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(format_ablation_markdown(report), encoding="utf-8")

    compact = {
        name: {k: round(float(m[k]), 4) for k in ("precision", "recall", "f1", "map50", "map75")}
        for name, m in models_report.items()
    }
    print(f"[06_evaluate] Completed (run_id={run_id}) -> {json_path} & {md_path}")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
