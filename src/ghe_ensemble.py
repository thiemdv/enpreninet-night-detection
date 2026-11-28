"""F03: GHE Ensemble Preprocessor (15-Step Algorithm, Fig. 4 & Tiled-Interpolated Modes).

Implements the 15-step Gaussian + local histogram equalization pipeline supporting
five explicit modes:
  - tiled_ghe_gray     [PROJECT/PAPER]: Normalized PDF = f(x) / N, s'(x) = round(H * C(x)),
                                        W x W tile grid (default W=64, sigma=1.0) with bilinear
                                        LUT interpolation across neighboring tiles & optional
                                        clip_limit (CLAHE-like), stacked into (640, 640, 3).
  - tiled_ghe_lab      [PROJECT]:       Color-preserving variant applying tiled GHE to the L
                                        channel in CIE LAB space (preserves traffic-light color).
  - fig4_literal       [PAPER]:         Sliding window PDF(x) = f(x) / H, s'(x) = clip(round(H * C(x)), 0, 255).
  - fig4_normalized    [PROJECT]:       Sliding window PDF(x) = f(x) / W^2, s'(x) = round(H * C(x)).
  - standard_local_he  [PROJECT]:       Sliding window PDF(x) = f(x) / W^2, s'(x) = round(255 * C(x)).

Reads split manifests from <root>/splits/{train,val,test}.txt, cleans stale
outputs under <root>/images/<method>/, writes lossless (640, 640, 3) PNG images,
records preprocessing metadata + preprocessing_fingerprint + timing in ghe_meta.json,
and saves a visual comparison artifact under artifacts/<run_id>/.

Usage:
    python src/ghe_ensemble.py --config configs/config.yaml --dummy
    python src/ghe_ensemble.py --config configs/config.yaml --run-id pilot --data-root /content/work/pilot_processed
    python src/ghe_ensemble.py --config configs/config.yaml --run-id full --method tiled_ghe_lab --clip-limit 3.0
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import yaml
from numpy.lib.stride_tricks import sliding_window_view

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

SLIDING_METHODS = ("fig4_literal", "fig4_normalized", "standard_local_he")
TILED_METHODS = ("tiled_ghe_gray", "tiled_ghe_lab")
VALID_METHODS = (*SLIDING_METHODS, *TILED_METHODS)

CV2_BORDER_MAP = {
    "reflect": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
}
NP_PAD_MODE_MAP = {
    "reflect": "reflect",
    "replicate": "edge",
    "constant": "constant",
}


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def resolve_ghe_artifact_path(cfg: dict[str, Any], run_id: str, method: str) -> Path:
    """Resolve isolated comparison artifact path under artifacts/<run_id>/."""
    return resolve_path(cfg["paths"]["artifacts_dir"]) / run_id / f"ghe_comparison_{method}.png"


def compute_preprocessing_fingerprint(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_gaussian_kernel(kernel_size: int = 3, sigma: float = 2.0) -> np.ndarray:
    """Steps 4-7: Build normalized 2D Gaussian kernel of size k x k with std sigma."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"gaussian_kernel_size must be a positive odd integer, got {kernel_size}")
    if sigma <= 0.0:
        raise ValueError(f"gaussian_sigma must be > 0, got {sigma}")
    half = kernel_size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def apply_gaussian_filter(
    gray: np.ndarray,
    kernel: np.ndarray,
    border_mode: str = "reflect",
) -> np.ndarray:
    """Step 8: Convolve single-channel image with normalized Gaussian kernel."""
    if border_mode not in CV2_BORDER_MAP:
        raise ValueError(f"Unsupported border_mode: {border_mode!r}")
    filtered = cv2.filter2D(
        gray.astype(np.float64),
        ddepth=cv2.CV_64F,
        kernel=kernel,
        borderType=CV2_BORDER_MAP[border_mode],
    )
    return np.rint(np.clip(filtered, 0.0, 255.0)).astype(np.uint8)


def _tile_lut(tile: np.ndarray, clip_limit: float = 0.0) -> np.ndarray:
    """Steps 6-12 for a single tile: compute 256-entry LUT mapping [0, H] via round(H * CDF)."""
    h_max = int(tile.max())
    lut = np.arange(256, dtype=np.uint8)
    if h_max == 0:
        return lut

    hist = np.bincount(tile.ravel(), minlength=h_max + 1)[: h_max + 1].astype(np.float64)
    if clip_limit > 0.0:
        limit = max(1.0, float(clip_limit) * float(tile.size) / float(h_max + 1))
        excess = np.maximum(hist - limit, 0.0).sum()
        hist = np.minimum(hist, limit) + excess / float(h_max + 1)

    pdf = hist / hist.sum()
    cdf = np.cumsum(pdf)
    lut[: h_max + 1] = np.rint(float(h_max) * cdf).clip(0, 255).astype(np.uint8)
    lut[h_max + 1 :] = h_max
    return lut


def tiled_window_ghe(
    gray_gauss: np.ndarray,
    window_size: int = 64,
    clip_limit: float = 0.0,
    interpolate: bool = True,
) -> np.ndarray:
    """Steps 9-15 (tiled): W x W grid histogram equalization with optional bilinear LUT interpolation."""
    if window_size < 2:
        raise ValueError(f"Tiled window_size must be >= 2, got {window_size}")
    if clip_limit < 0.0:
        raise ValueError(f"clip_limit must be >= 0, got {clip_limit}")

    height, width = gray_gauss.shape
    if not interpolate:
        out = np.empty_like(gray_gauss)
        for y in range(0, height, window_size):
            for x in range(0, width, window_size):
                y2 = min(y + window_size, height)
                x2 = min(x + window_size, width)
                tile = gray_gauss[y:y2, x:x2]
                out[y:y2, x:x2] = _tile_lut(tile, clip_limit=clip_limit)[tile]
        return out

    ny = max(1, -(-height // window_size))
    nx = max(1, -(-width // window_size))
    luts = np.empty((ny, nx, 256), dtype=np.uint8)
    for i in range(ny):
        y = i * window_size
        y2 = min(y + window_size, height)
        for j in range(nx):
            x = j * window_size
            x2 = min(x + window_size, width)
            luts[i, j] = _tile_lut(gray_gauss[y:y2, x:x2], clip_limit=clip_limit)

    yy = np.arange(height, dtype=np.float64)
    xx = np.arange(width, dtype=np.float64)
    fy = np.clip((yy - window_size / 2.0) / float(window_size), 0.0, float(ny - 1))
    fx = np.clip((xx - window_size / 2.0) / float(window_size), 0.0, float(nx - 1))
    i0 = np.floor(fy).astype(np.intp)
    i1 = np.minimum(i0 + 1, ny - 1)
    a = (fy - i0)[:, None]
    j0 = np.floor(fx).astype(np.intp)
    j1 = np.minimum(j0 + 1, nx - 1)
    b = (fx - j0)[None, :]

    # Direct 3D fancy indexing avoids allocating four (H, W, 256) intermediate arrays (~416 MB/img -> ~1.6 MB/img)
    v = gray_gauss
    m00 = luts[i0[:, None], j0[None, :], v]
    m01 = luts[i0[:, None], j1[None, :], v]
    m10 = luts[i1[:, None], j0[None, :], v]
    m11 = luts[i1[:, None], j1[None, :], v]

    out = (
        (1.0 - a) * (1.0 - b) * m00
        + (1.0 - a) * b * m01
        + a * (1.0 - b) * m10
        + a * b * m11
    )
    return np.rint(out).clip(0, 255).astype(np.uint8)


def sliding_window_ghe(
    gray_gauss: np.ndarray,
    window_size: int = 15,
    method: str = "standard_local_he",
    border_mode: str = "reflect",
    zero_division_policy: str = "identity",
    uniform_window_policy: str = "identity",
    chunk_rows: int = 32,
) -> np.ndarray:
    """Steps 9-15 (sliding): Vectorized row-chunked W x W sliding-window histogram equalization."""
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer >= 3, got {window_size}")
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be > 0, got {chunk_rows}")
    if method not in SLIDING_METHODS:
        raise ValueError(f"Unsupported sliding method {method!r}; expected one of {SLIDING_METHODS}")
    if border_mode not in NP_PAD_MODE_MAP:
        raise ValueError(f"Unsupported border_mode: {border_mode!r}")
    if zero_division_policy not in {"identity", "zero"}:
        raise ValueError(f"Unsupported zero_division_policy: {zero_division_policy!r}")
    if uniform_window_policy not in {"identity", "stretch"}:
        raise ValueError(f"Unsupported uniform_window_policy: {uniform_window_policy!r}")

    height, width = gray_gauss.shape
    pad = window_size // 2
    window_area = float(window_size * window_size)

    pad_kwargs: dict[str, Any] = {"mode": NP_PAD_MODE_MAP[border_mode]}
    if border_mode == "constant":
        pad_kwargs["constant_values"] = 0
    padded = np.pad(gray_gauss, pad_width=pad, **pad_kwargs)

    out = np.empty((height, width), dtype=np.uint8)

    for r0 in range(0, height, chunk_rows):
        r1 = min(r0 + chunk_rows, height)
        strip = padded[r0 : r1 + 2 * pad, :]
        windows = sliding_window_view(strip, (window_size, window_size))
        windows_flat = windows.reshape(r1 - r0, width, window_size * window_size)

        centers = gray_gauss[r0:r1, :]
        count_le = (windows_flat <= centers[..., None]).sum(axis=-1, dtype=np.int32).astype(np.float64)
        h_max = windows_flat.max(axis=-1).astype(np.float64)
        zero_mask = h_max == 0.0

        if method == "fig4_literal":
            h_safe = np.where(zero_mask, 1.0, h_max)
            cdf = count_le / h_safe
            mapped = np.rint(h_safe * cdf)
        elif method == "fig4_normalized":
            cdf = count_le / window_area
            mapped = np.rint(h_max * cdf)
        else:  # standard_local_he
            cdf = count_le / window_area
            mapped = np.rint(255.0 * cdf)
            if uniform_window_policy == "identity":
                win_min = windows_flat.min(axis=-1).astype(np.float64)
                uniform_mask = (win_min == h_max) & (~zero_mask)
                mapped = np.where(uniform_mask, centers.astype(np.float64), mapped)

        if zero_division_policy == "identity":
            mapped = np.where(zero_mask, centers.astype(np.float64), mapped)
        else:
            mapped = np.where(zero_mask, 0.0, mapped)

        out[r0:r1, :] = np.clip(mapped, 0.0, 255.0).astype(np.uint8)

    return out


def ghe_ensemble(
    image: np.ndarray,
    image_size: int | None = 640,
    gaussian_kernel_size: int = 3,
    gaussian_sigma: float = 2.0,
    window_size: int = 15,
    method: str = "standard_local_he",
    border_mode: str = "reflect",
    zero_division_policy: str = "identity",
    uniform_window_policy: str = "identity",
    chunk_rows: int = 32,
    clip_limit: float = 0.0,
    interpolate: bool = True,
) -> np.ndarray:
    """Full 15-step GHE Ensemble pipeline returning a (H, W, 3) uint8 image."""
    if method not in VALID_METHODS:
        raise ValueError(f"Unsupported method {method!r}; expected one of {VALID_METHODS}")

    if image.ndim == 3:
        if image_size is not None and (image.shape[0] != image_size or image.shape[1] != image_size):
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        bgr = image
    elif image.ndim == 2:
        if image_size is not None and (image.shape[0] != image_size or image.shape[1] != image_size):
            gray_in = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        else:
            gray_in = image.copy()
        bgr = np.stack([gray_in, gray_in, gray_in], axis=-1)
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")

    kernel = build_gaussian_kernel(kernel_size=gaussian_kernel_size, sigma=gaussian_sigma)

    if method == "tiled_ghe_lab":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_gauss = apply_gaussian_filter(lab[:, :, 0], kernel=kernel, border_mode=border_mode)
        lab[:, :, 0] = tiled_window_ghe(
            l_gauss,
            window_size=window_size,
            clip_limit=clip_limit,
            interpolate=interpolate,
        )
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else bgr[:, :, 0]
    gray_gauss = apply_gaussian_filter(gray, kernel=kernel, border_mode=border_mode)

    if method == "tiled_ghe_gray":
        equalized = tiled_window_ghe(
            gray_gauss,
            window_size=window_size,
            clip_limit=clip_limit,
            interpolate=interpolate,
        )
        return np.stack([equalized, equalized, equalized], axis=-1)

    equalized = sliding_window_ghe(
        gray_gauss=gray_gauss,
        window_size=window_size,
        method=method,
        border_mode=border_mode,
        zero_division_policy=zero_division_policy,
        uniform_window_policy=uniform_window_policy,
        chunk_rows=chunk_rows,
    )
    return np.stack([equalized, equalized, equalized], axis=-1)


def save_comparison_artifact(
    sample_img: np.ndarray,
    artifact_path: Path,
    image_size: int,
    gaussian_kernel_size: int,
    gaussian_sigma: float,
    window_size: int,
    border_mode: str,
    zero_division_policy: str,
    uniform_window_policy: str,
    chunk_rows: int,
    method: str = "tiled_ghe_gray",
    clip_limit: float = 0.0,
    interpolate: bool = True,
) -> None:
    """Generate a 2x4 visual + intensity histogram comparison artifact."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    resized = (
        cv2.resize(sample_img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        if sample_img.shape[:2] != (image_size, image_size)
        else sample_img
    )
    raw_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized

    # Each entry: (title, display_img, hist_1d_source, is_rgb)
    panels: list[tuple[str, np.ndarray, np.ndarray, bool]] = []
    if method in TILED_METHODS:
        raw_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if resized.ndim == 3 else np.stack([raw_gray] * 3, axis=-1)
        panels.append(("Raw Night Image", raw_rgb, raw_gray, True))

        clip_alt = 3.0 if clip_limit <= 0.0 else clip_limit
        gray_noclip = ghe_ensemble(
            resized,
            image_size=image_size,
            gaussian_kernel_size=gaussian_kernel_size,
            gaussian_sigma=gaussian_sigma,
            window_size=window_size,
            method="tiled_ghe_gray",
            border_mode=border_mode,
            clip_limit=0.0,
            interpolate=interpolate,
        )
        panels.append((f"tiled_ghe_gray (clip=0)", gray_noclip[..., 0], gray_noclip[..., 0], False))

        gray_clip = ghe_ensemble(
            resized,
            image_size=image_size,
            gaussian_kernel_size=gaussian_kernel_size,
            gaussian_sigma=gaussian_sigma,
            window_size=window_size,
            method="tiled_ghe_gray",
            border_mode=border_mode,
            clip_limit=clip_alt,
            interpolate=interpolate,
        )
        panels.append((f"tiled_ghe_gray (clip={clip_alt:g})", gray_clip[..., 0], gray_clip[..., 0], False))

        lab_clip = ghe_ensemble(
            resized,
            image_size=image_size,
            gaussian_kernel_size=gaussian_kernel_size,
            gaussian_sigma=gaussian_sigma,
            window_size=window_size,
            method="tiled_ghe_lab",
            border_mode=border_mode,
            clip_limit=clip_alt,
            interpolate=interpolate,
        )
        lab_rgb = cv2.cvtColor(lab_clip, cv2.COLOR_BGR2RGB)
        lab_gray = cv2.cvtColor(lab_clip, cv2.COLOR_BGR2GRAY)
        panels.append((f"tiled_ghe_lab (clip={clip_alt:g})", lab_rgb, lab_gray, True))
        suptitle = (
            f"EnPreNiNet Tiled GHE Comparison "
            f"(W={window_size}, sigma={gaussian_sigma}, interp={interpolate}, border={border_mode})"
        )
    else:
        panels.append(("Raw Grayscale", raw_gray, raw_gray, False))
        for mode in SLIDING_METHODS:
            out_3ch = ghe_ensemble(
                resized,
                image_size=image_size,
                gaussian_kernel_size=gaussian_kernel_size,
                gaussian_sigma=gaussian_sigma,
                window_size=window_size,
                method=mode,
                border_mode=border_mode,
                zero_division_policy=zero_division_policy,
                uniform_window_policy=uniform_window_policy,
                chunk_rows=chunk_rows,
            )
            panels.append((f"GHE ({mode})", out_3ch[..., 0], out_3ch[..., 0], False))
        suptitle = (
            f"EnPreNiNet Sliding GHE Comparison "
            f"(W={window_size}, sigma={gaussian_sigma}, border={border_mode})"
        )

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), dpi=120)
    for col, (title, disp_img, hist_2d, is_rgb) in enumerate(panels):
        if is_rgb:
            axes[0, col].imshow(disp_img)
        else:
            axes[0, col].imshow(disp_img, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].hist(hist_2d.ravel(), bins=256, range=(0, 255), color="steelblue", alpha=0.85)
        axes[1, col].set_xlim(0, 255)
        axes[1, col].set_xlabel("Intensity")
        if col == 0:
            axes[1, col].set_ylabel("Pixel Count")
        axes[1, col].grid(alpha=0.25)

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(str(artifact_path), bbox_inches="tight")
    plt.close(fig)


def _read_dataset_fingerprint(data_root: Path) -> str | None:
    meta_path = data_root / "dataset_meta.json"
    if not meta_path.is_file():
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    fp = data.get("dataset_fingerprint")
    return str(fp) if fp else None


def _process_single_ghe_task(task: tuple[Any, ...]) -> int:
    """Picklable worker function for multiprocessing GHE preprocessing."""
    (
        img_path_str,
        dst_path_str,
        image_size,
        k_size,
        sigma,
        window_size,
        method,
        border_mode,
        zero_division_policy,
        uniform_window_policy,
        chunk_rows,
        clip_limit,
        interpolate,
    ) = task
    img = cv2.imread(img_path_str, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path_str}")
    ghe_img = ghe_ensemble(
        img,
        image_size=image_size,
        gaussian_kernel_size=k_size,
        gaussian_sigma=sigma,
        window_size=window_size,
        method=method,
        border_mode=border_mode,
        zero_division_policy=zero_division_policy,
        uniform_window_policy=uniform_window_policy,
        chunk_rows=chunk_rows,
        clip_limit=clip_limit,
        interpolate=interpolate,
    )
    if not cv2.imwrite(dst_path_str, ghe_img):
        raise IOError(f"cv2.imwrite failed for {dst_path_str}")
    return 1


def process_dataset(
    cfg: dict[str, Any],
    data_root: Path,
    method: str,
    window_size: int,
    border_mode: str,
    zero_division_policy: str,
    uniform_window_policy: str,
    chunk_rows: int,
    artifact_path: Path,
    workers: int = 1,
    sigma: float | None = None,
    clip_limit: float = 0.0,
    interpolate: bool = True,
) -> dict[str, Any]:
    """Read split manifests from splits/*.txt, clean images/<method>/, and write GHE PNGs + ghe_meta.json."""
    ghe_cfg = cfg["preprocessing"]["ghe_ensemble"]
    image_size = int(ghe_cfg["image_size"])
    k_size = int(ghe_cfg["gaussian_kernel_size"])
    if sigma is None:
        sigma = float(ghe_cfg["tiled_sigma"]) if method in TILED_METHODS else float(ghe_cfg["gaussian_sigma"])
    effective_workers = max(1, int(workers))

    splits_root = data_root / cfg["paths"]["splits_subdir"]
    out_root = data_root / cfg["paths"]["ghe_images_root"] / method

    if not splits_root.is_dir():
        raise FileNotFoundError(
            f"Split manifests directory not found: {splits_root}. Run src/prepare_bdd.py first."
        )

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_fingerprint = _read_dataset_fingerprint(data_root)
    t0 = time.perf_counter()
    total_images = 0
    split_counts: dict[str, int] = {}
    sample_img: np.ndarray | None = None

    for split_name in ("train", "val", "test"):
        manifest_path = splits_root / f"{split_name}.txt"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing split manifest: {manifest_path}")

        split_out = out_root / split_name
        split_out.mkdir(parents=True, exist_ok=True)

        manifest_entries = [
            ln.strip() for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        tasks: list[tuple[Any, ...]] = []
        for rel_path in manifest_entries:
            img_path = data_root / Path(rel_path)
            if sample_img is None:
                first_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if first_img is None:
                    raise RuntimeError(f"Failed to read image listed in {manifest_path}: {img_path}")
                sample_img = first_img
            dst_path = split_out / f"{img_path.stem}.png"
            tasks.append(
                (
                    str(img_path),
                    str(dst_path),
                    image_size,
                    k_size,
                    sigma,
                    window_size,
                    method,
                    border_mode,
                    zero_division_policy,
                    uniform_window_policy,
                    chunk_rows,
                    clip_limit,
                    interpolate,
                )
            )

        if effective_workers == 1 or len(tasks) <= 1:
            count = sum(_process_single_ghe_task(t) for t in tasks)
        else:
            with ProcessPoolExecutor(max_workers=effective_workers) as pool:
                count = sum(
                    pool.map(_process_single_ghe_task, tasks, chunksize=max(1, len(tasks) // (effective_workers * 4)))
                )

        total_images += count
        split_counts[split_name] = count

    elapsed_sec = time.perf_counter() - t0
    ms_per_image = (elapsed_sec * 1000.0 / total_images) if total_images > 0 else 0.0

    if sample_img is not None:
        save_comparison_artifact(
            sample_img=sample_img,
            artifact_path=artifact_path,
            image_size=image_size,
            gaussian_kernel_size=k_size,
            gaussian_sigma=sigma,
            window_size=window_size,
            border_mode=border_mode,
            zero_division_policy=zero_division_policy,
            uniform_window_policy=uniform_window_policy,
            chunk_rows=chunk_rows,
            method=method,
            clip_limit=clip_limit,
            interpolate=interpolate,
        )

    preproc_params = {
        "dataset_fingerprint": dataset_fingerprint,
        "method": method,
        "image_size": image_size,
        "gaussian_kernel_size": k_size,
        "gaussian_sigma": sigma,
        "window_size": window_size,
        "border_mode": border_mode,
        "zero_division_policy": zero_division_policy,
        "uniform_window_policy": uniform_window_policy,
        "clip_limit": clip_limit,
        "interpolate": interpolate,
    }
    stats: dict[str, Any] = {
        **preproc_params,
        "preprocessing_fingerprint": compute_preprocessing_fingerprint(preproc_params),
        "chunk_rows": chunk_rows,
        "workers": effective_workers,
        "total_images": total_images,
        "elapsed_seconds": round(elapsed_sec, 4),
        "ms_per_image": round(ms_per_image, 2),
        "output_root": str(out_root),
        "splits": split_counts,
        "artifact": str(artifact_path),
    }
    (out_root / "ghe_meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply 15-step GHE Ensemble preprocessing to night-vision images.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    parser.add_argument("--dummy", action="store_true", help="Run on isolated data/dummy_processed")
    parser.add_argument("--run-id", type=str, default=None, help="Run namespace for artifacts (dummy, pilot, full)")
    parser.add_argument("--method", type=str, choices=VALID_METHODS, default=None, help="Override GHE mode")
    parser.add_argument("--sigma", type=float, default=None, help="Override Gaussian kernel sigma")
    parser.add_argument("--window-size", type=int, default=None, help="Override window/tile size W")
    parser.add_argument("--clip-limit", type=float, default=None, help="Override tiled GHE histogram clip_limit (>=0)")
    interp_group = parser.add_mutually_exclusive_group()
    interp_group.add_argument("--interpolate", dest="interpolate", action="store_true", help="Enable bilinear LUT interp")
    interp_group.add_argument("--no-interpolate", dest="interpolate", action="store_false", help="Disable bilinear interp")
    parser.set_defaults(interpolate=None)
    parser.add_argument("--border-mode", type=str, default=None, help="Override border padding mode")
    parser.add_argument("--zero-division-policy", type=str, default=None, help="Override H=0 policy")
    parser.add_argument("--uniform-window-policy", type=str, default=None, help="Override uniform window policy")
    parser.add_argument("--chunk-rows", type=int, default=32, help="Number of rows per vectorized sliding chunk")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel CPU worker processes (0=auto: 1 in dummy, min(cpu_count, 8) in real mode)",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Override processed data root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    cfg = load_config(config_path)
    ghe_cfg = cfg["preprocessing"]["ghe_ensemble"]

    run_id = args.run_id or ("dummy" if args.dummy else "full")
    method = args.method or str(ghe_cfg["method"])
    is_tiled = method in TILED_METHODS

    if args.window_size is not None:
        window_size = int(args.window_size)
    else:
        window_size = int(ghe_cfg["tiled_window_size"] if is_tiled else ghe_cfg["window_size"])

    if args.sigma is not None:
        sigma = float(args.sigma)
    else:
        sigma = float(ghe_cfg["tiled_sigma"] if is_tiled else ghe_cfg["gaussian_sigma"])

    clip_limit = float(args.clip_limit) if args.clip_limit is not None else float(ghe_cfg.get("clip_limit", 0.0))
    interpolate = bool(args.interpolate) if args.interpolate is not None else bool(ghe_cfg.get("interpolate", True))

    border_mode = args.border_mode or str(ghe_cfg["border_mode"])
    zero_div = args.zero_division_policy or str(ghe_cfg["zero_division_policy"])
    uniform_pol = args.uniform_window_policy or str(ghe_cfg["uniform_window_policy"])

    if args.workers > 0:
        workers = args.workers
    elif args.dummy:
        workers = 1
    else:
        workers = max(1, min(os.cpu_count() or 1, 8))

    default_root = resolve_path(cfg["paths"]["data_dummy_processed"] if args.dummy else cfg["paths"]["data_processed"])
    data_root = args.data_root or default_root
    artifact_path = resolve_ghe_artifact_path(cfg, run_id=run_id, method=method)

    stats = process_dataset(
        cfg=cfg,
        data_root=data_root,
        method=method,
        window_size=window_size,
        border_mode=border_mode,
        zero_division_policy=zero_div,
        uniform_window_policy=uniform_pol,
        chunk_rows=args.chunk_rows,
        artifact_path=artifact_path,
        workers=workers,
        sigma=sigma,
        clip_limit=clip_limit,
        interpolate=interpolate,
    )
    print(f"[ghe_ensemble] Completed (run_id={run_id}, method={method}, workers={workers})")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
