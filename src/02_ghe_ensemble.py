"""F03: GHE Ensemble Preprocessor (15-Step Algorithm, Fig. 4).

Implements the 15-step Gaussian + vectorized row-chunked sliding-window
histogram equalization pipeline supporting three explicit gray-level
transformation modes:
  - fig4_literal       [PAPER]:   PDF(x) = f(x) / H,   s'(x) = clip(round(H * C(x)), 0, 255)
  - fig4_normalized    [PROJECT]: PDF(x) = f(x) / W^2, s'(x) = round(H * C(x))
  - standard_local_he  [PROJECT]: PDF(x) = f(x) / W^2, s'(x) = round(255 * C(x))

Reads resized raw images from <root>/images/raw/{train,val,test}/, writes
lossless (640, 640, 3) PNG images to <root>/images/<method>/{train,val,test}/,
and generates a visual comparison + histogram artifact in artifacts/.

Usage:
    python src/02_ghe_ensemble.py --config configs/config.yaml --dummy
    python src/02_ghe_ensemble.py --config configs/config.yaml --method fig4_normalized
"""

from __future__ import annotations

import argparse
import json
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

VALID_METHODS = ("fig4_literal", "fig4_normalized", "standard_local_he")
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


def build_gaussian_kernel(kernel_size: int = 3, sigma: float = 2.0) -> np.ndarray:
    """Steps 4-7: Build normalized 2D Gaussian kernel of size k x k with std sigma."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"gaussian_kernel_size must be a positive odd integer, got {kernel_size}")
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
    """Step 8: Convolve grayscale image with normalized Gaussian kernel."""
    if border_mode not in CV2_BORDER_MAP:
        raise ValueError(f"Unsupported border_mode: {border_mode!r}")
    filtered = cv2.filter2D(
        gray.astype(np.float64),
        ddepth=cv2.CV_64F,
        kernel=kernel,
        borderType=CV2_BORDER_MAP[border_mode],
    )
    return np.rint(np.clip(filtered, 0.0, 255.0)).astype(np.uint8)


def sliding_window_ghe(
    gray_gauss: np.ndarray,
    window_size: int = 15,
    method: str = "standard_local_he",
    border_mode: str = "reflect",
    zero_division_policy: str = "identity",
    uniform_window_policy: str = "identity",
    chunk_rows: int = 32,
) -> np.ndarray:
    """Steps 9-15 (core): Vectorized row-chunked W x W sliding-window histogram equalization."""
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer >= 3, got {window_size}")
    if method not in VALID_METHODS:
        raise ValueError(f"Unsupported method {method!r}; expected one of {VALID_METHODS}")
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
        # Extract padded strip covering rows [r0, r1) with full W x W vertical context
        strip = padded[r0 : r1 + 2 * pad, :]
        # Zero-copy view: shape (chunk_h, width, W, W) -> (chunk_h, width, W*W)
        windows = sliding_window_view(strip, (window_size, window_size))
        windows_flat = windows.reshape(r1 - r0, width, window_size * window_size)

        centers = gray_gauss[r0:r1, :]
        # Exact cumulative frequency sum_{j=0}^{x_c} f(j) equals count of window pixels <= x_c
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
) -> np.ndarray:
    """Full 15-step GHE Ensemble pipeline returning a (H, W, 3) uint8 image with identical channels.

    Set image_size=None to keep the input spatial dimensions (useful for unit tests).
    """
    if image.ndim == 3:
        if image_size is not None and (image.shape[0] != image_size or image.shape[1] != image_size):
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        if image_size is not None and (image.shape[0] != image_size or image.shape[1] != image_size):
            gray = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        else:
            gray = image.copy()
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")

    kernel = build_gaussian_kernel(kernel_size=gaussian_kernel_size, sigma=gaussian_sigma)
    gray_gauss = apply_gaussian_filter(gray, kernel=kernel, border_mode=border_mode)
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
) -> None:
    """Generate a 2x4 visual + histogram comparison across raw grayscale and all 3 GHE modes."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    resized = (
        cv2.resize(sample_img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        if sample_img.shape[:2] != (image_size, image_size)
        else sample_img
    )
    raw_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized

    panels: list[tuple[str, np.ndarray]] = [("Raw Grayscale", raw_gray)]
    for mode in VALID_METHODS:
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
        panels.append((f"GHE ({mode})", out_3ch[..., 0]))

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), dpi=120)
    for col, (title, img_2d) in enumerate(panels):
        axes[0, col].imshow(img_2d, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].hist(img_2d.ravel(), bins=256, range=(0, 255), color="steelblue", alpha=0.85)
        axes[1, col].set_xlim(0, 255)
        axes[1, col].set_xlabel("Intensity")
        if col == 0:
            axes[1, col].set_ylabel("Pixel Count")
        axes[1, col].grid(alpha=0.25)

    fig.suptitle(
        f"EnPreNiNet GHE Ensemble Comparison (W={window_size}, sigma={gaussian_sigma}, border={border_mode})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(artifact_path), bbox_inches="tight")
    plt.close(fig)


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
) -> dict[str, Any]:
    """Process all splits from images/raw/{train,val,test} into images/<method>/{train,val,test}."""
    ghe_cfg = cfg["preprocessing"]["ghe_ensemble"]
    image_size = int(ghe_cfg["image_size"])
    k_size = int(ghe_cfg["gaussian_kernel_size"])
    sigma = float(ghe_cfg["gaussian_sigma"])

    raw_root = data_root / cfg["paths"]["images_raw_subdir"]
    out_root = data_root / cfg["paths"]["ghe_images_root"] / method

    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"Raw resized images directory not found: {raw_root}. Run src/01_prepare_bdd.py first."
        )

    stats: dict[str, Any] = {
        "method": method,
        "window_size": window_size,
        "output_root": str(out_root),
        "splits": {},
        "artifact": str(artifact_path),
    }

    sample_img: np.ndarray | None = None
    for split_name in ("train", "val", "test"):
        split_in = raw_root / split_name
        split_out = out_root / split_name
        split_out.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(
            p for p in split_in.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ) if split_in.is_dir() else []

        count = 0
        for img_path in img_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {img_path}")
            if sample_img is None:
                sample_img = img

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
            )
            dst_path = split_out / f"{img_path.stem}.png"
            cv2.imwrite(str(dst_path), ghe_img)
            count += 1

        stats["splits"][split_name] = count

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
        )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply 15-step GHE Ensemble preprocessing to night-vision images.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    parser.add_argument("--dummy", action="store_true", help="Run on isolated data/dummy_processed")
    parser.add_argument("--method", type=str, choices=VALID_METHODS, default=None, help="Override GHE mode")
    parser.add_argument("--window-size", type=int, default=None, help="Override sliding window size W")
    parser.add_argument("--border-mode", type=str, default=None, help="Override border padding mode")
    parser.add_argument("--zero-division-policy", type=str, default=None, help="Override H=0 policy")
    parser.add_argument("--uniform-window-policy", type=str, default=None, help="Override uniform window policy")
    parser.add_argument("--chunk-rows", type=int, default=32, help="Number of rows per vectorized chunk")
    parser.add_argument("--data-root", type=Path, default=None, help="Override processed data root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    cfg = load_config(config_path)
    ghe_cfg = cfg["preprocessing"]["ghe_ensemble"]

    method = args.method or str(ghe_cfg["method"])
    window_size = args.window_size if args.window_size is not None else int(ghe_cfg["window_size"])
    border_mode = args.border_mode or str(ghe_cfg["border_mode"])
    zero_div = args.zero_division_policy or str(ghe_cfg["zero_division_policy"])
    uniform_pol = args.uniform_window_policy or str(ghe_cfg["uniform_window_policy"])

    if args.dummy:
        data_root = args.data_root or resolve_path(cfg["paths"]["data_dummy_processed"])
        artifact_path = resolve_path(cfg["paths"]["artifacts_dir"]) / "ghe_comparison_dummy.png"
    else:
        data_root = args.data_root or resolve_path(cfg["paths"]["data_processed"])
        artifact_path = resolve_path(cfg["paths"]["artifacts_dir"]) / f"ghe_comparison_{method}.png"

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
    )
    print(f"[02_ghe_ensemble] Completed ({'dummy' if args.dummy else 'full'})")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
