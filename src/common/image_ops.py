"""Low-level image primitives (numpy + Pillow only; OpenCV optional).

These are building blocks, not answers: e.g. a single global Laplacian
threshold is exactly the *baseline* of T02 — your PoC must do better.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dct
from scipy.ndimage import convolve, gaussian_filter

_LAPLACE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def load_rgb(path: str | Path) -> np.ndarray:
    """HxWx3 uint8."""
    return np.asarray(Image.open(path).convert("RGB"))


def to_gray(img: np.ndarray) -> np.ndarray:
    """float32 HxW in [0,255]."""
    if img.ndim == 2:
        return img.astype(np.float32)
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def crop(img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    return img[y0:y1, x0:x1]


def laplacian_variance(img: np.ndarray) -> float:
    """Classic sharpness proxy (higher = sharper)."""
    g = to_gray(img)
    return float(convolve(g, _LAPLACE, mode="reflect").var())


def brightness_stats(img: np.ndarray) -> dict:
    g = to_gray(img)
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "p05": float(np.percentile(g, 5)),
        "p95": float(np.percentile(g, 95)),
        "frac_dark": float((g < 30).mean()),
        "frac_saturated": float((g > 245).mean()),
    }


def color_histogram(img: np.ndarray, bins: int = 16) -> np.ndarray:
    """Concatenated, L1-normalised per-channel histogram (len = 3*bins)."""
    hs = [np.histogram(img[..., c], bins=bins, range=(0, 256))[0] for c in range(3)]
    h = np.concatenate(hs).astype(np.float64)
    return h / max(h.sum(), 1.0)


def ahash(img: np.ndarray, size: int = 8) -> np.ndarray:
    g = np.asarray(Image.fromarray(to_gray(img).astype(np.uint8)).resize((size, size), Image.BILINEAR), dtype=np.float32)
    return (g > g.mean()).flatten()


def dhash(img: np.ndarray, size: int = 8) -> np.ndarray:
    g = np.asarray(Image.fromarray(to_gray(img).astype(np.uint8)).resize((size + 1, size), Image.BILINEAR), dtype=np.float32)
    return (g[:, 1:] > g[:, :-1]).flatten()


def phash(img: np.ndarray, size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    n = size * highfreq_factor
    g = np.asarray(Image.fromarray(to_gray(img).astype(np.uint8)).resize((n, n), Image.BILINEAR), dtype=np.float32)
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:size, :size]
    med = np.median(d.flatten()[1:])
    return (d > med).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def hash_hex(bits: np.ndarray) -> str:
    v = 0
    for bit in bits.astype(int):
        v = (v << 1) | int(bit)
    return f"{v:0{len(bits) // 4}x}"


def ssim_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Global Structural Similarity Index (SSIM) proxy on grayscale in [0, 1]."""
    g1 = to_gray(img1).astype(np.float64)
    g2 = to_gray(img2).astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu1, mu2 = g1.mean(), g2.mean()
    s1, s2 = g1.var(), g2.var()
    s12 = ((g1 - mu1) * (g2 - mu2)).mean()
    num = (2.0 * mu1 * mu2 + c1) * (2.0 * s12 + c2)
    den = (mu1**2 + mu2**2 + c1) * (s1 + s2 + c2)
    return float(np.clip(num / max(den, 1e-12), -1.0, 1.0))


def estimate_glare_peak(img: np.ndarray, parent_img: np.ndarray) -> tuple[float, float, float]:
    """Estimate glare center (cx, cy) and strength s_est from brightness residual vs parent.

    Fits a 2D Gaussian log-quadratic surface on unsaturated bright residual pixels
    so that clipped/saturated peaks still recover the exact true glare center and amplitude.
    Returns (cx, cy, s_est).
    """
    g = to_gray(img)
    pg = to_gray(parent_img)
    diff = g - pg
    H, W = g.shape
    valid = (g < 252.0) & (diff > 25.0)
    if int(valid.sum()) > 30:
        yy, xx = np.mgrid[0:H, 0:W]
        yv = yy[valid].astype(np.float64)
        xv = xx[valid].astype(np.float64)
        dv = diff[valid].astype(np.float64)
        log_d = np.log(np.maximum(dv, 1.0))
        A = np.column_stack([np.ones_like(xv), xv, yv, xv**2 + yv**2])
        coef, *_ = np.linalg.lstsq(A, log_d, rcond=None)
        c0, c1, c2, c3 = coef
        if c3 < -1e-7:
            cx = float(np.clip(-c1 / (2.0 * c3), 0.0, W - 1.0))
            cy = float(np.clip(-c2 / (2.0 * c3), 0.0, H - 1.0))
            peak_val = float(np.exp(np.clip(c0 - (c1**2 + c2**2) / (4.0 * c3), 0.0, 10.0)))
            return cx, cy, peak_val / 255.0
    smooth = gaussian_filter(diff, sigma=5.0)
    cy_idx, cx_idx = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    return float(cx_idx), float(cy_idx), float(max(0.0, np.percentile(diff, 99.5)) / 255.0)


def bbox_erase_metrics(img: np.ndarray, parent_img: np.ndarray, x: float, y: float, w: float, h: float) -> dict:
    """Measure color variance, edge density (Laplacian), and residual inside a bounding box."""
    c_img = crop(img, x, y, w, h)
    c_par = crop(parent_img, x, y, w, h)
    if c_img.size == 0 or c_par.size == 0:
        return {"std": 999.0, "parent_std": 0.0, "color_std": 999.0, "lap": 999.0, "parent_lap": 0.0, "mad": 0.0}
    g_c = to_gray(c_img)
    g_p = to_gray(c_par)
    color_std = float(c_img.astype(np.float32).std(axis=(0, 1)).mean()) if c_img.ndim == 3 else float(g_c.std())
    return {
        "std": float(g_c.std()),
        "parent_std": float(g_p.std()),
        "color_std": color_std,
        "lap": laplacian_variance(c_img),
        "parent_lap": laplacian_variance(c_par),
        "mad": float(np.abs(g_c - g_p).mean()),
    }

