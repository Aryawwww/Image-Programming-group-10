"""Non-learning matrix operators. Parameters come from configs/preprocess.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

INTERP = {
    "bicubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


def to_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    x = np.clip(img, 0.0, 1.0) if img.max() <= 1.5 else np.clip(img, 0.0, 255.0) / 255.0
    return (x * 255.0 + 0.5).astype(np.uint8)


def to_f32(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    x = img.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def resize_square(bgr: np.ndarray, size: int, interpolation: str = "bicubic") -> np.ndarray:
    flag = INTERP[interpolation]
    if bgr.shape[0] == size and bgr.shape[1] == size:
        return bgr
    return cv2.resize(bgr, (size, size), interpolation=flag)


def gray_world(bgr_u8: np.ndarray) -> np.ndarray:
    x = bgr_u8.astype(np.float32)
    means = x.reshape(-1, 3).mean(axis=0)
    means = np.maximum(means, 1.0)
    gray = float(means.mean())
    balanced = x / means * gray
    return np.clip(balanced, 0, 255).astype(np.uint8)


def clahe_L(bgr_u8: np.ndarray, clip_limit: float, tile: tuple[int, int]) -> np.ndarray:
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def bilateral(bgr_u8: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    return cv2.bilateralFilter(bgr_u8, int(d), float(sigma_color), float(sigma_space))


def unsharp(bgr_u8: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr_u8, (0, 0), sigmaX=sigma)
    sharp = cv2.addWeighted(bgr_u8, 1.0 + amount, blur, -amount, 0)
    return sharp


def bgr_to_lab_hsv(bgr_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    return lab, hsv


def excess_green(bgr_f32: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr_f32)
    return 2.0 * g - r - b


def excess_red(bgr_f32: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr_f32)
    return 1.4 * r - g


def _to_u8_signed(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros(x.shape, dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    return (y * 255.0 + 0.5).astype(np.uint8)


def otsu_mask(score: np.ndarray, invert: bool = False) -> np.ndarray:
    """Threshold a score map. For ExG, high values are vegetation (invert=False)."""
    u8 = _to_u8_signed(score)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        mask = cv2.bitwise_not(mask)
    return mask


def morphology_open_close(mask: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def keep_largest_fill(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    out = np.where(labels == keep, 255, 0).astype(np.uint8)
    inv = cv2.bitwise_not(out)
    n2, lab2, _, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h, w = out.shape
    filled = out.copy()
    for i in range(1, n2):
        ys, xs = np.where(lab2 == i)
        if ys.size == 0:
            continue
        touches_border = (
            ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1
        )
        if not touches_border:
            filled[lab2 == i] = 255
    return filled


def apply_mask(bgr_u8: np.ndarray, mask: np.ndarray, background: str = "zero") -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if background == "mean_gray":
        vis = m.astype(bool)
        if vis.any():
            fill = int(round(bgr_u8[vis].mean()))
        else:
            fill = 0
        canvas = np.full_like(bgr_u8, fill)
    else:
        canvas = np.zeros_like(bgr_u8)
    return np.where(m[..., None] == 1, bgr_u8, canvas)


def scharr_gradients(gray_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Scharr(gray_u8, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray_u8, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return gx, gy, mag


def scharr_magnitude(gray_u8: np.ndarray) -> np.ndarray:
    return scharr_gradients(gray_u8)[2]


def sobel_gradients(gray_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return gx, gy, mag


def canny_adaptive(gray_u8: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    """Hysteresis from leaf-gradient percentiles (PDF §3.3.6)."""
    mag = scharr_magnitude(gray_u8)
    scale = mag.max() + 1e-6
    mag_u8 = np.clip(mag / scale * 255.0, 0, 255).astype(np.uint8)
    lo = float(np.percentile(mag_u8, low_percentile))
    hi = float(np.percentile(mag_u8, high_percentile))
    if hi <= lo:
        hi = min(255.0, lo + 1.0)
    return cv2.Canny(gray_u8, lo, hi)


def laplacian_of_gaussian(gray_u8: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=sigma)
    return cv2.Laplacian(blur, cv2.CV_32F)


def lesion_roi(bgr_u8: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """Non-green pixels inside the leaf (PDF §3.3.5 optional lesion mask)."""
    b, g, r = cv2.split(bgr_u8)
    non_green = ((g < r) | (g < b)).astype(np.uint8) * 255
    return cv2.bitwise_and(non_green, leaf_mask)


def homomorphic_L(bgr_u8: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Optional illumination flattening on log(L+1)."""
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    log_l = np.log1p(l.astype(np.float32))
    low = cv2.GaussianBlur(log_l, (0, 0), sigmaX=sigma)
    high = log_l - low
    l2 = np.clip(np.expm1(high + float(np.mean(low))), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def four_channel_lab_grad(lab_u8: np.ndarray, grad: np.ndarray) -> np.ndarray:
    l, a, b = cv2.split(lab_u8)
    g = grad
    if g.max() > 1.5:
        g = g / (g.max() + 1e-6)
    ch = np.stack(
        [
            l.astype(np.float32) / 255.0,
            a.astype(np.float32) / 255.0,
            b.astype(np.float32) / 255.0,
            g.astype(np.float32),
        ],
        axis=-1,
    )
    return ch


@dataclass
class PipelineOutput:
    name: str
    bgr: np.ndarray
    lab: np.ndarray | None = None
    hsv: np.ndarray | None = None
    mask: np.ndarray | None = None
    grad: np.ndarray | None = None
    four_ch: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
