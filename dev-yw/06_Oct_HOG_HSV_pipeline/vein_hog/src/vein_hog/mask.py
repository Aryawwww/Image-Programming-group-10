"""Leaf mask: ExG, Otsu, open/close radius 3, largest component, holes filled."""

from __future__ import annotations

import cv2
import numpy as np


def leaf_mask(bgr_u8: np.ndarray) -> np.ndarray:
    """uint8 mask, 0 or 255. A failed mask is all zeros and does not raise."""
    try:
        image = np.asarray(bgr_u8)
        if (
            image.ndim != 3
            or image.shape[2] < 3
            or image.shape[0] == 0
            or image.shape[1] == 0
            or image.size == 0
        ):
            return _zero_mask(image)
        # BGR; extra channels (for example alpha) are ignored.
        blue, green, red = cv2.split(image[:, :, :3].astype(np.float32))
        excess_green = 2.0 * green - red - blue
        lo = float(np.min(excess_green))
        hi = float(np.max(excess_green))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
            return np.zeros(excess_green.shape, dtype=np.uint8)
        score = ((excess_green - lo) / (hi - lo) * 255.0).astype(np.uint8)
        _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return _largest_filled(mask)
    except Exception:
        return _zero_mask(bgr_u8)


def _zero_mask(image: np.ndarray | None) -> np.ndarray:
    if image is not None and hasattr(image, "shape") and len(image.shape) >= 2:
        return np.zeros((int(image.shape[0]), int(image.shape[1])), dtype=np.uint8)
    return np.zeros((0, 0), dtype=np.uint8)


def erode_interior(mask: np.ndarray, radius: int) -> np.ndarray:
    """Shrink a 0/255 mask so the leaf margin is outside the vein percentile.

    ``radius`` is in pixels on the mask's own grid. A non-positive radius
    returns the mask unchanged. The kernel width is ``2 * radius + 1``.
    """
    image = np.asarray(mask)
    if radius <= 0 or image.size == 0:
        return image
    width = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    return cv2.erode(image, kernel)


def _largest_filled(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.where(labels == keep, 255, 0).astype(np.uint8)
    inverse = cv2.bitwise_not(out)
    count2, labels2, _, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    height, width = out.shape
    filled = out.copy()
    for label in range(1, count2):
        ys, xs = np.where(labels2 == label)
        if ys.size == 0:
            continue
        touches = ys.min() == 0 or xs.min() == 0 or ys.max() == height - 1 or xs.max() == width - 1
        if not touches:
            filled[labels2 == label] = 255
    return filled
