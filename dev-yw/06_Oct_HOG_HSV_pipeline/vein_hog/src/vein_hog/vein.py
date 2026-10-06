"""Vein skeleton and uncalibrated completeness. Calibration stays on train."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize


def vein_skeleton(magnitude: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Thin edges inside the mask. Thick blobs are removed before skeletonizing."""
    mag = np.asarray(magnitude, dtype=np.float32)
    inside = np.asarray(mask) > 0
    if not np.any(inside):
        return np.zeros(mag.shape[:2], dtype=bool)
    threshold = float(np.percentile(mag[inside], 90))
    # A tied percentile of 0 would mark flat pixels as edges. Zero gradient is not a vein.
    edges = (mag >= threshold) & inside & (mag > 0)
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thick = cv2.morphologyEx(edges.astype(np.uint8) * 255, cv2.MORPH_OPEN, thick_kernel) > 0
    thin = edges & ~thick
    skeleton = skeletonize(thin)
    return np.asarray(skeleton, dtype=bool) & inside


def raw_completeness(skeleton: np.ndarray, mask: np.ndarray) -> float:
    """Largest 8-connected skeleton length divided by the mask major axis.

    Major axis is the longer side of ``cv2.minAreaRect`` on mask points.
    Fewer than 3 points, or a non-positive side, falls back to ``max(h, w)``.
    An empty mask returns 0. This ratio is not scaled toward 0.8.
    """
    inside = np.asarray(mask) > 0
    if inside.ndim != 2:
        inside = np.squeeze(inside)
        if inside.ndim != 2:
            return 0.0
    if not np.any(inside):
        return 0.0
    component = (np.asarray(skeleton) > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(component, connectivity=8)
    if count <= 1:
        length = 0.0
    else:
        length = float(stats[1:, cv2.CC_STAT_AREA].max())
    major = _major_axis(inside)
    if major <= 0.0:
        return 0.0
    return length / major


def _major_axis(inside: np.ndarray) -> float:
    height, width = inside.shape
    fallback = float(max(height, width))
    ys, xs = np.nonzero(inside)
    if xs.size < 3:
        return fallback
    points = np.stack((xs, ys), axis=1).astype(np.float32)
    try:
        (_, (side_w, side_h), _) = cv2.minAreaRect(points)
    except cv2.error:
        return fallback
    major = float(max(side_w, side_h))
    if major <= 1e-6:
        return fallback
    return major
