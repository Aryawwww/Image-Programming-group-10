"""Auxiliary HSV features on a leaf mask. Not a substitute for HOG."""

from __future__ import annotations

import cv2
import numpy as np

# 16 H bins + 8 S bins + mean H, mean S, mean V. Raw counts, not L2-normalized.
HSV_DIM = 27


def hsv_aux(bgr_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """H hist (16, [0, 180)), S hist (8, [0, 256)), then mean H, S, V.

    Only pixels with mask > 0 are counted. An empty mask returns zeros.
    Scaling and the alpha weight are applied later.
    """
    foreground = np.asarray(mask) > 0
    if not np.any(foreground):
        return np.zeros(HSV_DIM, dtype=np.float32)

    hsv = np.ascontiguousarray(cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV))
    mask_u8 = np.ascontiguousarray(foreground, dtype=np.uint8)
    hist_h = cv2.calcHist([hsv], [0], mask_u8, [16], [0, 180]).reshape(-1)
    hist_s = cv2.calcHist([hsv], [1], mask_u8, [8], [0, 256]).reshape(-1)
    means = hsv[foreground].mean(axis=0)
    feat = np.concatenate([hist_h, hist_s, means])
    return np.ascontiguousarray(feat, dtype=np.float32)


def skeleton_hs(
    bgr_u8: np.ndarray, skeleton: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """H and S (float) at True skeleton pixels, in the same pixel order."""
    on_skel = np.asarray(skeleton) > 0
    if not np.any(on_skel):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0][on_skel].astype(np.float64)
    s = hsv[:, :, 1][on_skel].astype(np.float64)
    return h, s


def healthy_hs_box(
    h_values: np.ndarray, s_values: np.ndarray
) -> tuple[float, float, float, float]:
    """(h_low, h_high, s_low, s_high) from the 5th and 95th percentiles.

    Arithmetic percentiles on OpenCV's 0–179 hue line, not a circular mean.
    """
    h = np.asarray(h_values, dtype=np.float64).reshape(-1)
    s = np.asarray(s_values, dtype=np.float64).reshape(-1)
    if h.size == 0 or s.size == 0:
        raise ValueError("h_values and s_values must both be non-empty")
    h_low, h_high = np.percentile(h, [5, 95])
    s_low, s_high = np.percentile(s, [5, 95])
    return float(h_low), float(h_high), float(s_low), float(s_high)


def anomaly_fraction(
    bgr_u8: np.ndarray,
    skeleton: np.ndarray,
    box: tuple[float, float, float, float],
) -> float:
    """Fraction of skeleton pixels whose H or S falls outside the box.

    Bounds are inclusive. Returns 0 when the skeleton has no pixels.
    """
    h, s = skeleton_hs(bgr_u8, skeleton)
    if h.size == 0:
        return 0.0
    h_low, h_high, s_low, s_high = box
    outside = (h < h_low) | (h > h_high) | (s < s_low) | (s > s_high)
    return float(np.mean(outside))
