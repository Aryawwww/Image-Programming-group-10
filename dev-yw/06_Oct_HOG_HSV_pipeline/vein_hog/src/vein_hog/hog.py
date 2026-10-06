"""Dalal–Triggs cell parameters on a whole 256×256 leaf. No person window."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog as skimage_hog

HOG_DIM = 34596
IMAGE_SIZE = 256


def hog_descriptor(bgr_u8: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """float32 vector of length 34596.

    The image is resized to 256×256 with bicubic interpolation. A mask, if
    given, is resized with nearest neighbour and zeros the gray image outside
    the leaf. Square-root compression is skimage's transform_sqrt.
    """
    bgr = cv2.resize(bgr_u8, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if mask is not None:
        resized = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        gray = gray.copy()
        gray[resized == 0] = 0.0
    feat = skimage_hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    out = np.asarray(feat, dtype=np.float32).reshape(-1)
    if out.shape != (HOG_DIM,):
        raise RuntimeError(f"HOG dimension {out.shape} != {(HOG_DIM,)}")
    return out


def hog_gradient_magnitude(gray_f01: np.ndarray) -> np.ndarray:
    """Centered difference [-1, 0, 1], the same gradient skimage HOG uses.

    ``gray_f01`` is already in [0, 1] and is already square-root compressed
    if the caller applied HOG ``transform_sqrt``. This function does not
    scale or square-root the input. Borders stay zero. Magnitude is
    ``hypot``, float32, with no division by 2.
    """
    image = np.asarray(gray_f01, dtype=np.float32)
    g_row = np.zeros_like(image)
    g_col = np.zeros_like(image)
    g_row[1:-1] = image[2:] - image[:-2]
    g_col[:, 1:-1] = image[:, 2:] - image[:, :-2]
    return np.hypot(g_row, g_col).astype(np.float32, copy=False)
