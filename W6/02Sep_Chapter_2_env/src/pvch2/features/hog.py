"""HOG on a leaf or lesion ROI (PDF §3.4). Not a 2.1 preprocessor."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog as sk_hog

from pvch2.features.roi import apply_roi


def hog_vector(
    gray_u8: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
) -> np.ndarray:
    work = gray_u8 if mask is None else apply_roi(gray_u8, mask)
    feat = sk_hog(
        work,
        orientations=int(orientations),
        pixels_per_cell=tuple(pixels_per_cell),
        cells_per_block=tuple(cells_per_block),
        block_norm="L2-Hys",
        feature_vector=True,
        transform_sqrt=False,
    )
    return np.asarray(feat, dtype=np.float32)


def hog_from_bgr(bgr_u8: np.ndarray, mask: np.ndarray | None = None, cfg: dict | None = None) -> np.ndarray:
    hog_cfg = (cfg or {}).get("features", {}).get("hog", {})
    gray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY)
    return hog_vector(
        gray,
        mask,
        orientations=int(hog_cfg.get("orientations", 9)),
        pixels_per_cell=tuple(hog_cfg.get("pixels_per_cell", [8, 8])),
        cells_per_block=tuple(hog_cfg.get("cells_per_block", [2, 2])),
    )
