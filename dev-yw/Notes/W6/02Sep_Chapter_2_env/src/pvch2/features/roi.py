"""ROI helpers for Section 3.4. Mask is a run setting, not a global front-end."""

from __future__ import annotations

import numpy as np


def apply_roi(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero pixels outside the mask. Keeps 256×256 so HOG length stays comparable."""
    m = mask > 0
    if image.ndim == 2:
        out = np.zeros_like(image)
        out[m] = image[m]
        return out
    out = np.zeros_like(image)
    out[m] = image[m]
    return out
