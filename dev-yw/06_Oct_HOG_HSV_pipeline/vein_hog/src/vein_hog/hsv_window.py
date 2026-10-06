"""Sliding-window controller for the HSV auxiliary weight.

Run this before a training job. It reads only the training split. Each
window is one HOG block, 16×16 pixels, stepped by one 8-pixel cell, on the
same 256×256 lattice as the descriptor. The weight written for training is
never below the pre-specified auxiliary 0.25 and never above 1, so the
standardized HSV block cannot drown the HOG block and cannot be turned off.

The sixteen-round report is not replaced. The test split and the validation
split are not opened.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vein_hog.extract import _prepare, load_bgr
from vein_hog.run import find_project

# Same number as the pre-specified report weight. Not chosen from test scores.
HSV_ALPHA_FLOOR = 0.25
HSV_ALPHA_CAP = 1.0
WINDOW = 16
STRIDE = 8
MIN_LEAF_FRACTION = 0.5
CONTRAST_LOW = 10.0
CONTRAST_HIGH = 90.0


def require_hsv_alpha(alpha: float) -> float:
    """Reject a training weight outside [floor, cap]."""
    value = float(alpha)
    if not np.isfinite(value):
        raise ValueError(f"HSV auxiliary weight must be finite, got {alpha}")
    if value < HSV_ALPHA_FLOOR - 1e-12:
        raise ValueError(
            f"HSV auxiliary weight {value} is below the floor {HSV_ALPHA_FLOOR}"
        )
    if value > HSV_ALPHA_CAP + 1e-12:
        raise ValueError(
            f"HSV auxiliary weight {value} is above the cap {HSV_ALPHA_CAP}"
        )
    return value


def control_strength(contrasts: np.ndarray) -> float:
    """Map one contrast per training leaf to a weight in [floor, cap].

    A leaf with no saturation spread keeps the floor. The median leaf, not
    the maximum, sets the weight, so one odd image cannot push it to the cap.
    """
    values = np.asarray(contrasts, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("leaf contrasts must be a non-empty finite array")
    level = float(np.clip(np.median(values), 0.0, 1.0))
    span = HSV_ALPHA_CAP - HSV_ALPHA_FLOOR
    return require_hsv_alpha(HSV_ALPHA_FLOOR + span * level)


def _integral(image: np.ndarray) -> np.ndarray:
    padded = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=np.float64)
    padded[1:, 1:] = np.cumsum(np.cumsum(image, axis=0), axis=1)
    return padded


def _rect(integral: np.ndarray, y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    y2 = y + window
    x2 = x + window
    return integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x]


def window_mean_saturation(
    saturation: np.ndarray,
    leaf: np.ndarray,
    *,
    window: int = WINDOW,
    stride: int = STRIDE,
    min_leaf_fraction: float = MIN_LEAF_FRACTION,
) -> np.ndarray:
    """Mean saturation of leaf pixels in each kept window.

    A window is kept when at least ``min_leaf_fraction`` of its pixels lie
    inside the leaf. Background pixels do not enter the mean.
    """
    sat = np.asarray(saturation, dtype=np.float64)
    inside = np.asarray(leaf) > 0
    if sat.ndim != 2 or sat.shape != inside.shape:
        raise ValueError("saturation and leaf must be 2-D and the same shape")
    if window < 1 or stride < 1:
        raise ValueError(f"window and stride must be positive, got {window}, {stride}")
    height, width = sat.shape
    if window > height or window > width:
        raise ValueError(f"window {window} does not fit a {height}×{width} image")
    weighted = np.where(inside, sat, 0.0)
    ii_s = _integral(weighted)
    ii_c = _integral(inside.astype(np.float64))
    ys = np.arange(0, height - window + 1, stride)
    xs = np.arange(0, width - window + 1, stride)
    y = ys[:, None]
    x = xs[None, :]
    total = _rect(ii_s, y, x, window)
    count = _rect(ii_c, y, x, window)
    keep = count >= (float(min_leaf_fraction) * window * window)
    means = np.zeros(total.shape, dtype=np.float64)
    np.divide(total, count, out=means, where=keep)
    return means[keep]


def leaf_contrast(means: np.ndarray) -> float:
    """(high − low) / 255 of window means. Fewer than two windows is no spread."""
    values = np.asarray(means, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return 0.0
    low, high = np.percentile(values, [CONTRAST_LOW, CONTRAST_HIGH])
    return float(np.clip((float(high) - float(low)) / 255.0, 0.0, 1.0))


def leaf_contrast_from_bgr(bgr: np.ndarray) -> tuple[float, int]:
    """Contrast of one image after the same 256 resize and leaf mask as HOG."""
    bgr256, mask256, mask_ok = _prepare(bgr)
    if not mask_ok:
        return 0.0, 0
    hsv = cv2.cvtColor(bgr256, cv2.COLOR_BGR2HSV)
    means = window_mean_saturation(hsv[:, :, 1], mask256)
    return leaf_contrast(means), int(means.size)


class HSVWindowController:
    """Observes training leaves, then returns the auxiliary weight."""

    def __init__(self) -> None:
        self.contrasts: list[float] = []
        self.n_windows = 0

    def observe(self, bgr: np.ndarray) -> float:
        contrast, count = leaf_contrast_from_bgr(bgr)
        self.contrasts.append(contrast)
        self.n_windows += count
        return contrast

    def strength(self) -> float:
        return control_strength(np.asarray(self.contrasts, dtype=np.float64))


def training_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Training rows only. Test and validation stay in the table and are not used."""
    if "split" not in table.columns:
        raise ValueError("split column is missing")
    rows = table.loc[table["split"] == "train"].reset_index(drop=True)
    if rows.empty:
        raise ValueError("training split is empty")
    return rows


def load_training_alpha(path: Path) -> float:
    """Read the weight written before training and apply the floor again."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("test_was_read") or payload.get("validation_was_read"):
        raise ValueError("HSV controller record must not have read test or validation")
    if payload.get("split") != "train":
        raise ValueError("HSV controller record is not from the training split")
    return require_hsv_alpha(float(payload["hsv_alpha"]))


def _pipeline_root() -> Path:
    return Path(__file__).resolve().parents[3]


def write_strength(path: Path, controller: HSVWindowController, n_train: int) -> dict[str, object]:
    alpha = controller.strength()
    contrasts = np.asarray(controller.contrasts, dtype=np.float64)
    payload: dict[str, object] = {
        "hsv_alpha": alpha,
        "floor": HSV_ALPHA_FLOOR,
        "cap": HSV_ALPHA_CAP,
        "window": WINDOW,
        "stride": STRIDE,
        "min_leaf_fraction": MIN_LEAF_FRACTION,
        "contrast_percentiles": [CONTRAST_LOW, CONTRAST_HIGH],
        "median_leaf_contrast": float(np.median(contrasts)),
        "n_train": int(n_train),
        "n_windows": int(controller.n_windows),
        "split": "train",
        "test_was_read": False,
        "validation_was_read": False,
        "replaces_report_round": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    repo = _pipeline_root()
    project = find_project(repo)
    table = pd.read_csv(repo / "vein_hog" / "data" / "split_map.csv")
    rows = training_frame(table)
    controller = HSVWindowController()
    total = len(rows)
    for index, row in rows.iterrows():
        image = load_bgr(project / str(row["local_relpath"]))
        controller.observe(image)
        done = int(index) + 1
        if done % 250 == 0 or done == total:
            print(f"HSV window {done}/{total}", flush=True)
    out = repo / "outputs" / "hsv_window" / "strength.json"
    payload = write_strength(out, controller, total)
    print(
        f"HSV_ALPHA {payload['hsv_alpha']:.6f} floor={HSV_ALPHA_FLOOR} n={total}",
        flush=True,
    )


if __name__ == "__main__":
    main()
