"""Raster handoff of the frozen S4 winner for a later image program.

Kept only as a vein picture. Nothing else imports this module, and it
is not an entry point.

Three training leaves, not augmented. Each PNG is 768×256 and has no
chart text. Columns 0–255 are the masked leaf. Columns 256–511 are the
interior gradient the S4 HOG bins: OpenCV hue is the unsigned direction,
value is the magnitude. Columns 512–767 keep leaf hue and saturation and
flatten value. The 10800-d vector is checked, not stored in the PNG.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from vein_hog.extract import IMAGE_SIZE, _prepare, load_bgr
from vein_hog.hsv import hsv_aux
from vein_hog.mask import erode_interior
from vein_hog.run import find_project
from vein_hog.search_s4 import descriptor_length, hog_vector

ORIENTATIONS = 12
PIXELS_PER_CELL = 16
HSV_ALPHA = 0.45
PANEL = IMAGE_SIZE
# Same radius as the vein percentile. The mask step stays outside the picture.
INTERIOR_RADIUS = 8
WINNER_JSON = (
    Path(__file__).resolve().parents[3] / "outputs" / "search_s4" / "winner.json"
)
OUT = Path(__file__).resolve().parents[3] / "outputs" / "S4-winner-sample-of-3"
LEAVES = (
    "dev-yw-main-ref/dataset/train/Tomato_healthy/Tomato_healthy_img_0.jpg",
    "dev-yw-main-ref/dataset/train/Tomato_Late_blight/Tomato_Late_blight_img_1000.jpg",
    "dev-yw-main-ref/dataset/train/Tomato_Septoria_leaf_spot/Tomato_Septoria_leaf_spot_img_1001.jpg",
)


def _require_winner() -> None:
    payload = json.loads(WINNER_JSON.read_text(encoding="utf-8"))
    expected = {
        "winner_round": 99,
        "hog": "ori12-cell16",
        "orientations": ORIENTATIONS,
        "pixels_per_cell": PIXELS_PER_CELL,
        "hsv_alpha": HSV_ALPHA,
        "replaces_report_round": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"winner.json {key}={payload.get(key)!r} is not {value!r}")


def _unsigned_gradient(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centered difference, borders 0. Same stencil the S4 HOG uses."""
    gy = np.zeros(gray.shape, dtype=np.float64)
    gx = np.zeros(gray.shape, dtype=np.float64)
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    return np.hypot(gx, gy), np.mod(np.arctan2(gy, gx), np.pi)


def _hog_panel(bgr256: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pixel orientation inside the leaf. A 16-pixel cell is not a picture.

    Pooled into 16-pixel cells, the twelve bins are nearly equal, so the
    usual glyph is a star and not a vein. This panel keeps the gradient
    before that pooling. Hue is the unsigned direction. Value is the
    magnitude, with the interior 99th percentile drawn as full brightness.
    """
    gray = cv2.cvtColor(bgr256, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    gray = np.sqrt(np.clip(gray, 0.0, 1.0))
    magnitude, angle = _unsigned_gradient(gray)
    interior = erode_interior(mask, INTERIOR_RADIUS) > 0
    positive = magnitude[interior]
    if positive.size == 0:
        raise RuntimeError("HOG picture has no interior pixels")
    scale = float(np.percentile(positive, 99))
    if scale <= 0.0:
        raise RuntimeError("HOG picture scale is zero")
    value = np.zeros(magnitude.shape, dtype=np.uint8)
    value[interior] = np.clip(magnitude[interior] / scale * 255.0, 0, 255).astype(np.uint8)
    if float(np.percentile(value[interior], 90)) < 80.0:
        raise RuntimeError("HOG picture is still too faint to hand off")
    picture = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    picture[..., 0] = np.clip(angle / np.pi * 180.0, 0, 179).astype(np.uint8)
    picture[..., 1] = np.where(interior, 255, 0).astype(np.uint8)
    picture[..., 2] = value
    panel = cv2.cvtColor(picture, cv2.COLOR_HSV2BGR)
    panel[~interior] = 0
    return panel


def _panels(bgr: np.ndarray) -> np.ndarray:
    bgr256, mask256, mask_ok = _prepare(bgr)
    if not mask_ok:
        raise RuntimeError("leaf mask failed")
    feat = hog_vector(bgr, ORIENTATIONS, PIXELS_PER_CELL)
    if feat.shape != (descriptor_length(ORIENTATIONS, PIXELS_PER_CELL),):
        raise RuntimeError("HOG dimension does not match the S4 winner")
    colour = hsv_aux(bgr256, mask256)
    if colour.shape != (27,) or float(np.abs(colour).sum()) == 0.0:
        raise RuntimeError("HSV auxiliary is empty")

    leaf = bgr256.copy()
    leaf[mask256 == 0] = 0

    hog_bgr = _hog_panel(bgr256, mask256)

    hsv = cv2.cvtColor(bgr256, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.where(mask256 > 0, 200, 0).astype(np.uint8)
    hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    hsv_bgr[mask256 == 0] = 0

    canvas = np.zeros((PANEL, PANEL * 3, 3), dtype=np.uint8)
    canvas[:, 0:PANEL] = leaf
    canvas[:, PANEL : PANEL * 2] = hog_bgr
    canvas[:, PANEL * 2 : PANEL * 3] = hsv_bgr
    return canvas


def main() -> None:
    _require_winner()
    project = find_project(Path(__file__).resolve())
    OUT.mkdir(parents=True, exist_ok=True)
    for rel in LEAVES:
        source = project / rel
        stem = source.stem
        canvas = _panels(load_bgr(source))
        dest = OUT / f"{stem}_HOG+HSV.png"
        if not cv2.imwrite(str(dest), canvas):
            raise RuntimeError(f"could not write {dest}")
        print(dest.name, canvas.shape, flush=True)
