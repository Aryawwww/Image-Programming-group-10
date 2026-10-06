"""Tests for the auxiliary HSV descriptor."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vein_hog.hsv import (  # noqa: E402
    HSV_DIM,
    anomaly_fraction,
    healthy_hs_box,
    hsv_aux,
    skeleton_hs,
)


def test_hsv_aux_solid_color_one_h_bin():
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[:, :] = (0, 255, 0)  # OpenCV green: H=60, S=255, V=255
    mask = np.full((8, 8), 255, dtype=np.uint8)
    feat = hsv_aux(bgr, mask)

    assert len(feat) == 27
    assert feat.shape == (HSV_DIM,)
    assert feat.dtype == np.float32
    assert int(np.count_nonzero(feat[:16])) == 1
    # Bin width is 180/16; H=60 lands in bin 5. Count is raw, not L2-normalized.
    assert feat[5] == 64
    assert int(np.count_nonzero(feat[16:24])) == 1
    assert feat[23] == 64
    np.testing.assert_allclose(feat[24:], [60, 255, 255])
    assert abs(float(np.linalg.norm(feat)) - 1.0) > 0.1


def test_hsv_aux_empty_mask_is_zero():
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[:, :] = (0, 0, 255)
    feat = hsv_aux(bgr, np.zeros((4, 4), dtype=np.uint8))
    assert feat.shape == (27,)
    assert feat.dtype == np.float32
    assert np.all(feat == 0)


def test_hsv_aux_ignores_pixels_outside_mask():
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[:, :] = (255, 0, 0)  # blue, different hue from the masked pixel
    bgr[0, 0] = (0, 0, 255)  # red
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 1
    feat = hsv_aux(bgr, mask)
    assert int(np.count_nonzero(feat[:16])) == 1
    assert feat[:16].sum() == 1
    np.testing.assert_allclose(feat[24:], [0, 255, 255])


def test_h_bin_covers_top_of_opencv_hue():
    hsv = np.uint8([[[179, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    feat = hsv_aux(bgr, np.ones((1, 1), dtype=np.uint8))
    assert feat[15] == 1
    assert int(np.count_nonzero(feat[:16])) == 1


def test_anomaly_fraction_h_outside_and_inside():
    bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    bgr[0, 0] = (0, 0, 255)
    skeleton = np.ones((1, 1), dtype=bool)
    h, s = skeleton_hs(bgr, skeleton)
    h0, s0 = float(h[0]), float(s[0])

    outside = (h0 + 1.0, h0 + 2.0, s0 - 1.0, s0 + 1.0)
    inside = (h0, h0, s0, s0)
    assert anomaly_fraction(bgr, skeleton, outside) == 1.0
    assert anomaly_fraction(bgr, skeleton, inside) == 0.0

    s_outside = (h0 - 1.0, h0 + 1.0, s0 + 10.0, s0 + 20.0)
    assert anomaly_fraction(bgr, skeleton, s_outside) == 1.0


def test_anomaly_fraction_empty_skeleton_is_zero():
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    skeleton = np.zeros((2, 2), dtype=bool)
    assert anomaly_fraction(bgr, skeleton, (0.0, 1.0, 0.0, 1.0)) == 0.0


def test_anomaly_fraction_is_a_fraction():
    bgr = np.zeros((1, 2, 3), dtype=np.uint8)
    bgr[0, 0] = (0, 0, 255)  # H=0
    bgr[0, 1] = (0, 255, 0)  # H=60
    skeleton = np.ones((1, 2), dtype=bool)
    assert anomaly_fraction(bgr, skeleton, (0.0, 1.0, 0.0, 255.0)) == 0.5


def test_skeleton_hs_matches_true_pixels_only():
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[0, 0] = (0, 0, 255)
    bgr[1, 1] = (255, 0, 0)
    skeleton = np.array([[True, False], [False, True]])
    h, s = skeleton_hs(bgr, skeleton)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    assert np.array_equal(h, [hsv[0, 0, 0], hsv[1, 1, 0]])
    assert np.array_equal(s, [hsv[0, 0, 1], hsv[1, 1, 1]])

    empty_h, empty_s = skeleton_hs(bgr, np.zeros((2, 2), dtype=bool))
    assert empty_h.shape == (0,)
    assert empty_s.shape == (0,)


def test_healthy_hs_box_percentiles_and_empty():
    h = np.linspace(0, 179, 40)
    s = np.linspace(10, 250, 40)
    got = healthy_hs_box(h, s)
    expect = (
        float(np.percentile(h, 5)),
        float(np.percentile(h, 95)),
        float(np.percentile(s, 5)),
        float(np.percentile(s, 95)),
    )
    assert got == expect
    with pytest.raises(ValueError):
        healthy_hs_box([], s)
    with pytest.raises(ValueError):
        healthy_hs_box(h, np.array([]))
