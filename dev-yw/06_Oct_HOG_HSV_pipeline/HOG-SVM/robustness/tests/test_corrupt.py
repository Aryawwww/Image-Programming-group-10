"""Endpoint checks for the six robustness corruptions."""

from __future__ import annotations

import numpy as np
import pytest

from corrupt import CATEGORIES, apply_corruption, brightness_factor, gaussian_blur_ksize, gaussian_noise_sigma


def _ramp() -> np.ndarray:
    column = np.linspace(0, 255, 64, dtype=np.uint8)
    plane = np.tile(column, (64, 1))
    return np.stack([plane, plane[::-1], np.full_like(plane, 40)], axis=-1)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left.ravel().astype(np.float64), right.ravel().astype(np.float64))[0, 1])


def test_severity_zero_is_a_copy() -> None:
    image = _ramp()
    for category in CATEGORIES:
        out = apply_corruption(image, category, 0, "abc")
        assert np.array_equal(out, image)
        assert out is not image


def test_same_seed_is_repeatable_and_another_seed_differs() -> None:
    image = _ramp()
    for category in CATEGORIES:
        first = apply_corruption(image, category, 40, "same")
        second = apply_corruption(image, category, 40, "same")
        assert np.array_equal(first, second)
        if category in {"random_noise", "poisson_noise", "multiplicative_noise", "salt_pepper"}:
            other = apply_corruption(image, category, 40, "other")
            assert not np.array_equal(first, other)


def test_salt_pepper_at_100_replaces_every_pixel() -> None:
    out = apply_corruption(_ramp(), "salt_pepper", 100, "pepper")
    assert set(np.unique(out).tolist()) == {0, 255}


def test_random_noise_at_100_drops_the_image() -> None:
    image = _ramp()
    out = apply_corruption(image, "random_noise", 100, "noise")
    assert abs(_correlation(image, out)) < 0.05
    assert not np.array_equal(out, image)


def test_limits_are_flat_or_structureless() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:] = 255
    blurred = apply_corruption(image, "gaussian_blur", 100, "flat")
    filtered = apply_corruption(image, "bilateral", 100, "flat")
    assert len(np.unique(blurred.reshape(-1, 3), axis=0)) == 1
    assert np.array_equal(blurred, filtered)
    poisson = apply_corruption(image, "poisson_noise", 100, "shot")
    assert abs(_correlation(image, poisson)) < 0.2
    multiplicative = apply_corruption(_ramp(), "multiplicative_noise", 100, "speckle")
    assert abs(_correlation(_ramp(), multiplicative)) < 0.05


def test_mid_severity_changes_a_step_edge_and_keeps_dtype() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:] = 255
    blurred = apply_corruption(image, "gaussian_blur", 50, "edge")
    assert blurred.dtype == np.uint8 and blurred.shape == image.shape
    assert int(blurred[16, 14, 0]) > 0
    assert int(blurred[16, 17, 0]) < 255
    for category in CATEGORIES:
        out = apply_corruption(image, category, 50, "mid")
        assert out.dtype == np.uint8 and out.shape == image.shape


def test_high_severity_filters_stay_finite_uint8() -> None:
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    image[::2] = 20
    for category in ("gaussian_blur", "bilateral", "multiplicative_noise", "poisson_noise"):
        out = apply_corruption(image, category, 99, "high")
        assert out.dtype == np.uint8
        assert np.isfinite(out.astype(np.float64)).all()


def test_gaussian_noise_and_brightness_endpoints() -> None:
    image = _ramp()
    for category in ("gaussian_noise", "brightness"):
        copy = apply_corruption(image, category, 0, "abc")
        assert np.array_equal(copy, image)
        assert copy is not image
        first = apply_corruption(image, category, 40, "same")
        second = apply_corruption(image, category, 40, "same")
        assert np.array_equal(first, second)
        assert first.dtype == np.uint8 and first.shape == image.shape
    noisy = apply_corruption(image, "gaussian_noise", 100, "noise")
    assert abs(_correlation(image, noisy)) < 0.2
    other = apply_corruption(image, "gaussian_noise", 40, "other")
    assert not np.array_equal(apply_corruption(image, "gaussian_noise", 40, "same"), other)
    white = apply_corruption(image, "brightness", 100, "bright")
    assert set(np.unique(white).tolist()) == {255}
    mid = apply_corruption(np.zeros((8, 8, 3), dtype=np.uint8), "brightness", 50, "bright")
    assert int(mid[0, 0, 0]) == 128


def test_unit_interval_noise_and_kernel_blur() -> None:
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    copy = gaussian_noise_sigma(image, 0.0, "same")
    assert np.array_equal(copy, image)
    assert copy is not image
    first = gaussian_noise_sigma(image, 0.05, "same")
    assert np.array_equal(first, gaussian_noise_sigma(image, 0.05, "same"))
    assert not np.array_equal(first, gaussian_noise_sigma(image, 0.05, "other"))
    assert not np.array_equal(first, gaussian_noise_sigma(image, 0.10, "same"))
    assert first.dtype == np.uint8 and first.shape == image.shape
    edge = np.zeros((32, 32, 3), dtype=np.uint8)
    edge[:, 16:] = 255
    narrow = gaussian_blur_ksize(edge, 3)
    wide = gaussian_blur_ksize(edge, 5)
    assert narrow.dtype == np.uint8 and narrow.shape == edge.shape
    assert wide.shape == edge.shape
    assert int(narrow[16, 15, 0]) > 0
    assert not np.array_equal(narrow, edge)
    assert not np.array_equal(wide, narrow)
    assert gaussian_blur_ksize(edge, 3) is not edge
    with pytest.raises(ValueError):
        gaussian_blur_ksize(edge, 4)
    with pytest.raises(ValueError):
        gaussian_noise_sigma(image, -0.05, "same")


def test_brightness_factors_scale_and_clip() -> None:
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    copy = brightness_factor(image, 1.0)
    assert np.array_equal(copy, image)
    assert copy is not image
    assert int(brightness_factor(image, 0.7)[0, 0, 0]) == 70
    assert int(brightness_factor(image, 1.3)[0, 0, 0]) == 130
    clipped = np.full((8, 8, 3), 200, dtype=np.uint8)
    assert int(brightness_factor(clipped, 1.3)[0, 0, 0]) == 255
    with pytest.raises(ValueError):
        brightness_factor(image, -0.1)


def test_rejects_bad_arguments() -> None:
    image = _ramp()
    with pytest.raises(ValueError):
        apply_corruption(image, "snow", 10, "abc")
    with pytest.raises(ValueError):
        apply_corruption(image, "random_noise", 101, "abc")
    with pytest.raises(ValueError):
        apply_corruption(image[:, :, 0], "random_noise", 10, "abc")
