"""Six image corruptions on a severity scale from 0 to 100."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

CATEGORIES = (
    "random_noise",
    "gaussian_blur",
    "poisson_noise",
    "multiplicative_noise",
    "salt_pepper",
    "bilateral",
)

# The three corruptions the S6 performance record is required to contain.
REQUIRED_CATEGORIES = (
    "gaussian_noise",
    "gaussian_blur",
    "brightness",
)

CATEGORY_EN = {
    "random_noise": "Random noise",
    "gaussian_noise": "Gaussian noise",
    "gaussian_blur": "Gaussian blur",
    "poisson_noise": "Poisson noise",
    "multiplicative_noise": "Multiplicative noise",
    "salt_pepper": "Salt and pepper",
    "bilateral": "Bilateral filter",
    "brightness": "Brightness",
}

CATEGORY_ZH = {
    "random_noise": "随机噪声",
    "gaussian_noise": "高斯噪声",
    "gaussian_blur": "高斯模糊",
    "poisson_noise": "泊松噪声",
    "multiplicative_noise": "乘性噪声",
    "salt_pepper": "椒盐噪声",
    "bilateral": "双边滤波",
    "brightness": "亮度",
}

# sigma = GAUSSIAN_NOISE_SIGMA_SCALE * t/(1-t). Severity 100 ignores the image.
GAUSSIAN_NOISE_SIGMA_SCALE = 25.0

# sigma = GAUSSIAN_SIGMA_SCALE * t / (1 - t), t = severity / 100.
GAUSSIAN_SIGMA_SCALE = 1.5
# lambda = (1 - t) * pixel + t * POISSON_FLAT. Display is the Poisson draw clipped to uint8.
POISSON_FLAT = 8.0
# multiplier = exp(sigma * Z), sigma = MULTIPLICATIVE_SIGMA_SCALE * t / (1 - t).
MULTIPLICATIVE_SIGMA_SCALE = 0.15
# sigmaColor = BILATERAL_COLOR_SCALE * t / (1 - t); sigmaSpace uses the space scale. Diameter caps at 9.
BILATERAL_COLOR_SCALE = 10.0
BILATERAL_SPACE_SCALE = 2.0
BILATERAL_D_MAX = 9

# t = 严重度 / 100。高斯、乘性和双边用 t/(1-t)：低档变化小，靠近 100 时迅速变强。
# 严重度 0 原样返回输入。严重度 100 在各函数里单独处理，使原图内容不可辨认。


def _severity_value(severity: float) -> float:
    try:
        value = float(severity)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"severity must be a number, got {severity!r}") from exc
    if not np.isfinite(value) or value < 0.0 or value > 100.0:
        raise ValueError(f"severity {severity} is outside 0..100")
    if abs(value - round(value)) < 1e-9:
        return float(int(round(value)))
    return value


def _rng(category: str, severity: float, image_sha256: str) -> np.random.Generator:
    token = str(int(severity)) if float(severity).is_integer() else f"{float(severity):.4f}"
    digest = hashlib.sha256(f"{category}|{token}|{image_sha256}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def _accepted(category: str) -> bool:
    return category in CATEGORIES or category in REQUIRED_CATEGORIES


def _check(image: np.ndarray, category: str, severity: float) -> np.ndarray:
    if not _accepted(category):
        raise ValueError(f"unknown corruption: {category}")
    _severity_value(severity)
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected uint8 BGR image, got shape {array.shape} dtype {array.dtype}")
    return np.ascontiguousarray(array)


def _blow(severity: int) -> float:
    """t/(1-t)。只用于 0 < 严重度 < 100；严重度 100 在调用前已经分开。"""
    t = severity / 100.0
    return t / (1.0 - t)


def _mean_image(image: np.ndarray) -> np.ndarray:
    means = np.rint(image.reshape(-1, image.shape[2]).mean(axis=0))
    colour = np.clip(means, 0, 255).astype(np.uint8)
    return np.broadcast_to(colour, image.shape).copy()


def _bilateral_diameter(sigma_space: float) -> int:
    raw = int(round(float(sigma_space) * 2.0))
    if raw < 1:
        raw = 1
    if raw % 2 == 0:
        raw += 1
    return min(raw, BILATERAL_D_MAX)


def _gaussian_noise(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """image + 25 * t/(1-t) * Z. Severity 100 is N(128, 64) and ignores the image."""
    if severity == 100:
        drawn = rng.normal(128.0, 64.0, size=image.shape)
        return np.clip(np.rint(drawn), 0, 255).astype(np.uint8)
    sigma = GAUSSIAN_NOISE_SIGMA_SCALE * _blow(severity)
    mixed = image.astype(np.float64) + sigma * rng.standard_normal(image.shape)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _brightness(image: np.ndarray, severity: int) -> np.ndarray:
    """(1-t)*image + t*255. Severity 100 is white."""
    if severity == 100:
        return np.full(image.shape, 255, dtype=np.uint8)
    t = severity / 100.0
    mixed = (1.0 - t) * image.astype(np.float64) + t * 255.0
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _random_noise(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """(1-t)*原图 + t*均匀噪声。100 时输出与原图像素无关。"""
    if severity == 100:
        return rng.integers(0, 256, size=image.shape, dtype=np.uint8)
    t = severity / 100.0
    noise = rng.integers(0, 256, size=image.shape, dtype=np.uint16)
    mixed = (1.0 - t) * image.astype(np.float64) + t * noise.astype(np.float64)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _gaussian_blur(image: np.ndarray, severity: int) -> np.ndarray:
    """sigma = 1.5 * t/(1-t)。100 时换成各通道均值，即模糊的极限。"""
    if severity == 100:
        return _mean_image(image)
    sigma = GAUSSIAN_SIGMA_SCALE * _blow(severity)
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _poisson(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """λ = (1-t)*像素 + t*8，再裁到 0–255。100 时全图像素同分布，画面接近全黑。"""
    t = severity / 100.0
    lam = (1.0 - t) * image.astype(np.float64) + t * POISSON_FLAT
    drawn = rng.poisson(np.clip(lam, 0.0, None))
    return np.clip(np.rint(drawn), 0, 255).astype(np.uint8)


def _multiplicative(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """原图乘 exp(0.15 * t/(1-t) * Z)。100 时换成整幅均匀噪声。"""
    if severity == 100:
        return rng.integers(0, 256, size=image.shape, dtype=np.uint8)
    sigma = MULTIPLICATIVE_SIGMA_SCALE * _blow(severity)
    factor = np.exp(sigma * rng.standard_normal(image.shape))
    mixed = image.astype(np.float64) * factor
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _salt_pepper(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """每个像素以 t/2 变成 255、t/2 变成 0。100 时每个像素都是 0 或 255。"""
    t = severity / 100.0
    draw = rng.random(image.shape[:2])
    out = image.copy()
    out[draw < (t / 2.0)] = 255
    out[(draw >= (t / 2.0)) & (draw < t)] = 0
    return out


def _bilateral(image: np.ndarray, severity: int) -> np.ndarray:
    """sigmaColor = 10*t/(1-t)，sigmaSpace = 2*t/(1-t)，直径不超过 9。100 时为各通道均值。"""
    if severity == 100:
        return _mean_image(image)
    ratio = _blow(severity)
    sigma_color = BILATERAL_COLOR_SCALE * ratio
    sigma_space = BILATERAL_SPACE_SCALE * ratio
    diameter = _bilateral_diameter(sigma_space)
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def _bgr(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected uint8 BGR image, got shape {array.shape} dtype {array.dtype}")
    return np.ascontiguousarray(array)


def _node_rng(sigma: float, image_sha256: str) -> np.random.Generator:
    """Seed for one test-set noise draw. Separate from the 0..100 severity seed."""
    digest = hashlib.sha256(f"gaussian_noise|{float(sigma):.2f}|{image_sha256}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def gaussian_noise_sigma(image: np.ndarray, sigma: float, image_sha256: str) -> np.ndarray:
    """Add N(0, sigma) on pixels scaled to [0, 1], then return uint8.

    sigma is the standard deviation on that unit interval. 0 copies the input.
    This is not the severity curve in _gaussian_noise.
    """
    source = _bgr(image)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"sigma {sigma} is not a non-negative finite number")
    if sigma == 0.0:
        return source.copy()
    scaled = source.astype(np.float64) / 255.0
    noisy = scaled + _node_rng(sigma, str(image_sha256)).normal(0.0, sigma, size=source.shape)
    return np.clip(np.rint(np.clip(noisy, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def opencv_blur_sigma(ksize: int) -> float:
    """Sigma OpenCV uses when sigmaX is 0 and the kernel side is ksize."""
    return 0.3 * ((int(ksize) - 1) * 0.5 - 1.0) + 0.8


def gaussian_blur_ksize(image: np.ndarray, ksize: int) -> np.ndarray:
    """Gaussian blur with an odd kernel. sigmaX=0 lets OpenCV set sigma from ksize."""
    source = _bgr(image)
    side = int(ksize)
    if side not in (3, 5):
        raise ValueError(f"kernel must be 3 or 5, got {ksize}")
    return cv2.GaussianBlur(source, (side, side), sigmaX=0)


def brightness_factor(image: np.ndarray, factor: float) -> np.ndarray:
    """Multiply BGR pixels by factor and clip to uint8. 1.0 copies the input.

    This matches PIL ImageEnhance.Brightness, which blends the image with black.
    It is not the severity curve in _brightness.
    """
    source = _bgr(image)
    factor = float(factor)
    if not np.isfinite(factor) or factor < 0.0:
        raise ValueError(f"brightness factor {factor} is not a non-negative finite number")
    if factor == 1.0:
        return source.copy()
    return np.clip(np.rint(source.astype(np.float64) * factor), 0, 255).astype(np.uint8)


def apply_corruption(image: np.ndarray, category: str, severity: float, image_sha256: str) -> np.ndarray:
    """Return a uint8 BGR image. Severity 0 is a copy of the input."""
    source = _check(image, category, severity)
    level = _severity_value(severity)
    if level == 0:
        return source.copy()
    rng = _rng(category, level, str(image_sha256))
    if category == "gaussian_noise":
        return _gaussian_noise(source, int(level), rng)
    if category == "brightness":
        return _brightness(source, int(level))
    if category == "random_noise":
        return _random_noise(source, level, rng)
    if category == "gaussian_blur":
        return _gaussian_blur(source, level)
    if category == "poisson_noise":
        return _poisson(source, level, rng)
    if category == "multiplicative_noise":
        return _multiplicative(source, level, rng)
    if category == "salt_pepper":
        return _salt_pepper(source, level, rng)
    return _bilateral(source, level)


def parameter_record() -> dict[str, object]:
    """The severity map, written beside the scores so a figure can be read later."""
    return {
        "severity": "integer 0..100, step 1. 0 copies the input. 100 has no recognizable original image.",
        "seed": "sha256(category|severity|image_sha256), first 8 bytes, little-endian",
        "gaussian_noise": (
            f"image + {GAUSSIAN_NOISE_SIGMA_SCALE} * t/(1-t) * Z; "
            "t=1 is N(128, 64) and ignores the image"
        ),
        "brightness": "(1-t)*image + t*255; t=1 is white",
        "random_noise": "(1-t)*image + t*U(0,255); t=1 is U only",
        "gaussian_blur": f"GaussianBlur sigma = {GAUSSIAN_SIGMA_SCALE} * t/(1-t); t=1 is the per-channel mean",
        "poisson_noise": f"Poisson((1-t)*pixel + t*{POISSON_FLAT}), clipped to uint8",
        "multiplicative_noise": f"image * exp({MULTIPLICATIVE_SIGMA_SCALE} * t/(1-t) * Z); t=1 is U(0,255)",
        "salt_pepper": "each pixel becomes 255 or 0 with probability t/2 each; t=1 replaces every pixel",
        "bilateral": (
            f"bilateralFilter sigmaColor = {BILATERAL_COLOR_SCALE} * t/(1-t), "
            f"sigmaSpace = {BILATERAL_SPACE_SCALE} * t/(1-t), diameter capped at {BILATERAL_D_MAX}; "
            "t=1 is the per-channel mean"
        ),
    }
