"""Independent pipelines. A mask is not inherited by colour or grayscale."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pvch2.config import load_config
from pvch2.preprocess import ops


def _read_bgr(path: Path, as_gray: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(path)
    if as_gray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _prep_common(bgr: np.ndarray, cfg: dict, apply: dict) -> np.ndarray:
    pre = cfg["preprocess"]
    size = int(pre["geometry"]["size"][0])
    interp = pre["geometry"]["interpolation"]
    if apply.get("resize", True):
        bgr = ops.resize_square(bgr, size, interp)
    if pre["unsharp_mask"]["enabled"]:
        bgr = ops.unsharp(
            bgr, pre["unsharp_mask"]["sigma"], pre["unsharp_mask"]["amount"]
        )
    if apply.get("gray_world") and pre["illumination"]["gray_world"]:
        bgr = ops.gray_world(bgr)
    if apply.get("clahe_L") and pre["illumination"]["clahe"]["enabled"]:
        tile = tuple(pre["illumination"]["clahe"]["tile_grid"])
        bgr = ops.clahe_L(bgr, pre["illumination"]["clahe"]["clip_limit"], tile)
    if apply.get("bilateral") and pre["denoise"]["bilateral"]["enabled"]:
        b = pre["denoise"]["bilateral"]
        bgr = ops.bilateral(bgr, b["d"], b["sigma_color"], b["sigma_space"])
    return bgr


def _leaf_mask(bgr: np.ndarray, cfg: dict) -> np.ndarray:
    pre = cfg["preprocess"]
    method = pre["leaf_mask"]["method"]
    f32 = ops.to_f32(bgr)
    if method == "otsu_a_star":
        lab, _ = ops.bgr_to_lab_hsv(bgr)
        score = lab[:, :, 1].astype(np.float32)
        mask = ops.otsu_mask(score, invert=False)
    else:
        score = ops.excess_green(f32)
        mask = ops.otsu_mask(score, invert=False)
    r = int(pre["leaf_mask"]["morphology"]["open_disk_radius"])
    mask = ops.morphology_open_close(mask, r)
    if pre["denoise"]["median_on_binary_mask_only"]["enabled"]:
        k = int(pre["denoise"]["median_on_binary_mask_only"]["ksize"])
        mask = cv2.medianBlur(mask, k)
    if pre["leaf_mask"]["keep_largest_component"]:
        mask = ops.keep_largest_fill(mask)
    return mask


def run_pipeline(name: str, image_path: Path, cfg: dict | None = None) -> ops.PipelineOutput:
    cfg = cfg or load_config()
    spec = cfg["pipelines"][name]
    apply = spec["apply"]
    as_gray = spec["source"] == "grayscale"
    bgr = _read_bgr(image_path, as_gray=as_gray)
    bgr = _prep_common(bgr, cfg, apply)

    lab = hsv = mask = grad = four = None
    if apply.get("lab_hsv") and cfg["preprocess"]["color_spaces"]["cie_lab"]:
        lab, hsv = ops.bgr_to_lab_hsv(bgr)
    if apply.get("leaf_mask"):
        mask = _leaf_mask(bgr, cfg)
        bgr = ops.apply_mask(bgr, mask, cfg["preprocess"]["leaf_mask"]["background"])
        if lab is not None:
            lab, hsv = ops.bgr_to_lab_hsv(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if apply.get("four_channel") or cfg["preprocess"]["gradient"]["save_magnitude"]:
        grad = ops.scharr_magnitude(gray)
    if apply.get("four_channel"):
        if lab is None:
            lab, hsv = ops.bgr_to_lab_hsv(bgr)
        four = ops.four_channel_lab_grad(lab, grad)

    return ops.PipelineOutput(
        name=name,
        bgr=bgr,
        lab=lab,
        hsv=hsv,
        mask=mask,
        grad=grad,
        four_ch=four,
        meta={
            "source": spec["source"],
            "path": str(image_path),
            "shape": list(bgr.shape),
            "used_mask": bool(apply.get("leaf_mask")),
        },
    )


def available_pipelines(cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_config()
    return list(cfg["pipelines"]["active"])
