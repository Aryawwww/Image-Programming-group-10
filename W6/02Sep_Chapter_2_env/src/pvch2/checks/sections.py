"""One check program per PDF §3.x.x."""

from __future__ import annotations

import numpy as np

import cv2

from pvch2.checks.common import CheckResult, check_dir, sample_color_bgr, save_u8
from pvch2.config import load_config
from pvch2.data.inventory import inventory_raw
from pvch2.features.hog import hog_from_bgr
from pvch2.paths import dataset_root, leaf_map_path, raw_dir
from pvch2.preprocess import ops
from pvch2.preprocess.pipelines import run_pipeline


def check_3_1(cfg: dict | None = None) -> CheckResult:
    cfg = cfg or load_config()
    r = CheckResult("3.1", "dataset facts")
    for kind in ("color", "grayscale", "segmented"):
        rec = inventory_raw(kind)
        r.add(rec["n_classes"] == 38, f"{kind} 38 classes", str(rec["n_classes"]))
        r.add(rec["n_images"] >= 54000, f"{kind} full PlantVillage", str(rec["n_images"]))
    r.add(leaf_map_path(cfg).is_file(), "leaf-map.json present")
    return r


def check_3_2(cfg: dict | None = None) -> CheckResult:
    cfg = cfg or load_config()
    r = CheckResult("3.2", "article-structure mapping")
    r.add(cfg["pipelines"]["active"] == ["colour", "grayscale", "segmented"], "three pipelines")
    r.add("preprocess" in cfg and "features" in cfg and "models" in cfg, "2.1 / 2.2 / 2.3-2.4 configs")
    r.add("robustness" in cfg, "2.5 config")
    r.add(cfg["pipelines"]["colour"]["apply"]["leaf_mask"] is False, "colour has no inherited mask")
    r.add(cfg["pipelines"]["segmented"]["apply"]["leaf_mask"] is True, "segmented owns the mask")
    return r


def check_3_3_1(cfg: dict | None = None) -> CheckResult:
    cfg = cfg or load_config()
    r = CheckResult("3.3.1", "geometry and numeric standardisation")
    bgr, _ = sample_color_bgr(cfg)
    size = int(cfg["preprocess"]["geometry"]["size"][0])
    out = ops.resize_square(bgr, size, cfg["preprocess"]["geometry"]["interpolation"])
    f32 = ops.to_f32(out)
    r.add(out.shape[:2] == (size, size), "256×256", str(out.shape))
    r.add(cfg["preprocess"]["geometry"]["interpolation"] == "bicubic", "bicubic")
    r.add(f32.dtype == np.float32, "float32")
    r.add(float(f32.min()) >= 0.0 and float(f32.max()) <= 1.0, "[0,1] range", f"{f32.min():.3f}..{f32.max():.3f}")
    d = check_dir("3.3.1", cfg)
    r.artifacts.append(save_u8(d / "resized.png", out))
    return r


def check_3_3_2(cfg: dict | None = None) -> CheckResult:
    cfg = cfg or load_config()
    r = CheckResult("3.3.2", "colour-space transform")
    bgr, path = sample_color_bgr(cfg)
    bgr = ops.resize_square(bgr, 256)
    lab, hsv = ops.bgr_to_lab_hsv(bgr)
    f32 = ops.to_f32(bgr)
    exg = ops.excess_green(f32)
    exr = ops.excess_red(f32)
    b, g, rr = f32[:, :, 0], f32[:, :, 1], f32[:, :, 2]
    r.add(lab.shape == (*bgr.shape[:2], 3), "Lab 3 channels", str(lab.shape))
    r.add(hsv.shape == (*bgr.shape[:2], 3), "HSV 3 channels", str(hsv.shape))
    r.add(np.allclose(exg, 2 * g - rr - b, atol=1e-5), "ExG = 2G-R-B")
    r.add(np.allclose(exr, 1.4 * rr - g, atol=1e-5), "ExR = 1.4R-G")
    ch_std = float(np.mean([bgr[:, :, i].std() for i in range(3)]))
    r.add(ch_std > 5.0, "colour pipeline is not grayscale", f"mean channel std={ch_std:.1f}")
    r.add(cfg["preprocess"]["color_spaces"]["cie_lab"] and cfg["preprocess"]["color_spaces"]["hsv"], "Lab and HSV enabled")
    colour = run_pipeline("colour", path, cfg)
    r.add(colour.lab is not None and colour.hsv is not None, "colour pipeline stores Lab/HSV")
    gray = run_pipeline("grayscale", path, cfg)
    r.add(gray.lab is None, "grayscale pipeline skips Lab/HSV")
    d = check_dir("3.3.2", cfg)
    r.artifacts += [
        save_u8(d / "lab_L.png", lab[:, :, 0]),
        save_u8(d / "hsv_H.png", hsv[:, :, 0]),
        save_u8(d / "exg.png", exg),
        save_u8(d / "exr.png", exr),
    ]
    return r


def check_3_3_3(cfg: dict | None = None) -> CheckResult:
    cfg = cfg or load_config()
    r = CheckResult("3.3.3", "illumination and contrast")
    bgr, _ = sample_color_bgr(cfg)
    bgr = ops.resize_square(bgr, 256)
    gw = ops.gray_world(bgr)
    lab0, _ = ops.bgr_to_lab_hsv(gw)
    l, a, b = cv2.split(lab0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    clahe_bgr = ops.clahe_L(gw, 2.0, (8, 8))
    r.add(cfg["preprocess"]["illumination"]["clahe"]["channel"] == "L", "CLAHE on L only")
    r.add(tuple(cfg["preprocess"]["illumination"]["clahe"]["tile_grid"]) == (8, 8), "tile 8×8")
    r.add(cfg["preprocess"]["illumination"]["clahe"]["clip_limit"] == 2.0, "clipLimit 2.0")
    r.add(np.array_equal(a, lab0[:, :, 1]) and np.array_equal(b, lab0[:, :, 2]), "a*/b* not equalised")
    r.add(not np.array_equal(l, l2), "L changed by CLAHE")
    r.add(clahe_bgr.shape == bgr.shape, "clahe_L returns BGR")
    homo = ops.homomorphic_L(gw)
    r.add(homo.shape == bgr.shape, "homomorphic callable (optional)")
    d = check_dir("3.3.3", cfg)
    r.artifacts += [save_u8(d / "gray_world.png", gw), save_u8(d / "clahe_L.png", clahe_bgr)]
    return r


def check_3_3_4(cfg: dict | None = None) -> CheckResult:
    cfg = cfg or load_config()
    r = CheckResult("3.3.4", "edge-preserving denoise")
    bgr, path = sample_color_bgr(cfg)
    bgr = ops.resize_square(bgr, 256)
    spec = cfg["preprocess"]["denoise"]["bilateral"]
    bi = ops.bilateral(bgr, spec["d"], spec["sigma_color"], spec["sigma_space"])
    gauss = cv2.GaussianBlur(bgr, (15, 15), 0)
    mean = cv2.blur(bgr, (15, 15))
    med_rgb = cv2.medianBlur(bgr, 3)
    r.add(spec["d"] == 5 and spec["sigma_color"] == 50.0 and spec["sigma_space"] == 50.0, "d=5, σ=50/50")
    r.add(not np.allclose(bi, bgr), "bilateral changes the image")
    r.add(cfg["preprocess"]["denoise"]["gaussian_or_mean_on_rgb"] is False, "no large Gaussian/mean on RGB")
    colour = run_pipeline("colour", path, cfg)
    r.add(not np.allclose(colour.bgr, gauss), "colour pipeline is not 15×15 Gaussian")
    r.add(not np.allclose(colour.bgr, mean), "colour pipeline is not 15×15 mean")
    r.add(cfg["preprocess"]["denoise"]["median_on_binary_mask_only"]["ksize"] == 3, "median 3×3 is mask-only")
    r.add(not np.allclose(colour.bgr, med_rgb), "median is not applied to colour RGB")
    d = check_dir("3.3.4", cfg)
    r.artifacts += [save_u8(d / "bilateral.png", bi), save_u8(d / "forbidden_gaussian15.png", gauss)]
    return r


def check_3_3_5(cfg: dict | None = None) -> CheckResult:
    cfg = cfg or load_config()
    r = CheckResult("3.3.5", "leaf segmentation")
    bgr, path = sample_color_bgr(cfg)
    bgr = ops.resize_square(bgr, 256)
    f32 = ops.to_f32(bgr)
    mask = ops.otsu_mask(ops.excess_green(f32))
    mask = ops.morphology_open_close(mask, 3)
    mask = ops.keep_largest_fill(mask)
    lesion = ops.lesion_roi(bgr, mask)
    masked = ops.apply_mask(bgr, mask, "zero")
    r.add(set(np.unique(mask).tolist()).issubset({0, 255}), "binary mask")
    r.add(int(mask.sum() // 255) > 1000, "leaf area nonempty", str(int(mask.sum() // 255)))
    r.add(int((masked[mask == 0]).sum()) == 0, "background is zero outside mask")
    r.add(cfg["preprocess"]["leaf_mask"]["morphology"]["open_disk_radius"] == 3, "disk radius 3")
    colour = run_pipeline("colour", path, cfg)
    gray = run_pipeline("grayscale", path, cfg)
    seg = run_pipeline("segmented", path, cfg)
    r.add(colour.mask is None, "colour pipeline has no mask")
    r.add(gray.mask is None, "grayscale pipeline has no mask")
    r.add(seg.mask is not None, "segmented pipeline owns the mask")
    r.add(lesion.shape == mask.shape, "lesion ROI is optional and same size")
    d = check_dir("3.3.5", cfg)
    r.artifacts += [
        save_u8(d / "leaf_mask.png", mask),
        save_u8(d / "lesion_roi.png", lesion),
        save_u8(d / "masked.png", masked),
        save_u8(d / "pipeline_segmented.png", seg.bgr),
        save_u8(d / "pipeline_colour.png", colour.bgr),
    ]
    return r


def check_3_3_6(cfg: dict | None = None) -> CheckResult:
    cfg = cfg or load_config()
    r = CheckResult("3.3.6", "edge and gradient maps")
    bgr, _ = sample_color_bgr(cfg)
    bgr = ops.resize_square(bgr, 256)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx, gy, mag = ops.scharr_gradients(gray)
    sx, sy, sm = ops.sobel_gradients(gray)
    canny = ops.canny_adaptive(gray, 20, 40)
    log = ops.laplacian_of_gaussian(gray, 1.0)
    lab, _ = ops.bgr_to_lab_hsv(bgr)
    four = ops.four_channel_lab_grad(lab, mag)
    r.add(gx.shape == gray.shape and gy.shape == gray.shape, "Scharr Gx/Gy")
    r.add(mag.shape == gray.shape and float(mag.max()) > 0, "Scharr magnitude")
    r.add(sx.shape == gray.shape, "Sobel available")
    r.add(set(np.unique(canny).tolist()).issubset({0, 255}), "Canny 20%/40% binary")
    r.add(log.shape == gray.shape, "LoG available")
    r.add(four.shape == (256, 256, 4), "[L,a*,b*,|grad|] four-channel", str(four.shape))
    layout = cfg["preprocess"]["four_channel"]["layout"]
    r.add(layout == ["L", "a", "b", "grad_magnitude"], "four-channel layout", str(layout))
    r.add("HOG" in cfg["pipelines"]["not_in_preprocess"], "HOG is not a 2.1 operator")
    d = check_dir("3.3.6", cfg)
    r.artifacts += [
        save_u8(d / "scharr_gx.png", gx),
        save_u8(d / "scharr_gy.png", gy),
        save_u8(d / "scharr_mag.png", mag),
        save_u8(d / "canny.png", canny),
        save_u8(d / "log.png", log),
        save_u8(d / "four_L.png", four[:, :, 0]),
        save_u8(d / "four_grad.png", four[:, :, 3]),
    ]
    return r


def check_3_3_7(cfg: dict | None = None) -> CheckResult:
    cfg = cfg or load_config()
    r = CheckResult("3.3.7", "operators that must not live in 2.1")
    banned = set(cfg["pipelines"]["not_in_preprocess"])
    r.add("HOG" in banned and "SIFT" in banned, "HOG/SIFT listed out of 2.1")
    r.add("U-Net" in banned, "learned segmentation banned")
    for name in ("colour", "grayscale", "segmented"):
        apply = cfg["pipelines"][name]["apply"]
        r.add("hog" not in apply and "sift" not in apply, f"{name} apply has no HOG/SIFT")
    r.add(cfg["features"]["sift_bovw"]["enabled"] is False, "SIFT BoVW is a 2.2 optional, off")
    return r


def check_3_4(cfg: dict | None = None) -> CheckResult:
    cfg = cfg or load_config()
    r = CheckResult("3.4", "HOG on leaf / lesion ROI")
    bgr, path = sample_color_bgr(cfg)
    bgr = ops.resize_square(bgr, 256)
    f32 = ops.to_f32(bgr)
    leaf = ops.keep_largest_fill(ops.morphology_open_close(ops.otsu_mask(ops.excess_green(f32)), 3))
    lesion = ops.lesion_roi(bgr, leaf)
    hog_cfg = cfg["features"]["hog"]
    r.add(tuple(hog_cfg["pixels_per_cell"]) == (8, 8), "cell 8×8")
    r.add(tuple(hog_cfg["cells_per_block"]) == (2, 2), "block 2×2")
    r.add(int(hog_cfg["orientations"]) == 9, "9 bins")
    hog_full = hog_from_bgr(bgr, None, cfg)
    hog_leaf = hog_from_bgr(bgr, leaf, cfg)
    hog_lesion = hog_from_bgr(bgr, lesion, cfg)
    r.add(hog_full.ndim == 1 and hog_full.size > 100, "full-image HOG vector", str(hog_full.size))
    r.add(hog_leaf.size == hog_full.size, "leaf-ROI HOG same length (fixed 256 canvas)")
    r.add(hog_lesion.size == hog_full.size, "lesion-ROI HOG same length")
    r.add(not np.allclose(hog_full, hog_leaf, atol=1e-5), "leaf ROI changes HOG vs full image")
    r.add("HOG" in cfg["pipelines"]["not_in_preprocess"], "HOG stays in 2.2, not 2.1")
    colour = run_pipeline("colour", path, cfg)
    r.add(colour.mask is None, "HOG-on-ROI does not force a mask onto the colour pipeline")
    d = check_dir("3.4", cfg)
    r.artifacts += [
        save_u8(d / "leaf_roi.png", ops.apply_mask(bgr, leaf, "zero")),
        save_u8(d / "lesion_roi.png", ops.apply_mask(bgr, lesion, "zero")),
    ]
    np.save(d / "hog_leaf.npy", hog_leaf)
    r.artifacts.append(d / "hog_leaf.npy")
    return r


def check_3_5(cfg: dict | None = None) -> CheckResult:
    cfg = cfg or load_config()
    r = CheckResult("3.5", "2.3–2.5 companion settings")
    r.add(cfg["models"]["share_2_1_mask"] is False, "CNN does not inherit a global mask")
    r.add("mobilenet_v3_small" in cfg["models"]["lightweight_cnn"]["name"], "2.3 lightweight CNN named")
    r.add("resnet18" in cfg["models"]["transfer"]["candidates"], "2.4 transfer candidates")
    r.add(cfg["robustness"]["compare_pipelines"] == ["colour", "grayscale", "segmented"], "2.5 three conditions")
    r.add(cfg["robustness"]["split"]["strategy"] == "leaf_id", "leaf-level split")
    return r


def check_3_6(cfg: dict | None = None) -> CheckResult:
    cfg = cfg or load_config()
    r = CheckResult("3.6", "core citations on disk")
    cite = dataset_root(cfg) / cfg["paths"]["citation"]
    r.add(cite.is_file(), "PlantVillage CITATION.cff", str(cite))
    r.add("Mohanty" in cite.read_text(encoding="utf-8"), "Mohanty et al. in citation file")
    return r


def check_3_7(cfg: dict | None = None) -> CheckResult:
    cfg = cfg or load_config()
    r = CheckResult("3.7", "locked main 2.1 six-step config")
    pre = cfg["preprocess"]
    r.add(pre["geometry"]["size"] == [256, 256], "step1 resize 256")
    r.add(pre["color_spaces"]["cie_lab"] and pre["color_spaces"]["hsv"], "step2 Lab+HSV")
    r.add(pre["illumination"]["gray_world"] and pre["illumination"]["clahe"]["enabled"], "step3 gray-world+CLAHE")
    r.add(pre["denoise"]["bilateral"]["enabled"], "step4 bilateral")
    r.add(pre["leaf_mask"]["method"].startswith("otsu"), "step5 Otsu mask (segmented only)")
    r.add(pre["gradient"]["operator"] == "scharr", "step6 Scharr")
    r.add(pre["four_channel"]["layout"] == ["L", "a", "b", "grad_magnitude"], "optional 4-ch")
    return r


REGISTRY = {
    "3.1": check_3_1,
    "3.2": check_3_2,
    "3.3.1": check_3_3_1,
    "3.3.2": check_3_3_2,
    "3.3.3": check_3_3_3,
    "3.3.4": check_3_3_4,
    "3.3.5": check_3_3_5,
    "3.3.6": check_3_3_6,
    "3.3.7": check_3_3_7,
    "3.4": check_3_4,
    "3.5": check_3_5,
    "3.6": check_3_6,
    "3.7": check_3_7,
}
