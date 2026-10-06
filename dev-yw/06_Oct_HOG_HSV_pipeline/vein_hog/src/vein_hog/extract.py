"""Cache HOG, HSV and vein measurements for train and test only.

Validation rows are skipped here so step 4 cannot read them.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vein_hog.hog import HOG_DIM, hog_descriptor, hog_gradient_magnitude
from vein_hog.hsv import HSV_DIM, hsv_aux, skeleton_hs
from vein_hog.mask import erode_interior, leaf_mask
from vein_hog.vein import raw_completeness, vein_skeleton

IMAGE_SIZE = 256
DEFAULT_RADIUS = 8


def _prepare(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    mask = leaf_mask(bgr)
    mask_ok = bool(mask is not None and int((mask > 0).sum()) > 0)
    bgr256 = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    if mask_ok:
        mask256 = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    else:
        mask256 = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    return bgr256, mask256, mask_ok


def _one(bgr: np.ndarray, radius: int = DEFAULT_RADIUS) -> dict[str, object]:
    bgr256, mask256, mask_ok = _prepare(bgr)
    hog_mask = mask256 if mask_ok else None
    descriptor = np.asarray(hog_descriptor(bgr256, hog_mask), dtype=np.float32).reshape(-1)
    if descriptor.shape != (HOG_DIM,):
        raise RuntimeError(f"HOG dimension {descriptor.shape} != {(HOG_DIM,)}")
    # Gradient on the whole gray image. Zeroing outside the mask paints a
    # false step on the leaf margin and that step wins the 90th percentile.
    gray = cv2.cvtColor(bgr256, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = np.sqrt(np.clip(gray, 0.0, 1.0))
    magnitude = hog_gradient_magnitude(gray)
    if mask_ok:
        interior = erode_interior(mask256, radius)
    else:
        interior = np.zeros_like(mask256)
    skeleton = vein_skeleton(magnitude, interior)
    hue, sat = skeleton_hs(bgr256, skeleton)
    colour = hsv_aux(bgr256, mask256 if mask_ok else np.zeros_like(mask256))
    colour = np.asarray(colour, dtype=np.float32).reshape(-1)
    if colour.shape != (HSV_DIM,):
        raise RuntimeError(f"HSV dimension {colour.shape} != {(HSV_DIM,)}")
    return {
        "hog": descriptor,
        "hsv": colour,
        "raw_completeness": float(raw_completeness(skeleton, mask256)),
        "mask_ok": mask_ok,
        "hue": np.asarray(hue, dtype=np.float32).reshape(-1),
        "sat": np.asarray(sat, dtype=np.float32).reshape(-1),
        "bgr256": bgr256,
        "skeleton": np.asarray(skeleton, dtype=bool),
        "mask256": mask256,
        "interior": interior,
    }


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def extract_split(
    table: pd.DataFrame,
    project_root: Path,
    split: str,
    cache_dir: Path,
    example_dir: Path | None = None,
    *,
    allow_validation: bool = False,
    radius: int = DEFAULT_RADIUS,
) -> None:
    """Extract one split. Validation is refused unless step 5 passes allow_validation."""
    if split not in {"train", "test", "validation"}:
        raise ValueError(f"unknown split: {split}")
    if split == "validation" and not allow_validation:
        raise ValueError("validation is closed until step 5")
    rows = table[table["split"] == split].reset_index(drop=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hog = np.zeros((len(rows), HOG_DIM), dtype=np.float32)
    hsv = np.zeros((len(rows), HSV_DIM), dtype=np.float32)
    raw = np.zeros(len(rows), dtype=np.float32)
    mask_ok = np.zeros(len(rows), dtype=np.uint8)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    hue_parts: list[np.ndarray] = []
    sat_parts: list[np.ndarray] = []
    examples_per_class: dict[str, int] = {}

    for i, row in rows.iterrows():
        path = project_root / str(row["local_relpath"])
        sample = _one(load_bgr(path), radius=radius)
        hog[i] = sample["hog"]
        hsv[i] = sample["hsv"]
        raw[i] = sample["raw_completeness"]
        mask_ok[i] = 1 if sample["mask_ok"] else 0
        hue = sample["hue"]
        sat = sample["sat"]
        hue_parts.append(hue)
        sat_parts.append(sat)
        offsets[i + 1] = offsets[i] + len(hue)
        if example_dir is not None and examples_per_class.get(str(row["class_name"]), 0) < 2:
            _write_example(example_dir, row, sample)
            examples_per_class[str(row["class_name"])] = examples_per_class.get(str(row["class_name"]), 0) + 1
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"{split} {i + 1}/{len(rows)}", flush=True)

    meta = rows[["split", "class_name", "remote_relpath", "sha256", "local_relpath"]].copy()
    meta["raw_completeness"] = raw
    meta["mask_ok"] = mask_ok
    meta.to_csv(cache_dir / f"{split}_meta.csv", index=False)
    np.savez_compressed(
        cache_dir / f"{split}_features.npz",
        hog=hog,
        hsv=hsv,
        raw_completeness=raw,
        mask_ok=mask_ok,
        hue=np.concatenate(hue_parts) if hue_parts else np.zeros(0, dtype=np.float32),
        sat=np.concatenate(sat_parts) if sat_parts else np.zeros(0, dtype=np.float32),
        offsets=offsets,
    )
    summary = {
        "split": split,
        "count": int(len(rows)),
        "mask_failures": int((mask_ok == 0).sum()),
        "hog_dim": int(HOG_DIM),
        "hsv_dim": int(HSV_DIM),
        "interior_radius": int(radius),
    }
    (cache_dir / f"{split}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def skeleton_pixels(cache_path: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path) as data:
        start = int(data["offsets"][index])
        stop = int(data["offsets"][index + 1])
        return data["hue"][start:stop], data["sat"][start:stop]


def _write_example(example_dir: Path, row: pd.Series, sample: dict[str, object]) -> None:
    example_dir.mkdir(parents=True, exist_ok=True)
    bgr = np.asarray(sample["bgr256"])
    skeleton = np.asarray(sample["skeleton"])
    overlay = bgr.copy()
    overlay[skeleton] = (255, 255, 255)
    stem = f"{row['split']}_{row['class_name']}_{Path(str(row['remote_relpath'])).stem}"
    cv2.imwrite(str(example_dir / f"{stem}_skeleton.png"), overlay)
