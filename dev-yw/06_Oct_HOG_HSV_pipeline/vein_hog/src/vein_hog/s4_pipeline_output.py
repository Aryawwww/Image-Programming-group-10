"""Pipeline output of the frozen S4 winner, for the next pipeline.

Reads ``winner.json`` and does not change it. Pass any number of images or
folders. A folder is copied as a tree under the output: a file
``album/nest/shot.jpg`` is written under ``album/nest/`` there. Files given
by themselves land directly in the output directory. The default directory
is ``outputs/Pipeline_HOG+HSV``. ``--out`` replaces it; a relative ``--out``
is resolved from the pipeline root, not from the shell's current directory.

Any picture is accepted. If the leaf mask is empty, HOG and HSV use the
whole frame instead of stopping. The SVM row layout is unchanged.

Paths may contain spaces. An unquoted Windows command line splits a path
at each space. Those words are joined back when they are one real path, or
when they are only a prefix of a directory name: ``42177 Image`` is not
itself a folder. ``--out`` may name a directory that does not exist yet.
Reading and writing go through memory, so OpenCV never opens the path.

SVM row, saved as ``<stem>_svm_row.npy``, dtype float32, shape ``(1, hog_dim + 27)``.
For this winner that is ``(1, 10827)``. Give it to the SVM as that one row.
Indexes are half-open:

- ``[0:10800]`` HOG. scikit-image ``feature_vector`` in C order, shape
  ``(15, 15, 2, 2, 12)`` before it is flattened. 15 blocks come from a
  256 pixel image, 16-pixel cells and 2×2 blocks. 12 orientations, L2-Hys,
  square-root on, gray outside the leaf set to 0.
- ``[10800:10816]`` hue histogram, 16 bins on ``[0, 180)``, raw counts, mask only.
- ``[10816:10824]`` saturation histogram, 8 bins on ``[0, 256)``, raw counts, mask only.
- ``[10824]`` mean hue, OpenCV 0–179.
- ``[10825]`` mean saturation, 0–255.
- ``[10826]`` mean value, 0–255.

``hsv_alpha`` is not an element of the row. The fitted SVM standardizes the
HOG block and the HSV block separately, then multiplies the HSV block by
this weight (0.45 for the winner). The file is the row before that
standardization. ``A / 255`` in the PNG is only the nearest uint8 to the
weight; the weight to use is the JSON value.

``<stem>_pipeline.png`` is 256×1024, BGRA, no chart text. Each panel stores
the same number in B, G and R. Alpha is 0 off the leaf and
``int(round(hsv_alpha * 255))`` on the leaf. H, S and V are 0 outside
the mask. A HOG stroke that crosses the mask keeps that same alpha.

- x 0–255: HOG star picture on black. Display only. Each cell is the
  orientation star from the same HOG call. Brightness is a gamma on the
  peak so the stars are visible; the row above is not scaled.
- x 256–511: H, OpenCV hue 0–179, not stretched.
- x 512–767: S, 0–255.
- x 768–1023: V, 0–255.

``<stem>_HOG.png`` is the same star picture with no alpha, so a viewer does
not fade it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog as skimage_hog

from vein_hog.extract import IMAGE_SIZE, _prepare
from vein_hog.hsv import HSV_DIM, hsv_aux
from vein_hog.run import find_project
from vein_hog.search_s4 import BLOCK_NORM, CELLS_PER_BLOCK, descriptor_length, hog_vector

# Display only. The SVM row is not passed through this curve.
STAR_GAMMA = 0.35
PIPELINE_ROOT = Path(__file__).resolve().parents[3]
WINNER_JSON = PIPELINE_ROOT / "outputs" / "search_s4" / "winner.json"
DEFAULT_OUT = PIPELINE_ROOT / "outputs" / "Pipeline_HOG+HSV"
IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".jp2",
    ".pbm",
    ".pgm",
    ".ppm",
}


def load_winner(path: Path = WINNER_JSON) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("replaces_report_round") is not False:
        raise RuntimeError("winner.json must not replace report round 10")
    orientations = int(payload["orientations"])
    cell = int(payload["pixels_per_cell"])
    if int(payload["cells_per_block"]) != CELLS_PER_BLOCK or payload["block_norm"] != BLOCK_NORM:
        raise RuntimeError("winner block settings are not the S4 block")
    if int(payload["hog_dim"]) != descriptor_length(orientations, cell):
        raise RuntimeError("winner hog_dim does not match its cell settings")
    if int(payload["hsv_dim"]) != HSV_DIM:
        raise RuntimeError("winner hsv_dim is not 27")
    return payload


def frame(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Resized picture and mask. An empty leaf mask becomes the whole frame."""
    bgr256, mask256, mask_ok = _prepare(bgr)
    if not mask_ok:
        mask256 = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8)
    return bgr256, mask256, mask_ok


def svm_row(bgr: np.ndarray, winner: dict, bgr256: np.ndarray, mask256: np.ndarray) -> np.ndarray:
    """One float32 row: HOG, then the 27 HSV numbers. Alpha stays outside."""
    orientations = int(winner["orientations"])
    cell = int(winner["pixels_per_cell"])
    hog = hog_vector(bgr, orientations, cell)
    colour = hsv_aux(bgr256, mask256)
    if hog.shape != (int(winner["hog_dim"]),) or colour.shape != (HSV_DIM,):
        raise RuntimeError("feature row does not match the winner")
    row = np.concatenate([hog, colour]).astype(np.float32, copy=False)
    return row.reshape(1, -1)


def _gray_for_hog(bgr256: np.ndarray, mask256: np.ndarray) -> np.ndarray:
    image = cv2.resize(bgr256, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    resized = cv2.resize(mask256, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    gray = gray.copy()
    gray[resized == 0] = 0.0
    return gray


def star_picture(bgr: np.ndarray, winner: dict, bgr256: np.ndarray, mask256: np.ndarray) -> np.ndarray:
    """uint8 stars on black. Same HOG call as the row; only the picture is lifted."""
    orientations = int(winner["orientations"])
    cell = int(winner["pixels_per_cell"])
    feat, hog_image = skimage_hog(
        _gray_for_hog(bgr256, mask256),
        orientations=orientations,
        pixels_per_cell=(cell, cell),
        cells_per_block=(CELLS_PER_BLOCK, CELLS_PER_BLOCK),
        block_norm=BLOCK_NORM,
        transform_sqrt=True,
        feature_vector=True,
        visualize=True,
    )
    plain = hog_vector(bgr, orientations, cell)
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    if not np.allclose(feat, plain):
        raise RuntimeError("star picture is not the S4 HOG row")
    ink = np.asarray(hog_image, dtype=np.float64)
    shown = np.zeros(ink.shape, dtype=np.uint8)
    peak = float(ink.max()) if ink.size else 0.0
    if peak > 0.0:
        on = ink > 0
        shown[on] = np.clip((ink[on] / peak) ** STAR_GAMMA * 255.0, 0, 255).astype(np.uint8)
    return shown


def pipeline_png(star: np.ndarray, bgr256: np.ndarray, mask256: np.ndarray, alpha: float) -> np.ndarray:
    """BGRA strip: HOG, H, S, V. Alpha is the winning weight on the leaf."""
    hsv = cv2.cvtColor(bgr256, cv2.COLOR_BGR2HSV)
    hsv[mask256 == 0] = 0
    panels = (star, hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2])
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE * 4, 4), dtype=np.uint8)
    for index, panel in enumerate(panels):
        x0 = index * IMAGE_SIZE
        view = canvas[:, x0 : x0 + IMAGE_SIZE]
        view[:, :, 0] = panel
        view[:, :, 1] = panel
        view[:, :, 2] = panel
    inside = np.zeros((IMAGE_SIZE, IMAGE_SIZE * 4), dtype=bool)
    inside[:, :IMAGE_SIZE] = (mask256 > 0) | (star > 0)
    for index in range(1, 4):
        x0 = index * IMAGE_SIZE
        inside[:, x0 : x0 + IMAGE_SIZE] = mask256 > 0
    canvas[:, :, 3] = np.where(inside, int(round(float(alpha) * 255.0)), 0).astype(np.uint8)
    return canvas


def read_bgr(path: Path) -> np.ndarray:
    """Open an image without giving the path to OpenCV."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise FileNotFoundError(path)
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    """Write a PNG without giving the path to OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    buf.tofile(path)


def _is_flag(token: str) -> bool:
    return token.startswith("-") and token != "-"


def _attached_value(token: str) -> tuple[str, str]:
    """Split ``--out=PATH`` into the prefix and the path body."""
    if token.startswith("--") and "=" in token:
        key, body = token.split("=", 1)
        return f"{key}=", body
    return "", token


def _is_fresh_absolute(token: str) -> bool:
    """True when ``token`` opens its own drive or UNC path."""
    if len(token) >= 2 and token[1] == ":":
        return True
    return token.startswith("\\\\") or token.startswith("//")


def _strict_name_prefix(joined: str) -> bool:
    """True when ``joined`` is a space-boundary prefix of one existing name.

    ``E:\\IT\\Projects\\42177 Image`` is not a directory. The real folder
    continues after that space.
    """
    path = Path(joined)
    parent = path.parent
    head = path.name
    if not head or path.exists() or not parent.is_dir():
        return False
    try:
        children = parent.iterdir()
    except OSError:
        return False
    for child in children:
        name = child.name
        if len(name) > len(head) and name[: len(head)].casefold() == head.casefold() and name[len(head)] == " ":
            return True
    return False


def _missing_out_continues(body: str, nxt: str) -> bool:
    """Join a not-yet-created ``--out`` directory that was split on spaces."""
    if _is_fresh_absolute(nxt):
        return False
    if ("\\" in nxt or "/" in nxt) and Path(nxt).exists():
        return False
    suffix = Path(nxt).suffix.lower()
    if suffix in IMAGE_SUFFIXES and "\\" not in nxt and "/" not in nxt:
        return False
    if Path(body).is_file():
        return False
    joined = Path(f"{body} {nxt}")
    if not joined.parent.exists():
        return False
    if "\\" in nxt or "/" in nxt:
        return True
    return (not Path(body).exists()) and Path(body).parent.exists()


def _can_extend(body: str, nxt: str, allow_missing: bool) -> bool:
    if not body or _is_flag(nxt) or _is_fresh_absolute(nxt):
        return False
    if Path(body).is_file():
        return False
    joined = f"{body} {nxt}"
    if Path(joined).exists() or _strict_name_prefix(joined):
        return True
    if not allow_missing:
        return False
    return _missing_out_continues(body, nxt)


def rejoin_split_paths(args: list[str]) -> list[str]:
    """Put a space-split Windows path back into one argument.

    CommandLineToArgvW splits an unquoted path at every space. A partial
    string such as ``42177 Image`` is not a directory; it is kept when it
    is a prefix of a real name, or when it is the ``--out`` directory this
    script has not created yet.
    """
    joined: list[str] = []
    index = 0
    expect_out = False
    while index < len(args):
        token = args[index]
        prefix, body = _attached_value(token)
        if prefix == "" and _is_flag(token):
            joined.append(token)
            expect_out = token == "--out"
            index += 1
            continue
        allow_missing = expect_out or prefix.startswith("--out=")
        acc = body
        while index + 1 < len(args) and _can_extend(acc, args[index + 1], allow_missing):
            index += 1
            acc = f"{acc} {args[index]}"
        joined.append(f"{prefix}{acc}")
        expect_out = False
        index += 1
    return joined


def resolve_input(path: str, project: Path) -> Path:
    given = Path(path)
    if given.exists():
        return given
    rooted = project / path
    if rooted.exists():
        return rooted
    raise FileNotFoundError(path)


def resolve_out(text: str | None) -> Path:
    if text is None:
        return DEFAULT_OUT
    chosen = Path(text)
    if not chosen.is_absolute():
        chosen = PIPELINE_ROOT / chosen
    return chosen


def image_jobs(source: Path) -> list[tuple[Path, Path]]:
    """Each pair is the image and the directory to keep under the output."""
    if source.is_file():
        return [(source, Path())]
    if not source.is_dir():
        raise FileNotFoundError(source)
    jobs = []
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            jobs.append((path, path.relative_to(source).parent))
    if not jobs:
        raise RuntimeError(f"no images under {source}")
    return jobs


def write_leaf(source: Path, winner: dict, out_dir: Path) -> None:
    bgr = read_bgr(source)
    bgr256, mask256, mask_ok = frame(bgr)
    row = svm_row(bgr, winner, bgr256, mask256)
    star = star_picture(bgr, winner, bgr256, mask256)
    picture = pipeline_png(star, bgr256, mask256, float(winner["hsv_alpha"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    hog_path = out_dir / f"{stem}_HOG.png"
    pipeline_path = out_dir / f"{stem}_pipeline.png"
    row_path = out_dir / f"{stem}_svm_row.npy"
    write_png(hog_path, star)
    write_png(pipeline_path, picture)
    np.save(row_path, row)
    bright = star[star > 0]
    p50 = 0.0 if bright.size == 0 else float(np.median(bright))
    where = "full-frame" if not mask_ok else "leaf-mask"
    print(
        f"{source.name} {where} row {row.shape} star_p50 {p50:.0f} -> {hog_path}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write S4 winner HOG and HSV outputs.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image files or folders. A folder keeps its tree in the output.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: outputs/Pipeline_HOG+HSV",
    )
    args = parser.parse_args(rejoin_split_paths(list(argv if argv is not None else sys.argv[1:])))
    winner = load_winner()
    project = find_project(Path(__file__).resolve())
    out_root = resolve_out(args.out)
    for item in args.inputs:
        source = resolve_input(item, project)
        for image, relative in image_jobs(source):
            write_leaf(image, winner, out_root / relative)


if __name__ == "__main__":
    main()
