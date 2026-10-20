"""Search S6: HOG geometries times HSV channel layouts on the training split.

HOG stays the descriptor. HSV stays an auxiliary block, scaled on the
training rows and multiplied by a weight. Layouts choose which of H, S, and
V enter that block. Weights stay inside [0.25, 0.50]. C stays 1. This is
not a seventeenth cell of the published C × α grid.

The winner is the highest minimum Healthy-versus-one-disease AUC on the
same training holdout as S1 and S4. Test and validation images are not read.
The frozen S4 winner and the report round are not replaced.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from vein_hog.extract import _prepare, load_bgr
from vein_hog.hog import HOG_DIM, IMAGE_SIZE
from vein_hog.hsv import hsv_aux
from vein_hog.hsv_window import HSV_ALPHA_FLOOR, require_hsv_alpha
from vein_hog.rounds import SEED, TRAIN_FRACTIONS, sgd_alpha
from vein_hog.run import find_project
from vein_hog.search_s1 import (
    C_VALUE,
    DISEASES,
    HEALTHY,
    SELECT_FRACTION,
    _nested_parts,
    _take_fraction,
    class_aucs,
)
from vein_hog.search_s4 import (
    BLOCK_NORM,
    CELLS_PER_BLOCK,
    SGD_MAX_ITER,
    SGD_TOL,
    descriptor_length,
    hog_vector,
)

S6_ALPHA_MAX = 0.50
S6_ALPHAS = (0.25, 0.40, 0.45, 0.50)
S6_WORKERS = 4
TRAIN_ROWS = 3000
FORBIDDEN_SEGMENTS = frozenset({"test", "validation"})
COMPONENT_NAMES = ("h_hist", "h_mean", "s_hist", "s_mean", "v_hist", "v_mean")
COMPONENT_WIDTH = {
    "h_hist": 16,
    "h_mean": 1,
    "s_hist": 8,
    "s_mean": 1,
    "v_hist": 8,
    "v_mean": 1,
}
LAYOUTS: dict[str, tuple[str, ...]] = {
    "published": ("h_hist", "s_hist", "h_mean", "s_mean", "v_mean"),
    "H": ("h_hist", "h_mean"),
    "S": ("s_hist", "s_mean"),
    "V": ("v_hist", "v_mean"),
    "HS": ("h_hist", "s_hist", "h_mean", "s_mean"),
    "HV": ("h_hist", "v_hist", "h_mean", "v_mean"),
    "SV": ("s_hist", "v_hist", "s_mean", "v_mean"),
    "HSV": ("h_hist", "s_hist", "v_hist", "h_mean", "s_mean", "v_mean"),
}
HOG_SPECS = tuple(
    {"hog": f"ori{orientations}-cell{cell}", "orientations": orientations, "pixels_per_cell": cell}
    for orientations in (6, 9, 12, 18)
    for cell in (8, 16, 32)
)


def layout_dim(layout: str) -> int:
    parts = LAYOUTS[layout]
    return int(sum(COMPONENT_WIDTH[name] for name in parts))


def hsv_strengths() -> tuple[float, ...]:
    values = tuple(require_hsv_alpha(value) for value in S6_ALPHAS)
    if values[0] != HSV_ALPHA_FLOOR or values[-1] != S6_ALPHA_MAX:
        raise RuntimeError(f"HSV weights must run from 0.25 to 0.50, got {values}")
    if any(value > S6_ALPHA_MAX + 1e-12 for value in values):
        raise RuntimeError("an HSV weight is above 0.50")
    return values


def iter_s6(reference: float) -> list[dict[str, float | int | str]]:
    reference = require_hsv_alpha(reference)
    if reference > S6_ALPHA_MAX + 1e-12:
        raise ValueError(f"HSV reference {reference} is above the S6 cap {S6_ALPHA_MAX}")
    strengths = hsv_strengths()
    rounds: list[dict[str, float | int | str]] = []
    number = 1
    for spec in HOG_SPECS:
        length = descriptor_length(int(spec["orientations"]), int(spec["pixels_per_cell"]))
        for layout in LAYOUTS:
            width = layout_dim(layout)
            for alpha in strengths:
                rounds.append(
                    {
                        "round": number,
                        "hog": str(spec["hog"]),
                        "orientations": int(spec["orientations"]),
                        "pixels_per_cell": int(spec["pixels_per_cell"]),
                        "cells_per_block": CELLS_PER_BLOCK,
                        "hog_dim": length,
                        "hsv_layout": layout,
                        "hsv_dim": width,
                        "hsv_alpha": float(alpha),
                    }
                )
                number += 1
    expected = len(HOG_SPECS) * len(LAYOUTS) * len(strengths)
    if expected != 384 or len(rounds) != expected:
        raise RuntimeError(f"expected 384 rounds, got {len(rounds)}")
    return rounds


def assert_train_paths(paths: list[str]) -> None:
    """Refuse a path that is not dataset/train, including any test or validation segment."""
    if len(paths) != TRAIN_ROWS:
        raise RuntimeError(f"expected {TRAIN_ROWS} training images, got {len(paths)}")
    for raw in paths:
        parts = [part.lower() for part in Path(raw).parts]
        if any(part in FORBIDDEN_SEGMENTS for part in parts):
            raise RuntimeError(f"training path enters a held-out split: {raw}")
        if "dataset" not in parts:
            raise RuntimeError(f"training path is not under dataset/train: {raw}")
        dataset_at = parts.index("dataset")
        if dataset_at + 1 >= len(parts) or parts[dataset_at + 1] != "train":
            raise RuntimeError(f"training path is not dataset/train: {raw}")


def channel_blocks(bgr_u8: np.ndarray) -> dict[str, np.ndarray]:
    """Mask-only H, S, and V pieces. The published block matches hsv_aux."""
    bgr256, mask256, mask_ok = _prepare(bgr_u8)
    mask = mask256 if mask_ok else np.zeros_like(mask256)
    published = np.asarray(hsv_aux(bgr256, mask), dtype=np.float32).reshape(-1)
    empty = {name: np.zeros(width, dtype=np.float32) for name, width in COMPONENT_WIDTH.items()}
    if not np.any(mask > 0):
        if published.shape != (layout_dim("published"),) or np.any(published):
            raise RuntimeError("empty-mask HSV is not the published zero block")
        return empty

    hsv = np.ascontiguousarray(cv2.cvtColor(bgr256, cv2.COLOR_BGR2HSV))
    mask_u8 = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    hist_h = cv2.calcHist([hsv], [0], mask_u8, [16], [0, 180]).reshape(-1)
    hist_s = cv2.calcHist([hsv], [1], mask_u8, [8], [0, 256]).reshape(-1)
    hist_v = cv2.calcHist([hsv], [2], mask_u8, [8], [0, 256]).reshape(-1)
    means = hsv[mask_u8 > 0].mean(axis=0)
    rebuilt = np.concatenate([hist_h, hist_s, means]).astype(np.float32, copy=False)
    if rebuilt.shape != published.shape or not np.allclose(rebuilt, published):
        raise RuntimeError("channel blocks do not rebuild the published HSV vector")
    return {
        "h_hist": np.asarray(hist_h, dtype=np.float32),
        "h_mean": np.asarray([means[0]], dtype=np.float32),
        "s_hist": np.asarray(hist_s, dtype=np.float32),
        "s_mean": np.asarray([means[1]], dtype=np.float32),
        "v_hist": np.asarray(hist_v, dtype=np.float32),
        "v_mean": np.asarray([means[2]], dtype=np.float32),
    }


def assemble_channels(blocks: dict[str, np.ndarray], layout: str) -> np.ndarray:
    parts = [np.asarray(blocks[name], dtype=np.float32).reshape(-1) for name in LAYOUTS[layout]]
    out = np.concatenate(parts).astype(np.float32, copy=False)
    if out.shape != (layout_dim(layout),):
        raise RuntimeError(f"{layout} width {out.shape} != {layout_dim(layout)}")
    return out


def featurize_leaf(bgr_u8: np.ndarray, orientations: int, pixels_per_cell: int, layout: str) -> tuple[np.ndarray, np.ndarray]:
    """One HOG row and one assembled HSV row. The caller chooses the image."""
    hog = hog_vector(bgr_u8, int(orientations), int(pixels_per_cell))
    colour = assemble_channels(channel_blocks(bgr_u8), layout)
    return hog, colour


def _pipeline_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _paths(meta: pd.DataFrame, project: Path) -> list[str]:
    return [str(project / str(row)) for row in meta["local_relpath"].tolist()]


def _load_train_meta(cache_dir: Path, project: Path) -> tuple[pd.DataFrame, list[str]]:
    meta = pd.read_csv(cache_dir / "train_meta.csv")
    if set(meta["split"]) != {"train"}:
        raise RuntimeError("S6 must not read validation rows")
    paths = _paths(meta, project)
    assert_train_paths(paths)
    return meta, paths


def _hog_job(job: tuple[str, int, int]) -> np.ndarray:
    path, orientations, pixels_per_cell = job
    return hog_vector(load_bgr(Path(path)), orientations, pixels_per_cell)


def _compute_hog(paths: list[str], orientations: int, pixels_per_cell: int) -> np.ndarray:
    assert_train_paths(paths)
    jobs = [(path, int(orientations), int(pixels_per_cell)) for path in paths]
    width = descriptor_length(orientations, pixels_per_cell)
    out = np.empty((len(jobs), width), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=S6_WORKERS) as pool:
        for index, vector in enumerate(pool.map(_hog_job, jobs, chunksize=8)):
            out[index] = vector
            done = index + 1
            if done % 250 == 0 or done == len(jobs):
                print(f"S6 hog ori{orientations}-cell{pixels_per_cell} {done}/{len(jobs)}", flush=True)
    return out


def _published_train_hog(cache_dir: Path) -> np.ndarray:
    path = cache_dir / "train_features.npz"
    if path.name != "train_features.npz":
        raise RuntimeError("only the training feature cache may be reused")
    with np.load(path) as data:
        hog = np.asarray(data["hog"], dtype=np.float32)
    if hog.shape != (TRAIN_ROWS, HOG_DIM):
        raise RuntimeError(f"cached training HOG shape {hog.shape} != {(TRAIN_ROWS, HOG_DIM)}")
    if hog.shape[1] != descriptor_length(9, 8):
        raise RuntimeError("cached training HOG is not ori9-cell8")
    return hog


def _load_npy(path: Path, shape: tuple[int, ...]) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        array = np.load(path)
    except (OSError, ValueError):
        return None
    if array.shape != shape or array.dtype != np.float32:
        return None
    return np.asarray(array, dtype=np.float32)


def ensure_hog(spec: dict[str, float | int | str], paths: list[str], cache_dir: Path, npy_dir: Path) -> np.ndarray:
    name = str(spec["hog"])
    width = int(spec["hog_dim"])
    path = npy_dir / f"hog_{name}.npy"
    cached = _load_npy(path, (TRAIN_ROWS, width))
    if cached is not None:
        print(f"S6 reuse {path.name} {cached.shape}", flush=True)
        return cached
    if int(spec["orientations"]) == 9 and int(spec["pixels_per_cell"]) == 8:
        hog = _published_train_hog(cache_dir)
    else:
        hog = _compute_hog(paths, int(spec["orientations"]), int(spec["pixels_per_cell"]))
    if hog.shape != (TRAIN_ROWS, width):
        raise RuntimeError(f"{name} HOG shape {hog.shape} != {(TRAIN_ROWS, width)}")
    npy_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, hog)
    print(f"S6 wrote {path.name} {hog.shape}", flush=True)
    return hog


def _hsv_job(path: str) -> dict[str, np.ndarray]:
    return channel_blocks(load_bgr(Path(path)))


def ensure_hsv(paths: list[str], npy_dir: Path) -> dict[str, np.ndarray]:
    assert_train_paths(paths)
    npy_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for name, width in COMPONENT_WIDTH.items():
        cached = _load_npy(npy_dir / f"{name}.npy", (TRAIN_ROWS, width))
        if cached is None:
            missing.append(name)
        else:
            found[name] = cached
    if not missing:
        print("S6 reuse HSV channel npy", flush=True)
        return found
    columns = {name: np.empty((TRAIN_ROWS, COMPONENT_WIDTH[name]), dtype=np.float32) for name in COMPONENT_NAMES}
    with ProcessPoolExecutor(max_workers=S6_WORKERS) as pool:
        for index, blocks in enumerate(pool.map(_hsv_job, paths, chunksize=8)):
            for name in COMPONENT_NAMES:
                columns[name][index] = blocks[name]
            done = index + 1
            if done % 250 == 0 or done == len(paths):
                print(f"S6 hsv channels {done}/{len(paths)}", flush=True)
    for name, values in columns.items():
        np.save(npy_dir / f"{name}.npy", values)
        print(f"S6 wrote {name}.npy {values.shape}", flush=True)
    return columns


def assemble_matrix(components: dict[str, np.ndarray], layout: str) -> np.ndarray:
    parts = [components[name] for name in LAYOUTS[layout]]
    out = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if out.shape != (TRAIN_ROWS, layout_dim(layout)):
        raise RuntimeError(f"{layout} matrix {out.shape} != {(TRAIN_ROWS, layout_dim(layout))}")
    return out


def fit_classifier(
    hog: np.ndarray,
    hsv: np.ndarray,
    labels: np.ndarray,
    fit_index: np.ndarray,
    hsv_alpha: float,
) -> tuple[SGDClassifier, StandardScaler, StandardScaler]:
    """Linear SVM. HOG and HSV are scaled on the fit rows; HSV then carries the weight."""
    alpha = require_hsv_alpha(hsv_alpha)
    if alpha > S6_ALPHA_MAX + 1e-12:
        raise ValueError(f"HSV weight {alpha} is above the S6 cap {S6_ALPHA_MAX}")
    hog_scaler = StandardScaler().fit(hog[fit_index])
    hsv_scaler = StandardScaler().fit(hsv[fit_index])
    design = np.hstack(
        [hog_scaler.transform(hog[fit_index]), hsv_scaler.transform(hsv[fit_index]) * alpha]
    ).astype(np.float32, copy=False)
    clf = SGDClassifier(
        loss="hinge",
        alpha=sgd_alpha(C_VALUE, len(fit_index)),
        class_weight="balanced",
        random_state=SEED,
        max_iter=SGD_MAX_ITER,
        tol=SGD_TOL,
    )
    clf.fit(design, labels[fit_index])
    return clf, hog_scaler, hsv_scaler


def healthy_scores(
    hog: np.ndarray,
    hsv: np.ndarray,
    labels: np.ndarray,
    fit_index: np.ndarray,
    score_indices: list[np.ndarray],
    hsv_alpha: float,
) -> list[np.ndarray]:
    clf, hog_scaler, hsv_scaler = fit_classifier(hog, hsv, labels, fit_index, hsv_alpha)
    classes = [str(item) for item in clf.classes_]
    if HEALTHY not in classes:
        raise RuntimeError("healthy class missing from the classifier")
    column = classes.index(HEALTHY)
    alpha = require_hsv_alpha(hsv_alpha)
    found: list[np.ndarray] = []
    for index in score_indices:
        design = np.hstack(
            [hog_scaler.transform(hog[index]), hsv_scaler.transform(hsv[index]) * alpha]
        ).astype(np.float32, copy=False)
        scores = np.asarray(clf.decision_function(design), dtype=np.float64)
        found.append(scores[:, column])
    return found


def _record(spec: dict[str, float | int | str], fraction: float, subset: str, labels: np.ndarray, score: np.ndarray) -> dict[str, float | int | str]:
    if subset not in {"fit", "select"}:
        raise RuntimeError(f"S6 record subset must be fit or select, got {subset}")
    aucs = class_aucs(labels, score)
    values = [aucs[name] for name in DISEASES]
    row: dict[str, float | int | str] = {
        "round": int(spec["round"]),
        "hog": str(spec["hog"]),
        "orientations": int(spec["orientations"]),
        "pixels_per_cell": int(spec["pixels_per_cell"]),
        "cells_per_block": int(spec["cells_per_block"]),
        "hog_dim": int(spec["hog_dim"]),
        "hsv_layout": str(spec["hsv_layout"]),
        "hsv_dim": int(spec["hsv_dim"]),
        "hsv_alpha": float(spec["hsv_alpha"]),
        "C": C_VALUE,
        "fraction": float(fraction),
        "subset": subset,
        "min_class_auc": float(min(values)),
        "mean_class_auc": float(np.mean(values)),
        "n": int(len(labels)),
    }
    for name in DISEASES:
        row[f"auc_{name}"] = aucs[name]
    return row


def choose_winner(table: pd.DataFrame, reference: float) -> int:
    """Highest select-subset min AUC at fraction 1. Fewer features win a tie."""
    reference = require_hsv_alpha(reference)
    if reference > S6_ALPHA_MAX + 1e-12:
        raise ValueError(f"HSV reference {reference} is above the S6 cap {S6_ALPHA_MAX}")
    if "subset" not in table.columns or not set(table["subset"]).issubset({"fit", "select"}):
        raise RuntimeError("winner is chosen before the test or validation split is scored")
    alphas = table["hsv_alpha"].astype(float)
    if (alphas < HSV_ALPHA_FLOOR - 1e-12).any() or (alphas > S6_ALPHA_MAX + 1e-12).any():
        raise RuntimeError("a round is outside the HSV interval [0.25, 0.50]")
    full = table[np.isclose(table["fraction"].astype(float), 1.0) & (table["subset"] == "select")].copy()
    expected = len(HOG_SPECS) * len(LAYOUTS) * len(hsv_strengths())
    if len(full) != expected:
        raise RuntimeError(f"expected {expected} select rows at fraction 1, got {len(full)}")
    full["near_reference"] = (full["hsv_alpha"].astype(float) - reference).abs()
    full = full.sort_values(
        ["min_class_auc", "mean_class_auc", "hog_dim", "hsv_dim", "near_reference", "round"],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
    )
    return int(full.iloc[0]["round"])


def _eval_spec(
    spec: dict[str, float | int | str],
    hog: np.ndarray,
    hsv: np.ndarray,
    labels: np.ndarray,
    select_index: np.ndarray,
    parts: list[np.ndarray],
) -> list[dict[str, float | int | str]]:
    local: list[dict[str, float | int | str]] = []
    for fraction in TRAIN_FRACTIONS:
        used = _take_fraction(parts, float(fraction))
        select_score, fit_score = healthy_scores(
            hog, hsv, labels, used, [select_index, used], float(spec["hsv_alpha"])
        )
        local.append(_record(spec, float(fraction), "select", labels[select_index], select_score))
        local.append(_record(spec, float(fraction), "fit", labels[used], fit_score))
    print(
        f"S6 round {spec['round']}/384 {spec['hog']} {spec['hsv_layout']} "
        f"hsv_alpha={float(spec['hsv_alpha']):.2f} select_min={local[-2]['min_class_auc']:.3f}",
        flush=True,
    )
    return local


def _partial_frame(path: Path, specs: list[dict[str, float | int | str]]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    expected_rounds = {int(item["round"]) for item in specs}
    if set(frame["round"].astype(int)) != expected_rounds:
        return None
    if len(frame) != len(specs) * len(TRAIN_FRACTIONS) * 2:
        return None
    if not set(frame["subset"]).issubset({"fit", "select"}):
        return None
    return frame


def run_search(cache_dir: Path, out_dir: Path, project: Path, reference: float) -> dict[str, object]:
    reference = require_hsv_alpha(reference)
    specs = iter_s6(reference)
    train_meta, paths = _load_train_meta(cache_dir, project)
    labels = train_meta["class_name"].to_numpy()
    fit_index, select_index = _split_indices(labels)
    parts = _nested_parts(labels, fit_index, SEED)
    npy_dir = out_dir / "npy"
    components = ensure_hsv(paths, npy_dir)
    partial_dir = out_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for geom in HOG_SPECS:
        group = [item for item in specs if item["hog"] == geom["hog"]]
        partial_path = partial_dir / f"{geom['hog']}.csv"
        existing = _partial_frame(partial_path, group)
        if existing is not None:
            print(f"S6 reuse {partial_path.name} ({len(existing)} rows)", flush=True)
            frames.append(existing)
            continue
        hog = ensure_hog(group[0], paths, cache_dir, npy_dir)
        print(f"S6 geometry {geom['hog']} dim={hog.shape[1]} layouts={len(LAYOUTS)}", flush=True)
        records: list[dict[str, float | int | str]] = []
        for layout in LAYOUTS:
            hsv = assemble_matrix(components, layout)
            layout_specs = [item for item in group if item["hsv_layout"] == layout]
            with ThreadPoolExecutor(max_workers=S6_WORKERS) as pool:
                futures = [
                    pool.submit(_eval_spec, spec, hog, hsv, labels, select_index, parts) for spec in layout_specs
                ]
                for future in as_completed(futures):
                    records.extend(future.result())
        records.sort(key=lambda row: (int(row["round"]), float(row["fraction"]), str(row["subset"])))
        frame = pd.DataFrame(records)
        frame.to_csv(partial_path, index=False)
        frames.append(frame)
        del hog

    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values(["round", "fraction", "subset"], kind="mergesort")
    winner = choose_winner(table, reference)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "rounds.csv", index=False)
    chosen = next(item for item in specs if int(item["round"]) == winner)
    chosen_row = table[
        (table["round"] == winner) & np.isclose(table["fraction"].astype(float), 1.0) & (table["subset"] == "select")
    ].iloc[0]
    payload: dict[str, object] = {
        "search": "S6",
        "winner_round": winner,
        "hog": str(chosen["hog"]),
        "orientations": int(chosen["orientations"]),
        "pixels_per_cell": int(chosen["pixels_per_cell"]),
        "cells_per_block": CELLS_PER_BLOCK,
        "block_norm": BLOCK_NORM,
        "hog_dim": int(chosen["hog_dim"]),
        "hsv_layout": str(chosen["hsv_layout"]),
        "hsv_dim": int(chosen["hsv_dim"]),
        "hsv_alpha": float(chosen["hsv_alpha"]),
        "hsv_reference": reference,
        "hsv_floor": HSV_ALPHA_FLOOR,
        "hsv_cap": S6_ALPHA_MAX,
        "C": C_VALUE,
        "rule": (
            "max select min_class_auc, then mean, then fewer HOG numbers, then fewer HSV numbers, "
            "then nearer the HSV reference, then lower round"
        ),
        "select_fraction": SELECT_FRACTION,
        "seed": SEED,
        "select_min_class_auc": float(chosen_row["min_class_auc"]),
        "select_mean_class_auc": float(chosen_row["mean_class_auc"]),
        "image_size": IMAGE_SIZE,
        "test_was_read": False,
        "validation_was_read": False,
        "replaces_report_round": False,
        "replaces_s4_winner": False,
    }
    (out_dir / "winner.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"S6_WINNER {winner} {chosen['hog']} layout={chosen['hsv_layout']} "
        f"hsv_alpha={float(chosen['hsv_alpha']):.2f} select_min={float(chosen_row['min_class_auc']):.3f}",
        flush=True,
    )
    return payload


def _split_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fit_index, select_index = next(
        StratifiedShuffleSplit(n_splits=1, test_size=SELECT_FRACTION, random_state=SEED).split(
            np.zeros(len(labels)), labels
        )
    )
    return np.asarray(fit_index, dtype=int), np.asarray(select_index, dtype=int)


def write_fit_predictions(cache_dir: Path, out_dir: Path, project: Path) -> None:
    """Accuracy rows for the 80% fit subset. Called only after winner.json exists."""
    winner_path = out_dir / "winner.json"
    before = winner_path.read_text(encoding="utf-8")
    payload = json.loads(before)
    if payload.get("test_was_read") or payload.get("validation_was_read"):
        raise RuntimeError("fit predictions refuse a winner that already read a held-out split")
    if payload.get("replaces_report_round") or payload.get("replaces_s4_winner"):
        raise RuntimeError("fit predictions refuse a winner that replaces a frozen model")
    train_meta, paths = _load_train_meta(cache_dir, project)
    labels = train_meta["class_name"].to_numpy()
    fit_index, _select_index = _split_indices(labels)
    hog = _load_npy(out_dir / "npy" / f"hog_{payload['hog']}.npy", (TRAIN_ROWS, int(payload["hog_dim"])))
    if hog is None:
        raise RuntimeError("winner HOG npy is missing")
    components = ensure_hsv(paths, out_dir / "npy")
    hsv = assemble_matrix(components, str(payload["hsv_layout"]))
    clf, _hog_scaler, _hsv_scaler = fit_classifier(hog, hsv, labels, fit_index, float(payload["hsv_alpha"]))
    predicted = np.asarray(clf.predict(np.hstack([
        _hog_scaler.transform(hog[fit_index]),
        _hsv_scaler.transform(hsv[fit_index]) * float(payload["hsv_alpha"]),
    ]).astype(np.float32, copy=False)), dtype=str)
    frame = train_meta.iloc[fit_index][["sha256", "class_name"]].copy()
    frame["predicted_class"] = predicted
    frame["subset"] = "fit"
    frame["round"] = int(payload["winner_round"])
    if len(frame) != len(fit_index):
        raise RuntimeError("fit prediction rows do not match the fit subset")
    frame.to_csv(out_dir / "fit_predictions.csv", index=False)
    if winner_path.read_text(encoding="utf-8") != before:
        raise RuntimeError("fit predictions rewrote winner.json")
    print(f"S6_FIT_PREDICTIONS {len(frame)}", flush=True)


def load_winner_model(out_dir: Path, cache_dir: Path, project: Path) -> dict[str, np.ndarray]:
    """Refit the frozen S6 winner on the training fit subset. Does not read winner selection again."""
    payload = json.loads((out_dir / "winner.json").read_text(encoding="utf-8"))
    if payload.get("test_was_read") or payload.get("validation_was_read"):
        raise RuntimeError("S6 model refit refuses a winner that already read a held-out split")
    train_meta, paths = _load_train_meta(cache_dir, project)
    labels = train_meta["class_name"].to_numpy()
    fit_index, _select_index = _split_indices(labels)
    hog = _load_npy(out_dir / "npy" / f"hog_{payload['hog']}.npy", (TRAIN_ROWS, int(payload["hog_dim"])))
    if hog is None:
        raise RuntimeError("winner HOG npy is missing")
    components = ensure_hsv(paths, out_dir / "npy")
    hsv = assemble_matrix(components, str(payload["hsv_layout"]))
    alpha = float(payload["hsv_alpha"])
    clf, hog_scaler, hsv_scaler = fit_classifier(hog, hsv, labels, fit_index, alpha)
    classes = np.asarray([str(item) for item in clf.classes_])
    return {
        "coef": np.asarray(clf.coef_, dtype=np.float64),
        "intercept": np.asarray(clf.intercept_, dtype=np.float64).reshape(-1),
        "classes": classes,
        "hog_mean": np.asarray(hog_scaler.mean_, dtype=np.float64),
        "hog_scale": np.asarray(hog_scaler.scale_, dtype=np.float64),
        "hsv_mean": np.asarray(hsv_scaler.mean_, dtype=np.float64),
        "hsv_scale": np.asarray(hsv_scaler.scale_, dtype=np.float64),
        "hsv_alpha": np.asarray([alpha], dtype=np.float64),
        "orientations": np.asarray([int(payload["orientations"])], dtype=np.int64),
        "pixels_per_cell": np.asarray([int(payload["pixels_per_cell"])], dtype=np.int64),
        "hsv_layout": np.asarray([str(payload["hsv_layout"])]),
    }


METRIC_LABELS = (
    "Tomato_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Late_blight",
    "Tomato_Septoria_leaf_spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
)
CLASS_REPORT = (
    ("Tomato_Bacterial_spot", "Bacterial Spot"),
    ("Tomato_Late_blight", "Late Blight"),
    ("Tomato_Septoria_leaf_spot", "Septoria Leaf Spot"),
    ("Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato Yellow Leaf Curl Virus"),
    ("Tomato_healthy", "Healthy"),
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy and macro precision, recall, and F1. Macro treats the five classes equally."""
    true_labels = np.asarray(y_true).astype(str)
    predicted = np.asarray(y_pred).astype(str)
    precision, recall, f1, _support = precision_recall_fscore_support(
        true_labels,
        predicted,
        labels=list(METRIC_LABELS),
        average="macro",
        zero_division=0,
    )
    return {
        "Accuracy": float(np.mean(true_labels == predicted)),
        "Macro Precision": float(precision),
        "Macro Recall": float(recall),
        "Macro F1-score": float(f1),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float | str]]:
    """Precision, recall, and F1 for each class. No robustness split is read."""
    true_labels = np.asarray(y_true).astype(str)
    predicted = np.asarray(y_pred).astype(str)
    keys = [key for key, _name in CLASS_REPORT]
    precision, recall, f1, _support = precision_recall_fscore_support(
        true_labels,
        predicted,
        labels=keys,
        average=None,
        zero_division=0,
    )
    rows: list[dict[str, float | str]] = []
    for (_key, name), prec, rec, score in zip(CLASS_REPORT, precision, recall, f1):
        rows.append(
            {
                "Class": name,
                "Precision": float(prec),
                "Recall": float(rec),
                "F1-score": float(score),
            }
        )
    return rows


def write_classification_metrics(cache_dir: Path, out_dir: Path, project: Path) -> pd.DataFrame:
    """Four scores on the training rows, at each fit fraction. Does not read test or validation."""
    winner_path = out_dir / "winner.json"
    before = winner_path.read_text(encoding="utf-8")
    payload = json.loads(before)
    if payload.get("test_was_read") or payload.get("validation_was_read"):
        raise RuntimeError("metrics refuse a winner that already read a held-out split")
    if payload.get("replaces_report_round") or payload.get("replaces_s4_winner"):
        raise RuntimeError("metrics refuse a winner that replaces a frozen model")
    train_meta, paths = _load_train_meta(cache_dir, project)
    labels = train_meta["class_name"].to_numpy()
    fit_index, select_index = _split_indices(labels)
    parts = _nested_parts(labels, fit_index, SEED)
    hog = _load_npy(out_dir / "npy" / f"hog_{payload['hog']}.npy", (TRAIN_ROWS, int(payload["hog_dim"])))
    if hog is None:
        raise RuntimeError("winner HOG npy is missing")
    components = ensure_hsv(paths, out_dir / "npy")
    hsv = assemble_matrix(components, str(payload["hsv_layout"]))
    alpha = float(payload["hsv_alpha"])
    rows: list[dict[str, float | int | str]] = []
    class_rows: list[dict[str, float | int | str]] = []
    for fraction in TRAIN_FRACTIONS:
        used = _take_fraction(parts, float(fraction))
        clf, hog_scaler, hsv_scaler = fit_classifier(hog, hsv, labels, used, alpha)

        def predict(index: np.ndarray) -> np.ndarray:
            design = np.hstack(
                [hog_scaler.transform(hog[index]), hsv_scaler.transform(hsv[index]) * alpha]
            ).astype(np.float32, copy=False)
            return np.asarray(clf.predict(design), dtype=str)

        for subset, index in (("fit", used), ("select", select_index)):
            predicted = predict(index)
            scores = classification_metrics(labels[index], predicted)
            rows.append(
                {
                    "round": int(payload["winner_round"]),
                    "fraction": float(fraction),
                    "subset": subset,
                    "n": int(len(index)),
                    **scores,
                }
            )
            for class_score in per_class_metrics(labels[index], predicted):
                class_rows.append(
                    {
                        "round": int(payload["winner_round"]),
                        "fraction": float(fraction),
                        "subset": subset,
                        "n": int(len(index)),
                        **class_score,
                    }
                )
            print(
                f"S6_METRIC fraction={float(fraction):.1f} {subset} "
                f"Accuracy={scores['Accuracy']:.3f} "
                f"MacroF1={scores['Macro F1-score']:.3f}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    per_class = pd.DataFrame(class_rows)
    frame.to_csv(out_dir / "metrics.csv", index=False)
    per_class.to_csv(out_dir / "metrics_per_class.csv", index=False)
    if winner_path.read_text(encoding="utf-8") != before:
        raise RuntimeError("metrics rewrote winner.json")
    return frame


def predict_rows(hog: np.ndarray, hsv: np.ndarray, model: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    hog_s = (hog - model["hog_mean"]) / model["hog_scale"]
    hsv_s = ((hsv - model["hsv_mean"]) / model["hsv_scale"]) * float(model["hsv_alpha"][0])
    design = np.hstack([hog_s, hsv_s]).astype(np.float64, copy=False)
    scores = design @ model["coef"].T + model["intercept"]
    classes = np.asarray(model["classes"]).astype(str)
    predicted = classes[np.argmax(scores, axis=1)]
    return predicted, np.asarray(scores, dtype=np.float64)


def main() -> None:
    root = _pipeline_root()
    project = find_project(root)
    from vein_hog.search_s4 import reference_alpha

    reference = reference_alpha(root / "outputs" / "hsv_window" / "strength.json")
    out_dir = root / "outputs" / "search_s6"
    cache_dir = root / "outputs" / "cache"
    winner_path = out_dir / "winner.json"
    rounds_path = out_dir / "rounds.csv"
    predictions_path = out_dir / "fit_predictions.csv"
    if winner_path.exists() and rounds_path.exists() and predictions_path.exists():
        payload = json.loads(winner_path.read_text(encoding="utf-8"))
        rows = pd.read_csv(rounds_path)
        if (
            not payload.get("test_was_read")
            and not payload.get("validation_was_read")
            and len(rows) == 384 * len(TRAIN_FRACTIONS) * 2
        ):
            print(
                f"S6_WINNER {payload['winner_round']} {payload['hog']} layout={payload['hsv_layout']} "
                f"hsv_alpha={float(payload['hsv_alpha']):.2f} select_min={float(payload['select_min_class_auc']):.3f}",
                flush=True,
            )
            print("S6_FIT_PREDICTIONS already written", flush=True)
            return
    if not (winner_path.exists() and rounds_path.exists()):
        run_search(cache_dir, out_dir, project, reference)
    write_fit_predictions(cache_dir, out_dir, project)


if __name__ == "__main__":
    main()
