"""Search S4: HOG geometries times HSV weights from 0.25 to 0.50.

HOG stays the descriptor. HSV stays the auxiliary block, scaled on the
training rows and multiplied by a weight. The weights run from 0.25 to 0.50
in steps of 0.01. Nothing above 0.50 is trained. Weights for one HOG
geometry are fitted concurrently. Cell size and orientation count are the
HOG hyperparameters. Block size stays 2×2 and the block norm stays L2-Hys.
C stays 1. This is not a seventeenth cell of the original C × α grid, and
it does not replace the pre-specified report round.

The winner is the highest minimum Healthy-versus-one-disease AUC on the
same kind of training holdout as S1. The test split is scored only after
winner.json exists. Validation is not read.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog as skimage_hog

from vein_hog.extract import _prepare, load_bgr
from vein_hog.hog import HOG_DIM, IMAGE_SIZE
from vein_hog.hsv import HSV_DIM
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

BLOCK_NORM = "L2-Hys"
CELLS_PER_BLOCK = 2
SGD_MAX_ITER = 1500
SGD_TOL = 1e-3
S4_ALPHA_MAX = 0.50
S4_ALPHA_STEP = 0.01
S4_WORKERS = 4
HOG_SPECS = (
    {"hog": "ori9-cell8", "orientations": 9, "pixels_per_cell": 8},
    {"hog": "ori9-cell16", "orientations": 9, "pixels_per_cell": 16},
    {"hog": "ori6-cell16", "orientations": 6, "pixels_per_cell": 16},
    {"hog": "ori12-cell16", "orientations": 12, "pixels_per_cell": 16},
)


def descriptor_length(orientations: int, pixels_per_cell: int, cells_per_block: int = CELLS_PER_BLOCK) -> int:
    if IMAGE_SIZE % int(pixels_per_cell) != 0:
        raise ValueError(f"cell {pixels_per_cell} does not divide {IMAGE_SIZE}")
    blocks = IMAGE_SIZE // int(pixels_per_cell) - int(cells_per_block) + 1
    if blocks < 1:
        raise ValueError(f"cell {pixels_per_cell} leaves no block")
    return int(blocks * blocks * cells_per_block * cells_per_block * orientations)


def hsv_strengths() -> tuple[float, ...]:
    """0.25, 0.26, …, 0.50. The search does not train a weight above 0.50."""
    steps = int(round((S4_ALPHA_MAX - HSV_ALPHA_FLOOR) / S4_ALPHA_STEP))
    values = tuple(
        require_hsv_alpha(round(HSV_ALPHA_FLOOR + step * S4_ALPHA_STEP, 2)) for step in range(steps + 1)
    )
    if len(values) != 26 or values[0] != HSV_ALPHA_FLOOR or values[-1] != S4_ALPHA_MAX:
        raise RuntimeError(f"expected 26 HSV weights from 0.25 to 0.50, got {values}")
    if any(value > S4_ALPHA_MAX + 1e-12 for value in values):
        raise RuntimeError("an HSV weight is above 0.50")
    return values


def iter_s4(reference: float) -> list[dict[str, float | int | str]]:
    reference = require_hsv_alpha(reference)
    if reference > S4_ALPHA_MAX + 1e-12:
        raise ValueError(f"HSV reference {reference} is above the S4 cap {S4_ALPHA_MAX}")
    strengths = hsv_strengths()
    rounds: list[dict[str, float | int | str]] = []
    number = 1
    for spec in HOG_SPECS:
        length = descriptor_length(int(spec["orientations"]), int(spec["pixels_per_cell"]))
        for alpha in strengths:
            rounds.append(
                {
                    "round": number,
                    "hog": str(spec["hog"]),
                    "orientations": int(spec["orientations"]),
                    "pixels_per_cell": int(spec["pixels_per_cell"]),
                    "cells_per_block": CELLS_PER_BLOCK,
                    "hog_dim": length,
                    "hsv_alpha": float(alpha),
                }
            )
            number += 1
    expected = len(HOG_SPECS) * len(strengths)
    if len(rounds) != expected:
        raise RuntimeError(f"expected {expected} rounds, got {len(rounds)}")
    return rounds


def reference_alpha(path: Path) -> float:
    """The controller's training-split weight. Test and validation must not be in it."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("split") != "train" or payload.get("test_was_read") or payload.get("validation_was_read"):
        raise RuntimeError("HSV reference must be the training-split controller record")
    if abs(float(payload["floor"]) - HSV_ALPHA_FLOOR) > 1e-12:
        raise RuntimeError(f"controller floor {payload['floor']} is not {HSV_ALPHA_FLOOR}")
    return require_hsv_alpha(float(payload["hsv_alpha"]))


def hog_vector(bgr_u8: np.ndarray, orientations: int, pixels_per_cell: int) -> np.ndarray:
    """One HOG vector. Leaf pixels only, same resize as the published descriptor."""
    bgr256, mask256, mask_ok = _prepare(bgr_u8)
    image = cv2.resize(bgr256, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if mask_ok:
        resized = cv2.resize(mask256, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        gray = gray.copy()
        gray[resized == 0] = 0.0
    feat = skimage_hog(
        gray,
        orientations=int(orientations),
        pixels_per_cell=(int(pixels_per_cell), int(pixels_per_cell)),
        cells_per_block=(CELLS_PER_BLOCK, CELLS_PER_BLOCK),
        block_norm=BLOCK_NORM,
        transform_sqrt=True,
        feature_vector=True,
    )
    out = np.asarray(feat, dtype=np.float32).reshape(-1)
    expected = descriptor_length(orientations, pixels_per_cell)
    if out.shape != (expected,):
        raise RuntimeError(f"HOG dimension {out.shape} != {(expected,)}")
    return out


def _hog_job(job: tuple[str, int, int]) -> np.ndarray:
    path, orientations, pixels_per_cell = job
    return hog_vector(load_bgr(Path(path)), orientations, pixels_per_cell)


def compute_hog(paths: list[str], orientations: int, pixels_per_cell: int) -> np.ndarray:
    jobs = [(path, int(orientations), int(pixels_per_cell)) for path in paths]
    out = np.empty((len(jobs), descriptor_length(orientations, pixels_per_cell)), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for index, vector in enumerate(pool.map(_hog_job, jobs, chunksize=8)):
            out[index] = vector
            done = index + 1
            if done % 250 == 0 or done == len(jobs):
                print(f"S4 hog ori{orientations}-cell{pixels_per_cell} {done}/{len(jobs)}", flush=True)
    return out


def fit_healthy_many(
    hog: np.ndarray,
    hsv: np.ndarray,
    labels: np.ndarray,
    fit_index: np.ndarray,
    score_indices: list[np.ndarray],
    hsv_alpha: float,
) -> list[np.ndarray]:
    """Healthy-class scores. HOG and HSV are scaled separately; HSV then carries the weight."""
    alpha = require_hsv_alpha(hsv_alpha)
    hog_scaler = StandardScaler().fit(hog[fit_index])
    hsv_scaler = StandardScaler().fit(hsv[fit_index])

    def design(index: np.ndarray) -> np.ndarray:
        scaled_hog = hog_scaler.transform(hog[index])
        scaled_hsv = hsv_scaler.transform(hsv[index]) * alpha
        return np.hstack([scaled_hog, scaled_hsv]).astype(np.float32, copy=False)

    clf = SGDClassifier(
        loss="hinge",
        alpha=sgd_alpha(C_VALUE, len(fit_index)),
        class_weight="balanced",
        random_state=SEED,
        max_iter=SGD_MAX_ITER,
        tol=SGD_TOL,
    )
    clf.fit(design(fit_index), labels[fit_index])
    classes = [str(item) for item in clf.classes_]
    if HEALTHY not in classes:
        raise RuntimeError("healthy class missing from the classifier")
    column = classes.index(HEALTHY)
    found: list[np.ndarray] = []
    for index in score_indices:
        scores = np.asarray(clf.decision_function(design(index)), dtype=np.float64)
        found.append(scores[:, column])
    return found


def fit_healthy_score(
    hog: np.ndarray,
    hsv: np.ndarray,
    labels: np.ndarray,
    fit_index: np.ndarray,
    score_index: np.ndarray,
    hsv_alpha: float,
) -> np.ndarray:
    return fit_healthy_many(hog, hsv, labels, fit_index, [score_index], hsv_alpha)[0]


def _record(spec: dict[str, float | int | str], fraction: float, subset: str, labels: np.ndarray, score: np.ndarray) -> dict[str, float | int | str]:
    aucs = class_aucs(labels, score)
    values = [aucs[name] for name in DISEASES]
    row: dict[str, float | int | str] = {
        "round": int(spec["round"]),
        "hog": str(spec["hog"]),
        "orientations": int(spec["orientations"]),
        "pixels_per_cell": int(spec["pixels_per_cell"]),
        "cells_per_block": int(spec["cells_per_block"]),
        "hog_dim": int(spec["hog_dim"]),
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
    """Highest select-subset min AUC at fraction 1. Fewer HOG numbers win a tie."""
    reference = require_hsv_alpha(reference)
    if "subset" not in table.columns or (table["subset"] == "test").any():
        raise RuntimeError("winner is chosen before the test split is scored")
    alphas = table["hsv_alpha"].astype(float)
    if (alphas < HSV_ALPHA_FLOOR - 1e-12).any() or (alphas > S4_ALPHA_MAX + 1e-12).any():
        raise RuntimeError("a round is outside the HSV interval [0.25, 0.50]")
    full = table[np.isclose(table["fraction"].astype(float), 1.0) & (table["subset"] == "select")].copy()
    expected = len(HOG_SPECS) * len(hsv_strengths())
    if len(full) != expected:
        raise RuntimeError(f"expected {expected} select rows at fraction 1, got {len(full)}")
    full["near_reference"] = (full["hsv_alpha"].astype(float) - reference).abs()
    full = full.sort_values(
        ["min_class_auc", "mean_class_auc", "hog_dim", "near_reference", "round"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    return int(full.iloc[0]["round"])


def _pipeline_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _paths(meta: pd.DataFrame, project: Path) -> list[str]:
    return [str(project / str(row)) for row in meta["local_relpath"].tolist()]


def _published_hog(cache_dir: Path, split: str) -> np.ndarray:
    with np.load(cache_dir / f"{split}_features.npz") as data:
        hog = np.asarray(data["hog"], dtype=np.float32)
    if hog.shape[1] != HOG_DIM:
        raise RuntimeError(f"cached HOG width {hog.shape[1]} != {HOG_DIM}")
    return hog


def _hsv(cache_dir: Path, split: str) -> np.ndarray:
    with np.load(cache_dir / f"{split}_features.npz") as data:
        hsv = np.asarray(data["hsv"], dtype=np.float32)
    if hsv.shape[1] != HSV_DIM:
        raise RuntimeError(f"cached HSV width {hsv.shape[1]} != {HSV_DIM}")
    return hsv


def _geometry_hog(spec: dict[str, float | int | str], cache_dir: Path, split: str, paths: list[str]) -> np.ndarray:
    orientations = int(spec["orientations"])
    cell = int(spec["pixels_per_cell"])
    if orientations == 9 and cell == 8:
        return _published_hog(cache_dir, split)
    return compute_hog(paths, orientations, cell)


def run_search(cache_dir: Path, out_dir: Path, project: Path, reference: float) -> dict[str, object]:
    reference = require_hsv_alpha(reference)
    specs = iter_s4(reference)
    train_meta = pd.read_csv(cache_dir / "train_meta.csv")
    if set(train_meta["split"]) != {"train"}:
        raise RuntimeError("S4 must not read validation rows")
    hsv = _hsv(cache_dir, "train")
    if len(hsv) != len(train_meta):
        raise RuntimeError("HSV rows do not match the training table")
    labels = train_meta["class_name"].to_numpy()
    completeness = train_meta["raw_completeness"].to_numpy(dtype=np.float64)
    paths = _paths(train_meta, project)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=SELECT_FRACTION, random_state=SEED)
    fit_index, select_index = next(splitter.split(np.zeros(len(labels)), labels))
    fit_index = np.asarray(fit_index, dtype=int)
    select_index = np.asarray(select_index, dtype=int)
    parts = _nested_parts(labels, fit_index, SEED)

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | int | str]] = []
    score_rows: list[pd.DataFrame] = []
    total = len(specs)

    def _eval(spec: dict[str, float | int | str], hog: np.ndarray) -> tuple[list[dict[str, float | int | str]], pd.DataFrame]:
        local: list[dict[str, float | int | str]] = []
        frame: pd.DataFrame | None = None
        for fraction in TRAIN_FRACTIONS:
            used = _take_fraction(parts, float(fraction))
            select_score, fit_score = fit_healthy_many(
                hog, hsv, labels, used, [select_index, used], float(spec["hsv_alpha"])
            )
            local.append(_record(spec, float(fraction), "select", labels[select_index], select_score))
            local.append(_record(spec, float(fraction), "fit", labels[used], fit_score))
            if float(fraction) == 1.0:
                frame = train_meta.iloc[select_index][["sha256", "class_name"]].copy()
                frame["round"] = int(spec["round"])
                frame["subset"] = "select"
                frame["vein_completeness_interior"] = completeness[select_index]
                frame["healthy_score"] = select_score
        print(
            f"S4 round {spec['round']}/{total} {spec['hog']} hsv_alpha={float(spec['hsv_alpha']):.2f} "
            f"select_min={local[-2]['min_class_auc']:.3f}",
            flush=True,
        )
        if frame is None:
            raise RuntimeError(f"round {spec['round']} did not score fraction 1")
        return local, frame

    for geom in HOG_SPECS:
        group = [item for item in specs if item["hog"] == geom["hog"]]
        hog = _geometry_hog(group[0], cache_dir, "train", paths)
        if hog.shape != (len(labels), int(group[0]["hog_dim"])):
            raise RuntimeError(f"{geom['hog']} HOG shape {hog.shape}")
        print(f"S4 geometry {geom['hog']} dim={hog.shape[1]} alphas={len(group)}", flush=True)
        with ThreadPoolExecutor(max_workers=S4_WORKERS) as pool:
            futures = [pool.submit(_eval, spec, hog) for spec in group]
            for future in as_completed(futures):
                local, frame = future.result()
                records.extend(local)
                score_rows.append(frame)

    records.sort(key=lambda row: (int(row["round"]), float(row["fraction"]), str(row["subset"])))
    score_rows.sort(key=lambda frame: int(frame["round"].iloc[0]))
    table = pd.DataFrame(records)
    winner = choose_winner(table, reference)
    table.to_csv(out_dir / "rounds.csv", index=False)
    pd.concat(score_rows, ignore_index=True).to_csv(out_dir / "select_scores.csv", index=False)
    chosen = next(item for item in specs if int(item["round"]) == winner)
    payload: dict[str, object] = {
        "search": "S4",
        "winner_round": winner,
        "hog": str(chosen["hog"]),
        "orientations": int(chosen["orientations"]),
        "pixels_per_cell": int(chosen["pixels_per_cell"]),
        "cells_per_block": CELLS_PER_BLOCK,
        "block_norm": BLOCK_NORM,
        "hog_dim": int(chosen["hog_dim"]),
        "hsv_dim": HSV_DIM,
        "hsv_alpha": float(chosen["hsv_alpha"]),
        "hsv_reference": reference,
        "hsv_floor": HSV_ALPHA_FLOOR,
        "C": C_VALUE,
        "rule": "max select min_class_auc, then mean, then fewer HOG numbers, then nearer the HSV reference, then lower round",
        "select_fraction": SELECT_FRACTION,
        "seed": SEED,
        "test_was_read": False,
        "validation_was_read": False,
        "replaces_report_round": False,
    }
    (out_dir / "winner.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"S4_WINNER {winner} {chosen['hog']} hsv_alpha={float(chosen['hsv_alpha']):.2f}",
        flush=True,
    )
    return payload


def score_winner_on_test(cache_dir: Path, out_dir: Path, project: Path) -> dict[str, object]:
    """Diagnostic only. Called after winner.json exists. Does not change it."""
    payload = json.loads((out_dir / "winner.json").read_text(encoding="utf-8"))
    if payload.get("test_was_read"):
        raise RuntimeError("test diagnostic already written")
    train_meta = pd.read_csv(cache_dir / "train_meta.csv")
    test_meta = pd.read_csv(cache_dir / "test_meta.csv")
    if set(train_meta["split"]) != {"train"} or set(test_meta["split"]) != {"test"}:
        raise RuntimeError("test diagnostic must not contain validation rows")
    labels = train_meta["class_name"].to_numpy()
    fit_index, _select_index = next(
        StratifiedShuffleSplit(n_splits=1, test_size=SELECT_FRACTION, random_state=SEED).split(
            np.zeros(len(labels)), labels
        )
    )
    fit_index = np.asarray(fit_index, dtype=int)
    spec = {
        "orientations": int(payload["orientations"]),
        "pixels_per_cell": int(payload["pixels_per_cell"]),
    }
    train_hog = _geometry_hog(spec, cache_dir, "train", _paths(train_meta, project))
    test_hog = _geometry_hog(spec, cache_dir, "test", _paths(test_meta, project))
    train_hsv = _hsv(cache_dir, "train")
    test_hsv = _hsv(cache_dir, "test")
    alpha = require_hsv_alpha(float(payload["hsv_alpha"]))
    hog_scaler = StandardScaler().fit(train_hog[fit_index])
    hsv_scaler = StandardScaler().fit(train_hsv[fit_index])

    def design(hog: np.ndarray, hsv: np.ndarray, index: np.ndarray | None = None) -> np.ndarray:
        rows_hog = hog if index is None else hog[index]
        rows_hsv = hsv if index is None else hsv[index]
        return np.hstack(
            [hog_scaler.transform(rows_hog), hsv_scaler.transform(rows_hsv) * alpha]
        ).astype(np.float32, copy=False)

    clf = SGDClassifier(
        loss="hinge",
        alpha=sgd_alpha(C_VALUE, len(fit_index)),
        class_weight="balanced",
        random_state=SEED,
        max_iter=SGD_MAX_ITER,
        tol=SGD_TOL,
    )
    clf.fit(design(train_hog, train_hsv, fit_index), labels[fit_index])
    classes = [str(item) for item in clf.classes_]
    test_scores = np.asarray(clf.decision_function(design(test_hog, test_hsv)), dtype=np.float64)
    healthy_score = test_scores[:, classes.index(HEALTHY)]
    test_labels = test_meta["class_name"].to_numpy()
    aucs = class_aucs(test_labels, healthy_score)
    frame = test_meta[["sha256", "class_name"]].copy()
    frame["round"] = int(payload["winner_round"])
    frame["subset"] = "test"
    frame["vein_completeness_interior"] = test_meta["raw_completeness"].to_numpy(dtype=np.float64)
    frame["healthy_score"] = healthy_score
    frame.to_csv(out_dir / "test_scores.csv", index=False)
    diagnostic = {
        "winner_round": int(payload["winner_round"]),
        "used_to_choose_winner": False,
        "min_class_auc": float(min(aucs.values())),
        "mean_class_auc": float(np.mean(list(aucs.values()))),
        **{f"auc_{name}": aucs[name] for name in DISEASES},
    }
    (out_dir / "test_diagnostic.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    print(
        f"S4_TEST_DIAGNOSTIC min={diagnostic['min_class_auc']:.3f} not used to choose",
        flush=True,
    )
    return diagnostic


def main() -> None:
    root = _pipeline_root()
    project = find_project(root)
    reference = reference_alpha(root / "outputs" / "hsv_window" / "strength.json")
    out_dir = root / "outputs" / "search_s4"
    cache_dir = root / "outputs" / "cache"
    run_search(cache_dir, out_dir, project, reference)
    score_winner_on_test(cache_dir, out_dir, project)


if __name__ == "__main__":
    main()
