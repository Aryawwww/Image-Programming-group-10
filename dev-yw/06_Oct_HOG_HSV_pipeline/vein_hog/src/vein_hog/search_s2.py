"""Search S2: twenty skeleton settings. The classifier is not refit.

Percentile and the open-kernel width change interior completeness only.
C, α, and the cached HOG stay at the S1 winner. The reported completeness
column is not rewritten. Validation is not read. The test split is measured
only after the training-subset winner is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from vein_hog.extract import _prepare, load_bgr
from vein_hog.hog import hog_gradient_magnitude
from vein_hog.mask import erode_interior
from vein_hog.rounds import SEED
from vein_hog.run import find_project
from vein_hog.search_s1 import DISEASES, SELECT_FRACTION, _pipeline_root, class_aucs
from vein_hog.vein import raw_completeness, vein_skeleton

PERCENTILES = (80, 85, 90, 95, 97)
KERNELS = (3, 5, 7, 9)
RADIUS = 8


def iter_s2() -> list[dict[str, int | float]]:
    rounds: list[dict[str, int | float]] = []
    number = 1
    for percentile in PERCENTILES:
        for kernel in KERNELS:
            rounds.append(
                {
                    "round": number,
                    "percentile": float(percentile),
                    "open_kernel": int(kernel),
                }
            )
            number += 1
    if len(rounds) != 20:
        raise RuntimeError(f"expected 20 rounds, got {len(rounds)}")
    return rounds


def choose_winner(table: pd.DataFrame) -> int:
    """Highest select-subset min AUC. Ties stay near percentile 90 and kernel 5."""
    if "subset" not in table.columns or (table["subset"] == "test").any():
        raise RuntimeError("winner is chosen before the test split is scored")
    full = table[(table["subset"] == "select")].copy()
    if len(full) != 20:
        raise RuntimeError(f"expected 20 select rows, got {len(full)}")
    full["near_published"] = (full["percentile"] - 90).abs() + (full["open_kernel"] - 5).abs()
    full = full.sort_values(
        ["min_class_auc", "mean_class_auc", "near_published", "round"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return int(full.iloc[0]["round"])


def _one_image(bgr: np.ndarray, specs: list[dict[str, int | float]]) -> np.ndarray:
    bgr256, mask256, mask_ok = _prepare(bgr)
    gray = cv2.cvtColor(bgr256, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = np.sqrt(np.clip(gray, 0.0, 1.0))
    magnitude = hog_gradient_magnitude(gray)
    if mask_ok:
        interior = erode_interior(mask256, RADIUS)
    else:
        interior = np.zeros_like(mask256)
    values = np.empty(len(specs), dtype=np.float64)
    for index, spec in enumerate(specs):
        skeleton = vein_skeleton(
            magnitude,
            interior,
            percentile=float(spec["percentile"]),
            open_kernel=int(spec["open_kernel"]),
        )
        values[index] = raw_completeness(skeleton, mask256)
    return values


def _metric_frame(specs: list[dict[str, int | float]], labels: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    rows = []
    for index, spec in enumerate(specs):
        aucs = class_aucs(labels, values[:, index])
        numbers = [aucs[name] for name in DISEASES]
        row: dict[str, float | int | str] = {
            "round": int(spec["round"]),
            "percentile": float(spec["percentile"]),
            "open_kernel": int(spec["open_kernel"]),
            "subset": "select",
            "min_class_auc": float(min(numbers)),
            "mean_class_auc": float(np.mean(numbers)),
            "n": int(len(labels)),
        }
        for name in DISEASES:
            row[f"auc_{name}"] = aucs[name]
        rows.append(row)
    return pd.DataFrame(rows)


def run_search(out_dir: Path) -> dict[str, object]:
    root = _pipeline_root()
    project = find_project(root)
    train_meta = pd.read_csv(root / "outputs" / "cache" / "train_meta.csv")
    if set(train_meta["split"]) != {"train"}:
        raise RuntimeError("S2 cache must not contain validation rows")
    labels = train_meta["class_name"].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=SELECT_FRACTION, random_state=SEED)
    _fit_index, select_index = next(splitter.split(np.zeros(len(labels)), labels))
    select_index = np.asarray(select_index, dtype=int)
    specs = iter_s2()
    stored = np.zeros((len(train_meta), 20), dtype=np.float32)
    for row_index, rel in enumerate(train_meta["local_relpath"].astype(str)):
        stored[row_index] = _one_image(load_bgr(project / rel), specs)
        if (row_index + 1) % 200 == 0 or row_index + 1 == len(train_meta):
            print(f"S2 train {row_index + 1}/{len(train_meta)}", flush=True)
    published = train_meta["raw_completeness"].to_numpy(dtype=np.float64)
    baseline_index = next(
        index
        for index, spec in enumerate(specs)
        if float(spec["percentile"]) == 90 and int(spec["open_kernel"]) == 5
    )
    baseline = stored[:, baseline_index]
    drift = float(np.max(np.abs(baseline - published)))
    if drift > 1e-4:
        raise RuntimeError(f"percentile 90 kernel 5 drifted from the reported completeness by {drift}")
    table = _metric_frame(specs, labels[select_index], stored[select_index])
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "rounds.csv", index=False)
    winner = choose_winner(table)
    chosen = specs[winner - 1]
    frame = train_meta.iloc[select_index][["sha256", "class_name"]].copy()
    frame["round"] = winner
    frame["subset"] = "select"
    frame["vein_completeness_interior"] = stored[select_index, winner - 1]
    s1_scores = pd.read_csv(root / "outputs" / "search_s1" / "select_scores.csv")
    s1_winner = json.loads((root / "outputs" / "search_s1" / "winner.json").read_text(encoding="utf-8"))
    s1_scores = s1_scores[s1_scores["round"] == int(s1_winner["winner_round"])][["sha256", "healthy_score"]]
    frame = frame.merge(s1_scores, on="sha256", how="left")
    if frame["healthy_score"].isna().any():
        raise RuntimeError("S1 healthy score missing for an S2 selection row")
    frame.to_csv(out_dir / "select_scores.csv", index=False)
    payload = {
        "search": "S2",
        "winner_round": winner,
        "percentile": float(chosen["percentile"]),
        "open_kernel": int(chosen["open_kernel"]),
        "rule": "max select min_class_auc of completeness, then mean, then nearer to percentile 90 and kernel 5",
        "published_pair_drift": drift,
        "s1_winner_round": int(s1_winner["winner_round"]),
        "test_was_read": False,
        "validation_was_read": False,
        "classifier_was_refit": False,
    }
    (out_dir / "winner.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"S2_WINNER {winner} percentile={chosen['percentile']} kernel={chosen['open_kernel']}",
        flush=True,
    )
    return payload


def score_winner_on_test(out_dir: Path) -> dict[str, object]:
    payload = json.loads((out_dir / "winner.json").read_text(encoding="utf-8"))
    if payload.get("test_was_read"):
        raise RuntimeError("test diagnostic already written")
    root = _pipeline_root()
    project = find_project(root)
    test_meta = pd.read_csv(root / "outputs" / "cache" / "test_meta.csv")
    if set(test_meta["split"]) != {"test"}:
        raise RuntimeError("S2 test diagnostic must not contain validation rows")
    spec = next(item for item in iter_s2() if int(item["round"]) == int(payload["winner_round"]))
    values = np.empty(len(test_meta), dtype=np.float64)
    for row_index, rel in enumerate(test_meta["local_relpath"].astype(str)):
        bgr256, mask256, mask_ok = _prepare(load_bgr(project / rel))
        gray = cv2.cvtColor(bgr256, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray = np.sqrt(np.clip(gray, 0.0, 1.0))
        magnitude = hog_gradient_magnitude(gray)
        interior = erode_interior(mask256, RADIUS) if mask_ok else np.zeros_like(mask256)
        skeleton = vein_skeleton(
            magnitude,
            interior,
            percentile=float(spec["percentile"]),
            open_kernel=int(spec["open_kernel"]),
        )
        values[row_index] = raw_completeness(skeleton, mask256)
    labels = test_meta["class_name"].to_numpy()
    aucs = class_aucs(labels, values)
    s1_winner = int(payload["s1_winner_round"])
    s1_test = pd.read_csv(root / "outputs" / "search_s1" / "test_scores.csv")
    s1_test = s1_test[s1_test["round"] == s1_winner][["sha256", "healthy_score"]]
    frame = test_meta[["sha256", "class_name"]].copy()
    frame["round"] = int(spec["round"])
    frame["subset"] = "test"
    frame["vein_completeness_interior"] = values
    frame = frame.merge(s1_test, on="sha256", how="left")
    if frame["healthy_score"].isna().any():
        raise RuntimeError("S1 test score missing for an S2 diagnostic row")
    frame.to_csv(out_dir / "test_scores.csv", index=False)
    diagnostic = {
        "winner_round": int(spec["round"]),
        "used_to_choose_winner": False,
        "min_class_auc": float(min(aucs.values())),
        "mean_class_auc": float(np.mean(list(aucs.values()))),
        **{f"auc_{name}": aucs[name] for name in DISEASES},
    }
    (out_dir / "test_diagnostic.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    print(
        f"S2_TEST_DIAGNOSTIC min={diagnostic['min_class_auc']:.3f} not used to choose",
        flush=True,
    )
    return diagnostic


def main() -> None:
    out_dir = _pipeline_root() / "outputs" / "search_s2"
    run_search(out_dir)
    score_winner_on_test(out_dir)


if __name__ == "__main__":
    main()
