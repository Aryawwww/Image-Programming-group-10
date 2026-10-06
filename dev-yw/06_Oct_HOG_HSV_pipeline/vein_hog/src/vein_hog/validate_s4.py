"""Score the frozen S4 winner on the validation split.

The winner is already round 99: 12 orientations, 16-pixel cells, HSV weight
0.45. This module does not choose a round, does not rewrite winner.json, and
does not write validation rows into outputs/cache. HOG stays on scikit-image.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from vein_hog.extract import DEFAULT_RADIUS, _prepare, load_bgr
from vein_hog.hog import hog_gradient_magnitude
from vein_hog.hsv import HSV_DIM, hsv_aux
from vein_hog.hsv_window import require_hsv_alpha
from vein_hog.mask import erode_interior
from vein_hog.rounds import SEED, sgd_alpha
from vein_hog.run import find_project
from vein_hog.search_s1 import C_VALUE, DISEASES, HEALTHY, SELECT_FRACTION, class_aucs
from vein_hog.search_s4 import (
    CELLS_PER_BLOCK,
    SGD_MAX_ITER,
    SGD_TOL,
    _hsv,
    _paths,
    compute_hog,
)
from vein_hog.vein import raw_completeness, vein_skeleton

FROZEN_ROUND = 99
FROZEN_HOG = "ori12-cell16"
FROZEN_ALPHA = 0.45


def require_frozen_winner(payload: dict[str, object]) -> None:
    """The validation pass may score only the winner S4 already wrote down."""
    if payload.get("search") != "S4" or payload.get("replaces_report_round"):
        raise RuntimeError("validation is only for the frozen S4 winner")
    if int(payload["winner_round"]) != FROZEN_ROUND or str(payload["hog"]) != FROZEN_HOG:
        raise RuntimeError("winner.json is not S4 round 99")
    if abs(float(payload["hsv_alpha"]) - FROZEN_ALPHA) > 1e-9:
        raise RuntimeError("winner HSV weight is not 0.45")
    if int(payload["orientations"]) != 12 or int(payload["pixels_per_cell"]) != 16:
        raise RuntimeError("winner geometry is not 12 orientations and 16-pixel cells")


def _leaf_aux(path: str) -> tuple[np.ndarray, float]:
    bgr256, mask256, mask_ok = _prepare(load_bgr(Path(path)))
    colour = hsv_aux(bgr256, mask256 if mask_ok else np.zeros_like(mask256))
    gray = cv2.cvtColor(bgr256, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = np.sqrt(np.clip(gray, 0.0, 1.0))
    magnitude = hog_gradient_magnitude(gray)
    interior = erode_interior(mask256, DEFAULT_RADIUS) if mask_ok else np.zeros_like(mask256)
    skeleton = vein_skeleton(magnitude, interior)
    return np.asarray(colour, dtype=np.float32), float(raw_completeness(skeleton, mask256))


def score_validation(root: Path | None = None) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3] if root is None else Path(root)
    project = find_project(root)
    out_dir = root / "outputs" / "search_s4"
    cache_dir = root / "outputs" / "cache"
    payload = json.loads((out_dir / "winner.json").read_text(encoding="utf-8"))
    require_frozen_winner(payload)
    before = (out_dir / "winner.json").read_text(encoding="utf-8")

    train_meta = pd.read_csv(cache_dir / "train_meta.csv")
    if set(train_meta["split"]) != {"train"}:
        raise RuntimeError("training cache must not contain validation rows")
    table = pd.read_csv(root / "vein_hog" / "data" / "split_map.csv")
    validation = table[table["split"] == "validation"].reset_index(drop=True)
    if len(validation) != 375 or set(validation["split"]) != {"validation"}:
        raise RuntimeError(f"expected 375 validation rows, got {len(validation)}")

    labels = train_meta["class_name"].to_numpy()
    fit_index, _select_index = next(
        StratifiedShuffleSplit(n_splits=1, test_size=SELECT_FRACTION, random_state=SEED).split(
            np.zeros(len(labels)), labels
        )
    )
    fit_index = np.asarray(fit_index, dtype=int)
    train_hog = compute_hog(_paths(train_meta, project), 12, 16)
    train_hsv = _hsv(cache_dir, "train")
    if train_hog.shape[0] != len(train_meta) or train_hsv.shape != (len(train_meta), HSV_DIM):
        raise RuntimeError("training HOG and HSV rows do not match")

    val_paths = _paths(validation, project)
    val_hog = compute_hog(val_paths, 12, 16)
    val_hsv = np.empty((len(val_paths), HSV_DIM), dtype=np.float32)
    completeness = np.empty(len(val_paths), dtype=np.float64)
    for index, path in enumerate(val_paths):
        colour, score = _leaf_aux(path)
        if colour.shape != (HSV_DIM,):
            raise RuntimeError(f"HSV dimension {colour.shape} != {(HSV_DIM,)}")
        val_hsv[index] = colour
        completeness[index] = score
        done = index + 1
        if done % 100 == 0 or done == len(val_paths):
            print(f"S4 validation aux {done}/{len(val_paths)}", flush=True)

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
    decision = np.asarray(clf.decision_function(design(val_hog, val_hsv)), dtype=np.float64)
    healthy_score = decision[:, classes.index(HEALTHY)]
    predicted = np.asarray(classes)[decision.argmax(axis=1)]
    true_labels = validation["class_name"].to_numpy()
    aucs = class_aucs(true_labels, healthy_score)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted, labels=list(classes), zero_division=0
    )
    frame = validation[["sha256", "class_name"]].copy()
    frame["round"] = FROZEN_ROUND
    frame["subset"] = "validation"
    frame["vein_completeness_interior"] = completeness
    frame["healthy_score"] = healthy_score
    frame["predicted_class"] = predicted
    frame.to_csv(out_dir / "validation_scores.csv", index=False)
    per_class = pd.DataFrame(
        {
            "class_name": list(classes),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    per_class.to_csv(out_dir / "validation_per_class.csv", index=False)
    diagnostic: dict[str, object] = {
        "winner_round": FROZEN_ROUND,
        "hog": FROZEN_HOG,
        "orientations": 12,
        "pixels_per_cell": 16,
        "cells_per_block": CELLS_PER_BLOCK,
        "hsv_alpha": FROZEN_ALPHA,
        "C": C_VALUE,
        "used_to_choose_winner": False,
        "replaces_report_round": False,
        "n": int(len(frame)),
        "macro_f1": float(f1_score(true_labels, predicted, average="macro", zero_division=0)),
        "min_class_auc": float(min(aucs.values())),
        "mean_class_auc": float(np.mean(list(aucs.values()))),
        **{f"auc_{name}": aucs[name] for name in DISEASES},
    }
    (out_dir / "validation_diagnostic.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    if (out_dir / "winner.json").read_text(encoding="utf-8") != before:
        raise RuntimeError("validation rewrote winner.json")
    if (cache_dir / "validation_features.npz").exists() or (cache_dir / "validation_meta.csv").exists():
        raise RuntimeError("validation leaked into the shared feature cache")
    print(
        f"S4_VALIDATION min={diagnostic['min_class_auc']:.3f} macro_f1={diagnostic['macro_f1']:.3f} not used to choose",
        flush=True,
    )
    return diagnostic


def main() -> None:
    score_validation()


if __name__ == "__main__":
    main()
