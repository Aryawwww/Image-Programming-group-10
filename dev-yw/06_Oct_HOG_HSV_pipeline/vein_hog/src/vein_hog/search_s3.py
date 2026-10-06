"""Search S3: twenty Frangi ridge images, then the locked HOG cells.

The fitted vector is the S1 winner summary, not the raw 34,596 numbers.
C stays 1 and HSV stays off. Arrays go under outputs/search_s3 and do not
replace outputs/cache. The published completeness column is not rewritten.

The winner is the highest minimum Healthy-versus-one-disease AUC on the
same training holdout as S1. The S1 value 0.767 is a reference, not a vote.
Validation is not read. The test split is scored after winner.json exists.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from skimage.feature import hog as skimage_hog
from skimage.filters import frangi

from vein_hog.extract import _prepare, load_bgr
from vein_hog.hog import HOG_DIM
from vein_hog.mask import erode_interior
from vein_hog.rounds import SEED, TRAIN_FRACTIONS
from vein_hog.run import find_project
from vein_hog.search_s1 import (
    C_VALUE,
    DISEASES,
    SELECT_FRACTION,
    _fit_score,
    _nested_parts,
    _pipeline_root,
    _take_fraction,
    block_energy,
    class_aucs,
    crop_energy,
    reduce_energy,
)

RADIUS = 8
SUMMARY_BORDER = 2
SUMMARY_POOL = "orient_mean_std"
SUMMARY_DIM = 18
SOURCES = ("green", "exg")
SIGMA_SETS = (
    (1.0,),
    (1.0, 2.0),
    (2.0, 3.0),
    (1.0, 2.0, 3.0),
    (2.0, 3.0, 4.0),
)
HEALTHY_STEMS = ("img_0", "img_103")


def iter_s3() -> list[dict[str, object]]:
    rounds: list[dict[str, object]] = []
    number = 1
    for source in SOURCES:
        for black_ridges in (True, False):
            for sigmas in SIGMA_SETS:
                rounds.append(
                    {
                        "round": number,
                        "source": source,
                        "black_ridges": bool(black_ridges),
                        "polarity": "dark" if black_ridges else "pale",
                        "sigmas": sigmas,
                        "n_sigmas": len(sigmas),
                    }
                )
                number += 1
    if len(rounds) != 20:
        raise RuntimeError(f"expected 20 rounds, got {len(rounds)}")
    return rounds


def sigma_label(sigmas: tuple[float, ...]) -> str:
    return "-".join(str(int(value)) for value in sigmas)


def choose_winner(table: pd.DataFrame) -> int:
    """Highest select-subset min AUC at fraction 1. Fewer scales win a tie."""
    if "subset" not in table.columns or (table["subset"] == "test").any():
        raise RuntimeError("winner is chosen before the test split is scored")
    full = table[np.isclose(table["fraction"].astype(float), 1.0) & (table["subset"] == "select")].copy()
    if len(full) != 20:
        raise RuntimeError(f"expected 20 select rows at fraction 1, got {len(full)}")
    full = full.sort_values(
        ["min_class_auc", "mean_class_auc", "n_sigmas", "round"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return int(full.iloc[0]["round"])


def _stretch(channel: np.ndarray, interior: np.ndarray) -> np.ndarray:
    out = np.zeros(channel.shape, dtype=np.float32)
    values = channel[interior > 0]
    if values.size == 0:
        return out
    low, high = np.percentile(values, [1, 99])
    if float(high - low) < 1e-6:
        return out
    scaled = np.clip((channel - float(low)) / float(high - low), 0.0, 1.0).astype(np.float32)
    scaled[interior == 0] = 0.0
    return scaled


def _channels(bgr256: np.ndarray, interior: np.ndarray) -> dict[str, np.ndarray]:
    image = bgr256.astype(np.float32) / 255.0
    blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return {
        "green": _stretch(green, interior),
        "exg": _stretch(2.0 * green - red - blue, interior),
    }


def ridge_map(
    channel: np.ndarray,
    sigmas: tuple[float, ...],
    black_ridges: bool,
    interior: np.ndarray,
) -> np.ndarray:
    """Frangi response in [0, 1], zero outside the eroded leaf."""
    response = frangi(np.asarray(channel, dtype=np.float32), sigmas=list(sigmas), black_ridges=bool(black_ridges))
    response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    response[interior == 0] = 0.0
    inside = response[interior > 0]
    peak = float(np.percentile(inside, 99)) if inside.size else 0.0
    if peak > 1e-8:
        response = np.clip(response / peak, 0.0, 1.0).astype(np.float32)
    response[interior == 0] = 0.0
    return response


def hog_gray(gray01: np.ndarray) -> np.ndarray:
    """Locked cell parameters on an already prepared 256×256 gray image."""
    feat = skimage_hog(
        np.asarray(gray01, dtype=np.float32),
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    out = np.asarray(feat, dtype=np.float32).reshape(-1)
    if out.shape != (HOG_DIM,):
        raise RuntimeError(f"HOG dimension {out.shape} != {(HOG_DIM,)}")
    return out


def summary_from_gray(gray01: np.ndarray) -> np.ndarray:
    energy = crop_energy(block_energy(hog_gray(gray01)), SUMMARY_BORDER)
    reduced = reduce_energy(energy, SUMMARY_POOL)
    if reduced.shape != (1, SUMMARY_DIM):
        raise RuntimeError(f"summary shape {reduced.shape}")
    return np.asarray(reduced[0], dtype=np.float32)


def summarize_bgr(bgr: np.ndarray) -> np.ndarray:
    """(20, 18) summaries. One row per pre-specified ridge setting."""
    bgr256, mask, mask_ok = _prepare(bgr)
    if mask_ok:
        interior = erode_interior(mask, RADIUS)
    else:
        interior = np.zeros_like(mask)
    channels = _channels(bgr256, interior)
    specs = iter_s3()
    out = np.zeros((len(specs), SUMMARY_DIM), dtype=np.float32)
    for index, spec in enumerate(specs):
        gray = ridge_map(
            channels[str(spec["source"])],
            spec["sigmas"],  # type: ignore[arg-type]
            bool(spec["black_ridges"]),
            interior,
        )
        out[index] = summary_from_gray(gray)
    return out


def summarize_setting(bgr: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    """One (18,) summary. Used after the winner is known."""
    bgr256, mask, mask_ok = _prepare(bgr)
    interior = erode_interior(mask, RADIUS) if mask_ok else np.zeros_like(mask)
    channel = _channels(bgr256, interior)[str(spec["source"])]
    gray = ridge_map(channel, spec["sigmas"], bool(spec["black_ridges"]), interior)  # type: ignore[arg-type]
    return summary_from_gray(gray)


def features_one(path: str) -> np.ndarray:
    return summarize_bgr(load_bgr(Path(path)))


def _named_healthy(test_meta: pd.DataFrame) -> pd.DataFrame:
    healthy = test_meta[test_meta["class_name"] == "Tomato_healthy"]
    picked = []
    for stem in HEALTHY_STEMS:
        hit = healthy[healthy["local_relpath"].str.contains(rf"{stem}(?!\d)", regex=True)]
        if len(hit) != 1:
            raise RuntimeError(f"expected one healthy {stem}, found {len(hit)}")
        picked.append(hit.iloc[0])
    return pd.DataFrame(picked)


def _preview_one(bgr: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    bgr256, mask, mask_ok = _prepare(bgr)
    interior = erode_interior(mask, RADIUS) if mask_ok else np.zeros_like(mask)
    channel = _channels(bgr256, interior)[str(spec["source"])]
    return ridge_map(channel, spec["sigmas"], bool(spec["black_ridges"]), interior)  # type: ignore[arg-type]


def save_previews(project: Path, test_meta: pd.DataFrame, out_path: Path) -> None:
    """Two named healthy leaves. Their pixels are not used to rank rounds."""
    named = _named_healthy(test_meta)
    specs = iter_s3()
    fig, axes = plt.subplots(8, 5, figsize=(10, 14))
    for leaf_index, row in enumerate(named.itertuples(index=False)):
        bgr = load_bgr(project / str(row.local_relpath))
        for spec in specs:
            number = int(spec["round"]) - 1
            axis = axes[leaf_index * 4 + number // 5, number % 5]
            axis.imshow(_preview_one(bgr, spec), cmap="gray", vmin=0, vmax=1)
            axis.set_title(
                f"{number + 1} {spec['source']} {spec['polarity']} {sigma_label(spec['sigmas'])}",
                fontsize=7,
            )
            axis.axis("off")
    fig.suptitle("Frangi ridges on the two named healthy leaves", fontsize=12)
    fig.text(
        0.01,
        0.005,
        "Top half is Tomato_healthy img_0. Bottom half is img_103. "
        "These two test leaves are a picture only. They do not choose the round. Validation was not read.",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _record(spec: dict[str, object], fraction: float, subset: str, labels: np.ndarray, score: np.ndarray) -> dict[str, float | int | str]:
    aucs = class_aucs(labels, score)
    values = [aucs[name] for name in DISEASES]
    row: dict[str, float | int | str] = {
        "round": int(spec["round"]),
        "source": str(spec["source"]),
        "polarity": str(spec["polarity"]),
        "sigmas": sigma_label(spec["sigmas"]),  # type: ignore[arg-type]
        "n_sigmas": int(spec["n_sigmas"]),
        "C": C_VALUE,
        "alpha": 0.0,
        "fraction": float(fraction),
        "subset": subset,
        "min_class_auc": float(min(values)),
        "mean_class_auc": float(np.mean(values)),
        "n": int(len(labels)),
    }
    for name in DISEASES:
        row[f"auc_{name}"] = aucs[name]
    return row


def _s1_reference(root: Path) -> float:
    rounds = pd.read_csv(root / "outputs" / "search_s1" / "rounds.csv")
    hit = rounds[
        (rounds["round"] == 7)
        & np.isclose(rounds["fraction"].astype(float), 1.0)
        & (rounds["subset"] == "select")
    ]
    if len(hit) != 1:
        raise RuntimeError("S1 round 7 select reference is missing")
    return float(hit.iloc[0]["min_class_auc"])


def _same_holdout(root: Path, train_meta: pd.DataFrame, select_index: np.ndarray) -> None:
    scores = pd.read_csv(root / "outputs" / "search_s1" / "select_scores.csv")
    previous = set(scores.loc[scores["round"] == 7, "sha256"].astype(str))
    current = set(train_meta.iloc[select_index]["sha256"].astype(str))
    if previous != current:
        raise RuntimeError("S3 holdout is not the S1 holdout")


def run_search(out_dir: Path, preview_path: Path) -> dict[str, object]:
    root = _pipeline_root()
    project = find_project(root)
    train_meta = pd.read_csv(root / "outputs" / "cache" / "train_meta.csv")
    test_meta = pd.read_csv(root / "outputs" / "cache" / "test_meta.csv")
    if set(train_meta["split"]) != {"train"} or set(test_meta["split"]) != {"test"}:
        raise RuntimeError("S3 meta must not contain validation rows")
    labels = train_meta["class_name"].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=SELECT_FRACTION, random_state=SEED)
    fit_index, select_index = next(splitter.split(np.zeros(len(labels)), labels))
    fit_index = np.asarray(fit_index, dtype=int)
    select_index = np.asarray(select_index, dtype=int)
    _same_holdout(root, train_meta, select_index)
    save_previews(project, test_meta, preview_path)
    print(f"S3_PREVIEW {preview_path}", flush=True)

    paths = [str(project / rel) for rel in train_meta["local_relpath"].astype(str)]
    stored = np.zeros((len(paths), 20, SUMMARY_DIM), dtype=np.float32)
    workers = min(4, os.cpu_count() or 1)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for features in pool.map(features_one, paths, chunksize=8):
            stored[done] = features
            done += 1
            if done % 100 == 0 or done == len(paths):
                print(f"S3 train {done}/{len(paths)}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "train_summaries.npz", features=stored)

    parts = _nested_parts(labels, fit_index, SEED)
    specs = iter_s3()
    records: list[dict[str, float | int | str]] = []
    score_rows: list[pd.DataFrame] = []
    completeness = train_meta["raw_completeness"].to_numpy(dtype=np.float64)
    for spec in specs:
        features = stored[:, int(spec["round"]) - 1, :]
        print(
            f"S3 round {spec['round']}/20 {spec['source']} {spec['polarity']} {sigma_label(spec['sigmas'])}",
            flush=True,
        )
        for fraction in TRAIN_FRACTIONS:
            used = _take_fraction(parts, float(fraction))
            select_score = _fit_score(features, labels, used, select_index)
            fit_score = _fit_score(features, labels, used, used)
            records.append(_record(spec, float(fraction), "select", labels[select_index], select_score))
            records.append(_record(spec, float(fraction), "fit", labels[used], fit_score))
            if float(fraction) == 1.0:
                frame = train_meta.iloc[select_index][["sha256", "class_name"]].copy()
                frame["round"] = int(spec["round"])
                frame["subset"] = "select"
                frame["vein_completeness_interior"] = completeness[select_index]
                frame["healthy_score"] = select_score
                score_rows.append(frame)
        print(f"S3 round {spec['round']} select_min={records[-2]['min_class_auc']:.3f}", flush=True)

    table = pd.DataFrame(records)
    winner = choose_winner(table)
    table.to_csv(out_dir / "rounds.csv", index=False)
    pd.concat(score_rows, ignore_index=True).to_csv(out_dir / "select_scores.csv", index=False)
    chosen = specs[winner - 1]
    reference = _s1_reference(root)
    payload = {
        "search": "S3",
        "winner_round": winner,
        "source": str(chosen["source"]),
        "polarity": str(chosen["polarity"]),
        "sigmas": sigma_label(chosen["sigmas"]),  # type: ignore[arg-type]
        "summary": SUMMARY_POOL,
        "summary_border": SUMMARY_BORDER,
        "n_features": SUMMARY_DIM,
        "C": C_VALUE,
        "alpha": 0.0,
        "s1_reference_round": 7,
        "s1_reference_min_class_auc": reference,
        "rule": "max select min_class_auc, then mean_class_auc, then fewer scales, then lower round",
        "reference_not_used_to_choose": True,
        "test_was_read": False,
        "validation_was_read": False,
        "cache": "outputs/search_s3",
    }
    (out_dir / "winner.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"S3_WINNER {winner} {chosen['source']} {chosen['polarity']} {payload['sigmas']}",
        flush=True,
    )
    return payload


def score_winner_on_test(out_dir: Path) -> dict[str, object]:
    payload = json.loads((out_dir / "winner.json").read_text(encoding="utf-8"))
    if payload.get("test_was_read"):
        raise RuntimeError("test diagnostic already written")
    root = _pipeline_root()
    project = find_project(root)
    train_meta = pd.read_csv(root / "outputs" / "cache" / "train_meta.csv")
    test_meta = pd.read_csv(root / "outputs" / "cache" / "test_meta.csv")
    if set(train_meta["split"]) != {"train"} or set(test_meta["split"]) != {"test"}:
        raise RuntimeError("S3 test diagnostic must not contain validation rows")
    with np.load(out_dir / "train_summaries.npz") as stored_np:
        stored = stored_np["features"]
    labels = train_meta["class_name"].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=SELECT_FRACTION, random_state=SEED)
    fit_index, _select_index = next(splitter.split(np.zeros(len(labels)), labels))
    spec = next(item for item in iter_s3() if int(item["round"]) == int(payload["winner_round"]))
    train_features = stored[:, int(spec["round"]) - 1, :]
    test_features = np.zeros((len(test_meta), SUMMARY_DIM), dtype=np.float32)
    for row_index, rel in enumerate(test_meta["local_relpath"].astype(str)):
        test_features[row_index] = summarize_setting(load_bgr(project / rel), spec)
        if (row_index + 1) % 100 == 0 or row_index + 1 == len(test_meta):
            print(f"S3 test {row_index + 1}/{len(test_meta)}", flush=True)
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    from vein_hog.rounds import sgd_alpha
    from vein_hog.search_s1 import HEALTHY

    scaler = StandardScaler().fit(train_features[fit_index])
    clf = SGDClassifier(
        loss="hinge",
        alpha=sgd_alpha(C_VALUE, len(fit_index)),
        class_weight="balanced",
        random_state=SEED,
        max_iter=4000,
        tol=1e-4,
    )
    clf.fit(scaler.transform(train_features[fit_index]), labels[fit_index])
    classes = [str(item) for item in clf.classes_]
    scores = np.asarray(clf.decision_function(scaler.transform(test_features)), dtype=np.float64)
    healthy_score = scores[:, classes.index(HEALTHY)]
    test_labels = test_meta["class_name"].to_numpy()
    aucs = class_aucs(test_labels, healthy_score)
    frame = test_meta[["sha256", "class_name"]].copy()
    frame["round"] = int(spec["round"])
    frame["subset"] = "test"
    frame["vein_completeness_interior"] = test_meta["raw_completeness"].to_numpy(dtype=np.float64)
    frame["healthy_score"] = healthy_score
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
        f"S3_TEST_DIAGNOSTIC min={diagnostic['min_class_auc']:.3f} not used to choose",
        flush=True,
    )
    return diagnostic


def main() -> None:
    root = _pipeline_root()
    out_dir = root / "outputs" / "search_s3"
    preview = root / "visualize-seaborn" / "s3" / "00_ridge_previews.png"
    run_search(out_dir, preview)
    score_winner_on_test(out_dir)


if __name__ == "__main__":
    main()
