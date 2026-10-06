"""Search S1: twenty pre-specified interior-HOG summaries.

This is not a seventeenth cell of the C × α grid. C stays 1 and the HSV
weight stays 0. Cell size, orientations, and L2-Hys stay inside the cached
34,596-d descriptor. Each round only chooses how many outer blocks to drop
and which summary to keep.

The winner is the round with the highest minimum Healthy-versus-one-disease
AUC on a stratified 20% of the training rows. The test split is scored only
after that choice is written, and only as a picture. Validation is not read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from vein_hog.rounds import SEED, TRAIN_FRACTIONS, sgd_alpha

HEALTHY = "Tomato_healthy"
DISEASES = (
    "Tomato_Bacterial_spot",
    "Tomato_Late_blight",
    "Tomato_Septoria_leaf_spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
)
BORDERS = (1, 2, 3, 4)
POOLS = ("orient_mean", "orient_mean_std", "anisotropy", "entropy", "vein")
POOL_DIM = {
    "orient_mean": 9,
    "orient_mean_std": 18,
    "anisotropy": 2,
    "entropy": 2,
    "vein": 6,
}
BLOCKS = 31
ORIENTS = 9
C_VALUE = 1.0
SELECT_FRACTION = 0.2


def iter_s1() -> list[dict[str, int | str]]:
    rounds: list[dict[str, int | str]] = []
    number = 1
    for border in BORDERS:
        for pool in POOLS:
            rounds.append(
                {
                    "round": number,
                    "border": int(border),
                    "pool": pool,
                    "n_features": int(POOL_DIM[pool]),
                }
            )
            number += 1
    if len(rounds) != 20:
        raise RuntimeError(f"expected 20 rounds, got {len(rounds)}")
    return rounds


def crop_energy(energy: np.ndarray, border: int) -> np.ndarray:
    """Drop ``border`` blocks from each side. Zero keeps the full grid."""
    if border < 0:
        raise ValueError(f"border must be >= 0, got {border}")
    if border == 0:
        return energy
    height = int(energy.shape[1])
    if height <= 2 * border:
        raise ValueError(f"border {border} removes a {height}-block grid")
    return energy[:, border:-border, border:-border, :]


def _anisotropy(energy: np.ndarray) -> np.ndarray:
    total = energy.sum(axis=-1)
    peak = energy.max(axis=-1)
    out = np.zeros(total.shape, dtype=np.float32)
    np.divide(peak, total, out=out, where=total > 0)
    return out


def _entropy(energy: np.ndarray) -> np.ndarray:
    total = energy.sum(axis=-1, keepdims=True)
    safe = np.maximum(total, 1e-12)
    share = np.where(total > 0, energy / safe, 0.0).astype(np.float32, copy=False)
    log_share = np.zeros_like(share)
    np.log(share, out=log_share, where=share > 0)
    return (-(share * log_share).sum(axis=-1)).astype(np.float32, copy=False)


def _coherence(energy: np.ndarray) -> np.ndarray:
    """Neighbor agreement of the dominant unsigned orientation. 1 is identical."""
    dominant = np.argmax(energy, axis=-1).astype(np.int16)
    height, width = dominant.shape[1], dominant.shape[2]
    total = np.zeros(dominant.shape, dtype=np.float32)
    count = np.zeros(dominant.shape, dtype=np.float32)
    edges = (
        (dominant[:, :, :-1], dominant[:, :, 1:], (slice(None), slice(None), slice(None, width - 1)), (slice(None), slice(None), slice(1, None))),
        (dominant[:, :-1, :], dominant[:, 1:, :], (slice(None), slice(None, height - 1), slice(None)), (slice(None), slice(1, None), slice(None))),
    )
    for left, right, left_index, right_index in edges:
        delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
        delta = np.minimum(delta, ORIENTS - delta)
        agree = (1.0 - delta / 4.0).astype(np.float32)
        total[left_index] += agree
        total[right_index] += agree
        count[left_index] += 1.0
        count[right_index] += 1.0
    return (total / np.maximum(count, 1.0)).astype(np.float32, copy=False)


def _mean_std(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=(1, 2))
    std = values.std(axis=(1, 2))
    return np.stack((mean, std), axis=1).astype(np.float32, copy=False)


def reduce_energy(energy: np.ndarray, pool: str) -> np.ndarray:
    """Summarize one cropped grid. Columns are fixed by ``pool``."""
    if pool == "orient_mean":
        return energy.mean(axis=(1, 2)).astype(np.float32, copy=False)
    if pool == "orient_mean_std":
        mean = energy.mean(axis=(1, 2))
        std = energy.std(axis=(1, 2))
        return np.concatenate((mean, std), axis=1).astype(np.float32, copy=False)
    if pool == "anisotropy":
        return _mean_std(_anisotropy(energy))
    if pool == "entropy":
        return _mean_std(_entropy(energy))
    if pool == "vein":
        return np.concatenate(
            (
                _mean_std(_anisotropy(energy)),
                _mean_std(_entropy(energy)),
                _mean_std(_coherence(energy)),
            ),
            axis=1,
        ).astype(np.float32, copy=False)
    raise ValueError(f"unknown pool: {pool}")


def block_energy(hog: np.ndarray) -> np.ndarray:
    """(n, 31, 31, 9) orientation energy from the cached descriptor.

    skimage stores each block as (cell row, cell col, orientation) in C order.
    Summing the four cells keeps the nine unsigned directions.
    """
    blocks = np.asarray(hog, dtype=np.float32).reshape(-1, BLOCKS, BLOCKS, 2, 2, ORIENTS)
    return blocks.sum(axis=(3, 4))


def class_aucs(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    healthy = labels == HEALTHY
    healthy_score = score[healthy]
    if healthy_score.size == 0:
        raise RuntimeError("selection subset has no healthy leaves")
    found: dict[str, float] = {}
    for name in DISEASES:
        other = score[labels == name]
        if other.size == 0:
            raise RuntimeError(f"selection subset has no {name}")
        y_true = np.concatenate((np.ones(healthy_score.size), np.zeros(other.size)))
        y_score = np.concatenate((healthy_score, other))
        found[name] = float(roc_auc_score(y_true, y_score))
    return found


def choose_winner(table: pd.DataFrame) -> int:
    """Highest select-subset min AUC at fraction 1. Test rows are rejected."""
    if "subset" not in table.columns or (table["subset"] == "test").any():
        raise RuntimeError("winner is chosen before the test split is scored")
    full = table[np.isclose(table["fraction"].astype(float), 1.0) & (table["subset"] == "select")].copy()
    if len(full) != 20:
        raise RuntimeError(f"expected 20 select rows at fraction 1, got {len(full)}")
    full = full.sort_values(
        ["min_class_auc", "mean_class_auc", "n_features", "border", "round"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    return int(full.iloc[0]["round"])


def _nested_parts(labels: np.ndarray, indices: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for name in sorted(set(labels[indices].tolist())):
        group = indices[labels[indices] == name]
        parts.append(rng.permutation(group))
    return parts


def _take_fraction(parts: list[np.ndarray], fraction: float) -> np.ndarray:
    chosen = []
    for group in parts:
        count = max(1, int(round(len(group) * float(fraction))))
        chosen.append(group[:count])
    return np.concatenate(chosen)


def _fit_score(features: np.ndarray, labels: np.ndarray, fit_index: np.ndarray, score_index: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(features[fit_index])
    design = scaler.transform(features)
    clf = SGDClassifier(
        loss="hinge",
        alpha=sgd_alpha(C_VALUE, len(fit_index)),
        class_weight="balanced",
        random_state=SEED,
        max_iter=4000,
        tol=1e-4,
    )
    clf.fit(design[fit_index], labels[fit_index])
    classes = [str(item) for item in clf.classes_]
    if HEALTHY not in classes:
        raise RuntimeError("healthy class missing from the classifier")
    scores = clf.decision_function(design[score_index])
    return np.asarray(scores[:, classes.index(HEALTHY)], dtype=np.float64)


def _record(spec: dict[str, int | str], fraction: float, subset: str, labels: np.ndarray, score: np.ndarray) -> dict[str, float | int | str]:
    aucs = class_aucs(labels, score)
    values = [aucs[name] for name in DISEASES]
    row: dict[str, float | int | str] = {
        "round": int(spec["round"]),
        "border": int(spec["border"]),
        "pool": str(spec["pool"]),
        "n_features": int(spec["n_features"]),
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


def _pipeline_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_search(cache_dir: Path, out_dir: Path) -> dict[str, object]:
    train_meta = pd.read_csv(cache_dir / "train_meta.csv")
    if set(train_meta["split"]) != {"train"}:
        raise RuntimeError("S1 cache must not contain validation rows")
    with np.load(cache_dir / "train_features.npz") as train_np:
        energy = block_energy(train_np["hog"])
    labels = train_meta["class_name"].to_numpy()
    completeness = train_meta["raw_completeness"].to_numpy(dtype=np.float64)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=SELECT_FRACTION, random_state=SEED)
    fit_index, select_index = next(splitter.split(np.zeros(len(labels)), labels))
    fit_index = np.asarray(fit_index, dtype=int)
    select_index = np.asarray(select_index, dtype=int)
    parts = _nested_parts(labels, fit_index, SEED)

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | int | str]] = []
    score_rows: list[pd.DataFrame] = []
    for spec in iter_s1():
        cropped = crop_energy(energy, int(spec["border"]))
        features = reduce_energy(cropped, str(spec["pool"]))
        if features.shape != (len(labels), int(spec["n_features"])):
            raise RuntimeError(f"round {spec['round']} feature shape {features.shape}")
        print(
            f"S1 round {spec['round']}/20 border={spec['border']} pool={spec['pool']}",
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
        print(
            f"S1 round {spec['round']} select_min={records[-2]['min_class_auc']:.3f}",
            flush=True,
        )

    table = pd.DataFrame(records)
    winner = choose_winner(table)
    table.to_csv(out_dir / "rounds.csv", index=False)
    pd.concat(score_rows, ignore_index=True).to_csv(out_dir / "select_scores.csv", index=False)
    chosen = next(item for item in iter_s1() if int(item["round"]) == winner)
    payload = {
        "search": "S1",
        "winner_round": winner,
        "border": int(chosen["border"]),
        "pool": str(chosen["pool"]),
        "n_features": int(chosen["n_features"]),
        "C": C_VALUE,
        "alpha": 0.0,
        "rule": "max select min_class_auc, then mean_class_auc, then fewer features, then smaller border, then lower round",
        "select_fraction": SELECT_FRACTION,
        "seed": SEED,
        "test_was_read": False,
        "validation_was_read": False,
    }
    (out_dir / "winner.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"S1_WINNER {winner} {chosen['pool']} border={chosen['border']}", flush=True)
    return payload


def score_winner_on_test(cache_dir: Path, out_dir: Path) -> dict[str, object]:
    """Diagnostic only. Called after ``winner.json`` exists. Does not change it."""
    payload = json.loads((out_dir / "winner.json").read_text(encoding="utf-8"))
    if payload.get("test_was_read"):
        raise RuntimeError("test diagnostic already written")
    train_meta = pd.read_csv(cache_dir / "train_meta.csv")
    test_meta = pd.read_csv(cache_dir / "test_meta.csv")
    if set(train_meta["split"]) != {"train"} or set(test_meta["split"]) != {"test"}:
        raise RuntimeError("test diagnostic must not contain validation rows")
    with np.load(cache_dir / "train_features.npz") as train_np:
        train_energy = block_energy(train_np["hog"])
    with np.load(cache_dir / "test_features.npz") as test_np:
        test_energy = block_energy(test_np["hog"])
    labels = train_meta["class_name"].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=SELECT_FRACTION, random_state=SEED)
    fit_index, _select_index = next(splitter.split(np.zeros(len(labels)), labels))
    spec = next(item for item in iter_s1() if int(item["round"]) == int(payload["winner_round"]))
    train_features = reduce_energy(crop_energy(train_energy, int(spec["border"])), str(spec["pool"]))
    test_features = reduce_energy(crop_energy(test_energy, int(spec["border"])), str(spec["pool"]))
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
    test_scores = np.asarray(clf.decision_function(scaler.transform(test_features)), dtype=np.float64)
    healthy_score = test_scores[:, classes.index(HEALTHY)]
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
        f"S1_TEST_DIAGNOSTIC min={diagnostic['min_class_auc']:.3f} not used to choose",
        flush=True,
    )
    return diagnostic


def main() -> None:
    root = _pipeline_root()
    cache_dir = root / "outputs" / "cache"
    out_dir = root / "outputs" / "search_s1"
    run_search(cache_dir, out_dir)
    score_winner_on_test(cache_dir, out_dir)


if __name__ == "__main__":
    main()
