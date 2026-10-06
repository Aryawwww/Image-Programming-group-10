"""Fit the same 16 candidates and write seaborn diagnostics.

The test split is scored for the charts. It does not choose the drawn round.
That round is C=1, alpha=0.25, fixed before the scores are read. Validation
is not opened.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from vein_hog.plots import save_round_comparison, save_selected_diagnostics
from vein_hog.rounds import SEED, TRAIN_FRACTIONS, iter_rounds, sgd_alpha

TEST_F1_MIN = 0.90
GAP_MAX = 0.05
HEALTHY = "Tomato_healthy"
# Fixed before any test score is read. Not the maximum on the test split.
REPORT_C = 1.0
REPORT_ALPHA = 0.25


def _block(hog: np.ndarray, hsv: np.ndarray, alpha: float, hog_scaler: StandardScaler, hsv_scaler: StandardScaler) -> np.ndarray:
    hog_s = hog_scaler.transform(hog)
    hsv_s = hsv_scaler.transform(hsv) * float(alpha)
    return np.hstack([hog_s, hsv_s]).astype(np.float32, copy=False)


def _fit(x_hog: np.ndarray, x_hsv: np.ndarray, y: np.ndarray, c_value: float, alpha: float):
    hog_scaler = StandardScaler().fit(x_hog)
    hsv_scaler = StandardScaler().fit(x_hsv)
    design = _block(x_hog, x_hsv, alpha, hog_scaler, hsv_scaler)
    clf = SGDClassifier(
        loss="hinge",
        alpha=sgd_alpha(c_value, len(y)),
        class_weight="balanced",
        random_state=SEED,
        max_iter=1500,
        tol=1e-3,
    )
    clf.fit(design, y)
    return clf, hog_scaler, hsv_scaler


def _macro(clf, design: np.ndarray, y: np.ndarray) -> float:
    pred = clf.predict(design)
    return float(f1_score(y, pred, average="macro", zero_division=0))


_STATE: dict[str, np.ndarray] = {}


def _init_worker(
    train_hog: np.ndarray,
    train_hsv: np.ndarray,
    y_train: np.ndarray,
    test_hog: np.ndarray,
    test_hsv: np.ndarray,
    y_test: np.ndarray,
    order: np.ndarray,
) -> None:
    _STATE["train_hog"] = train_hog
    _STATE["train_hsv"] = train_hsv
    _STATE["y_train"] = y_train
    _STATE["test_hog"] = test_hog
    _STATE["test_hsv"] = test_hsv
    _STATE["y_test"] = y_test
    _STATE["order"] = order


def _pack_model(clf: SGDClassifier, hog_scaler: StandardScaler, hsv_scaler: StandardScaler) -> dict[str, np.ndarray]:
    return {
        "coef": np.asarray(clf.coef_, dtype=np.float32),
        "intercept": np.asarray(clf.intercept_, dtype=np.float32),
        "classes": np.asarray(clf.classes_),
        "hog_mean": np.asarray(hog_scaler.mean_, dtype=np.float32),
        "hog_scale": np.asarray(hog_scaler.scale_, dtype=np.float32),
        "hsv_mean": np.asarray(hsv_scaler.mean_, dtype=np.float32),
        "hsv_scale": np.asarray(hsv_scaler.scale_, dtype=np.float32),
    }


def _design(hog: np.ndarray, hsv: np.ndarray, alpha: float, model: dict[str, np.ndarray]) -> np.ndarray:
    hog_s = (hog - model["hog_mean"]) / model["hog_scale"]
    hsv_s = ((hsv - model["hsv_mean"]) / model["hsv_scale"]) * float(alpha)
    return np.hstack([hog_s, hsv_s]).astype(np.float32, copy=False)


def _labels_and_scores(design: np.ndarray, model: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    scores = design @ model["coef"].T + model["intercept"]
    classes = np.asarray(model["classes"]).astype(str)
    return classes[np.argmax(scores, axis=1)], scores


def evaluate_round(spec: dict[str, float | int]) -> tuple[list[dict[str, float | int | str]], dict[str, np.ndarray], float, float]:
    train_hog = _STATE["train_hog"]
    train_hsv = _STATE["train_hsv"]
    y_train = _STATE["y_train"]
    test_hog = _STATE["test_hog"]
    test_hsv = _STATE["test_hsv"]
    y_test = _STATE["y_test"]
    order = _STATE["order"]
    records: list[dict[str, float | int | str]] = []
    model: dict[str, np.ndarray] | None = None
    train_f1 = 0.0
    test_f1 = 0.0
    print(f"round {spec['round']}/16 C={spec['C']} alpha={spec['alpha']}", flush=True)
    for fraction in TRAIN_FRACTIONS:
        count = max(1, int(round(len(order) * float(fraction))))
        subset = order[:count]
        clf, hog_scaler, hsv_scaler = _fit(
            train_hog[subset],
            train_hsv[subset],
            y_train[subset],
            float(spec["C"]),
            float(spec["alpha"]),
        )
        train_design = _block(train_hog[subset], train_hsv[subset], float(spec["alpha"]), hog_scaler, hsv_scaler)
        test_design = _block(test_hog, test_hsv, float(spec["alpha"]), hog_scaler, hsv_scaler)
        train_score = _macro(clf, train_design, y_train[subset])
        test_score = _macro(clf, test_design, y_test)
        records.append(
            {
                "round": int(spec["round"]),
                "C": float(spec["C"]),
                "alpha": float(spec["alpha"]),
                "fraction": float(fraction),
                "split": "train",
                "macro_f1": train_score,
                "n_train": int(count),
            }
        )
        records.append(
            {
                "round": int(spec["round"]),
                "C": float(spec["C"]),
                "alpha": float(spec["alpha"]),
                "fraction": float(fraction),
                "split": "test",
                "macro_f1": test_score,
                "n_train": int(count),
            }
        )
        if fraction == 1.0:
            model = _pack_model(clf, hog_scaler, hsv_scaler)
            train_f1 = train_score
            test_f1 = test_score
    if model is None:
        raise RuntimeError("full-data model missing")
    print(f"round {spec['round']} done train={train_f1:.3f} test={test_f1:.3f}", flush=True)
    return records, model, train_f1, test_f1


def _disease_from_scores(scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
    names = [str(item) for item in classes]
    if HEALTHY not in names:
        raise RuntimeError("healthy class missing from the classifier")
    shifted = scores - scores.max(axis=1, keepdims=True)
    expo = np.exp(shifted)
    proba = expo / expo.sum(axis=1, keepdims=True)
    return 1.0 - proba[:, names.index(HEALTHY)]


def _previous_completeness(ann: Path) -> pd.DataFrame:
    """Keep the previous generation's vein_completeness column, once."""
    prev = ann / "previous_vein_completeness.csv"
    if prev.exists():
        return pd.read_csv(prev)
    parts: list[pd.DataFrame] = []
    for name in ("train.csv", "test.csv"):
        path = ann / name
        if not path.exists():
            continue
        old = pd.read_csv(path)
        if {"sha256", "vein_completeness"} <= set(old.columns):
            parts.append(old[["sha256", "vein_completeness"]])
    if not parts:
        frame = pd.DataFrame(columns=["sha256", "vein_completeness"])
    else:
        frame = pd.concat(parts, ignore_index=True).drop_duplicates("sha256")
    frame.to_csv(prev, index=False)
    return frame


def run_fit(cache_dir: Path, out_dir: Path) -> dict[str, object]:
    train_meta = pd.read_csv(cache_dir / "train_meta.csv")
    test_meta = pd.read_csv(cache_dir / "test_meta.csv")
    if set(train_meta["split"]) != {"train"} or set(test_meta["split"]) != {"test"}:
        raise RuntimeError("fit cache must not contain validation rows")
    with np.load(cache_dir / "train_features.npz") as train_np:
        train_hog = train_np["hog"]
        train_hsv = train_np["hsv"]
        train_raw = train_np["raw_completeness"]
        train_hue = train_np["hue"]
        train_sat = train_np["sat"]
        train_offsets = train_np["offsets"]
    with np.load(cache_dir / "test_features.npz") as test_np:
        test_hog = test_np["hog"]
        test_hsv = test_np["hsv"]
        test_raw = test_np["raw_completeness"]
        test_hue = test_np["hue"]
        test_sat = test_np["sat"]
        test_offsets = test_np["offsets"]

    y_train = train_meta["class_name"].to_numpy()
    y_test = test_meta["class_name"].to_numpy()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y_train))

    records: list[dict[str, float | int | str]] = []
    stored: dict[int, dict[str, np.ndarray]] = {}
    full_scores: dict[int, tuple[float, float]] = {}
    workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(train_hog, train_hsv, y_train, test_hog, test_hsv, y_test, order),
    ) as pool:
        for round_records, model, train_f1, test_f1 in pool.map(evaluate_round, iter_rounds()):
            records.extend(round_records)
            round_id = int(round_records[0]["round"])
            stored[round_id] = model
            full_scores[round_id] = (train_f1, test_f1)

    curve = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out_dir / "learning_curve.csv", index=False)
    save_round_comparison(curve, out_dir)

    full = curve[np.isclose(curve["fraction"], 1.0)]
    wide = full.pivot(index="round", columns="split", values="macro_f1").reset_index()
    wide = wide.merge(full[full["split"] == "test"][["round", "C", "alpha"]], on="round")
    wide["gap"] = (wide["train"] - wide["test"]).abs()
    wide["eligible"] = (wide["test"] >= TEST_F1_MIN) & (wide["gap"] <= GAP_MAX)
    report_rows = wide[np.isclose(wide["C"], REPORT_C) & np.isclose(wide["alpha"], REPORT_ALPHA)]
    if len(report_rows) != 1:
        raise RuntimeError("pre-specified round C=1, alpha=0.25 is missing")
    chosen = report_rows.iloc[0]
    # The numeric gate is recorded. It does not open the validation split,
    # and it does not choose which round is drawn.
    allow_validation = False

    round_id = int(chosen["round"])
    model = stored[round_id]
    train_f1, test_f1 = full_scores[round_id]
    alpha = float(chosen["alpha"])
    test_design = _design(test_hog, test_hsv, alpha, model)
    pred, test_scores_raw = _labels_and_scores(test_design, model)
    labels = sorted(set(y_train) | set(y_test))
    matrix = confusion_matrix(y_test, pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, pred, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "class_name": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    healthy = train_meta["class_name"] == HEALTHY
    healthy_raw = train_raw[healthy.to_numpy()]
    healthy_raw = healthy_raw[np.isfinite(healthy_raw)]
    median = float(np.median(healthy_raw)) if len(healthy_raw) else 0.0
    gate_path = out_dir.parent / "checks" / "vein_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    report_interior = bool(gate.get("passed"))

    ann = out_dir.parent / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    previous = _previous_completeness(ann)
    class_names = [str(item) for item in model["classes"]]

    def pack(meta: pd.DataFrame, raw: np.ndarray, scores: np.ndarray, pred_y: np.ndarray) -> pd.DataFrame:
        out = meta.copy()
        out = out.merge(previous, on="sha256", how="left")
        for index, name in enumerate(class_names):
            out[f"score_{name}"] = scores[:, index]
        positions = out["class_name"].map({name: index for index, name in enumerate(class_names)})
        if positions.isna().any():
            raise RuntimeError("a row has no score column for its class")
        out["class_score"] = scores[np.arange(len(out)), positions.to_numpy(dtype=int)]
        if report_interior:
            out["vein_completeness_interior"] = raw
        out["predicted_class"] = pred_y
        return out

    train_design = _design(train_hog, train_hsv, alpha, model)
    train_pred, train_scores_raw = _labels_and_scores(train_design, model)
    train_scores = pack(train_meta, train_raw, train_scores_raw, train_pred)
    test_scores = pack(test_meta, test_raw, test_scores_raw, pred)
    train_scores.to_csv(ann / "train.csv", index=False)
    test_scores.to_csv(ann / "test.csv", index=False)

    plotted = pd.concat(
        [
            test_scores.assign(split="test"),
            train_scores.sample(n=min(400, len(train_scores)), random_state=SEED).assign(split="train"),
        ],
        ignore_index=True,
    )
    save_selected_diagnostics(matrix, labels, per_class, plotted, out_dir)

    payload = {
        "rounds": 16,
        "selected_round": round_id,
        "C": float(chosen["C"]),
        "alpha": float(chosen["alpha"]),
        "train_macro_f1": float(train_f1),
        "test_macro_f1": float(test_f1),
        "gap": float(abs(train_f1 - test_f1)),
        "eligible_for_validation": bool(allow_validation),
        "gate": {"test_macro_f1_min": TEST_F1_MIN, "gap_max": GAP_MAX},
        "gate_cleared_by_any_round": bool(wide["eligible"].any()),
        "completeness_scale": None,
        "interior_completeness_reported": report_interior,
        "healthy_raw_median": median,
        "report_rule": "C=1 and alpha=0.25 were fixed before the test scores were read.",
        "note": "Sixteen rounds were refit. The drawn round was not the test-set maximum. Validation was not read. Completeness was not scaled to 0.8.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    wide.to_csv(out_dir / "rounds_full.csv", index=False)
    per_class.to_csv(out_dir / "selected_per_class.csv", index=False)
    model_dir = out_dir / "selected_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_dir / "linear.npz", **model)
    print(json.dumps(payload, indent=2), flush=True)
    return payload
