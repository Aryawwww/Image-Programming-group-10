"""Robustness sweep for the frozen S6 winner.

Same six corruptions, severities, and seaborn figures as the S4 sweep.
Scores the 375 main-ref validation images and does not change the winner.
The test split is not opened. Output stays under outputs/search_s6/robustness.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from corrupt import CATEGORIES, CATEGORY_ZH, REQUIRED_CATEGORIES, parameter_record
from plot_robustness import render, render_required
from s6_features import extract_job
from vein_hog.run import find_project
from vein_hog.search_s6 import load_winner_model, predict_rows

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
SEARCH_DIR = PIPELINE_ROOT / "outputs" / "search_s6"
OUT_DIR = SEARCH_DIR / "robustness"
WORKERS = 4
CLEAN_CATEGORY = "random_noise"


def _classes(model: dict[str, np.ndarray]) -> list[str]:
    return [str(item) for item in np.asarray(model["classes"]).astype(str)]


def _metrics(frame: pd.DataFrame, classes: list[str]) -> dict[str, object]:
    true_labels = frame["class_name"].to_numpy()
    predicted = frame["predicted_class"].to_numpy()
    precision, recall, f1, _support = precision_recall_fscore_support(
        true_labels, predicted, labels=classes, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _macro_support = precision_recall_fscore_support(
        true_labels, predicted, labels=classes, average="macro", zero_division=0
    )
    row: dict[str, object] = {
        "category": str(frame["category"].iloc[0]),
        "severity": int(frame["severity"].iloc[0]),
        "n": int(len(frame)),
        "Accuracy": float(np.mean(true_labels == predicted)),
        "Macro Precision": float(macro_precision),
        "Macro Recall": float(macro_recall),
        "Macro F1-score": float(macro_f1),
        "accuracy": float(np.mean(true_labels == predicted)),
        "macro_f1": float(f1_score(true_labels, predicted, labels=classes, average="macro", zero_division=0)),
    }
    for name, prec, rec, score in zip(classes, precision, recall, f1):
        row[f"precision__{name}"] = float(prec)
        row[f"recall__{name}"] = float(rec)
        row[f"f1__{name}"] = float(score)
    return row


def _level_path(category: str, severity: int) -> Path:
    return OUT_DIR / "levels" / category / f"{int(severity):03d}.csv"


def _read_level(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if len(frame) != 375:
        path.unlink()
        return None
    return frame


def _write_level(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def validation_table() -> tuple[pd.DataFrame, Path]:
    project = find_project(PIPELINE_ROOT)
    table = pd.read_csv(PIPELINE_ROOT / "vein_hog" / "data" / "split_map.csv")
    validation = table[table["split"] == "validation"].reset_index(drop=True)
    if len(validation) != 375 or set(validation["split"]) != {"validation"}:
        raise RuntimeError(f"expected 375 validation rows, got {len(validation)}")
    for relpath in validation["local_relpath"].astype(str):
        parts = [part.lower() for part in Path(relpath).parts]
        if "test" in parts:
            raise RuntimeError(f"robustness path enters the test split: {relpath}")
        dataset_at = parts.index("dataset")
        if parts[dataset_at + 1] != "validation":
            raise RuntimeError(f"robustness path is not dataset/validation: {relpath}")
    return validation, project


def _jobs(validation: pd.DataFrame, project: Path, category: str, severity: int, model: dict[str, np.ndarray]) -> list[tuple]:
    orientations = int(model["orientations"][0])
    cell = int(model["pixels_per_cell"][0])
    layout = str(model["hsv_layout"][0])
    jobs = []
    for row in validation.itertuples(index=False):
        jobs.append(
            (
                str(project / str(row.local_relpath)),
                category,
                int(severity),
                str(row.sha256),
                orientations,
                cell,
                layout,
            )
        )
    if len(jobs) != 375:
        raise RuntimeError(f"expected 375 jobs, got {len(jobs)}")
    return jobs


def true_class_scores(scores: np.ndarray, true_labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    index = {str(name): column for column, name in enumerate(classes)}
    columns = np.array([index[str(name)] for name in true_labels], dtype=int)
    return scores[np.arange(len(columns)), columns]


def _score_frame(
    model: dict[str, np.ndarray],
    validation: pd.DataFrame,
    hog: np.ndarray,
    hsv: np.ndarray,
    category: str,
    severity: int,
) -> pd.DataFrame:
    classes = np.asarray(_classes(model))
    predicted, scores = predict_rows(hog, hsv, model)
    true_labels = validation["class_name"].to_numpy()
    return pd.DataFrame(
        {
            "sha256": validation["sha256"].astype(str).to_numpy(),
            "class_name": true_labels.astype(str),
            "predicted_class": predicted.astype(str),
            "true_class_score": true_class_scores(scores, true_labels, classes),
            "category": category,
            "severity": int(severity),
        }
    )


def _stack(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if len(pairs) != 375:
        raise RuntimeError(f"expected 375 feature rows, got {len(pairs)}")
    hog = np.stack([item[0] for item in pairs]).astype(np.float64, copy=False)
    hsv = np.stack([item[1] for item in pairs]).astype(np.float64, copy=False)
    return hog, hsv


def _load_model(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def ensure_model(winner_text: str) -> dict[str, np.ndarray]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "model.npz"
    stamp_path = OUT_DIR / "model_winner.txt"
    if cache_path.exists() and stamp_path.exists() and stamp_path.read_text(encoding="utf-8") == winner_text:
        print("沿用已保存的 S6 系数。", flush=True)
        return _load_model(cache_path)
    project = find_project(PIPELINE_ROOT)
    print("正在按 S6 赢家拟合训练集上的线性 SVM。", flush=True)
    model = load_winner_model(SEARCH_DIR, PIPELINE_ROOT / "outputs" / "cache", project)
    np.savez_compressed(cache_path, **model)
    stamp_path.write_text(winner_text, encoding="utf-8")
    if (SEARCH_DIR / "winner.json").read_text(encoding="utf-8") != winner_text:
        raise RuntimeError("refit rewrote winner.json")
    return model


def _write_scores(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values(["category", "severity"])
    frame.to_csv(OUT_DIR / "scores.csv", index=False)
    return frame


def sweep(model: dict[str, np.ndarray]) -> pd.DataFrame:
    validation, project = validation_table()
    classes = _classes(model)
    rows: list[dict[str, object]] = []
    pending: list[tuple[str, int]] = []
    clean: pd.DataFrame | None = None
    for category in CATEGORIES:
        for severity in range(101):
            existing = _read_level(_level_path(category, severity))
            if existing is not None:
                rows.append(_metrics(existing, classes))
                if severity == 0 and clean is None:
                    clean = existing
            else:
                pending.append((category, severity))
    if pending:
        print(f"待测 {len(pending)} 个严重度，已完成 {len(rows)} 个。", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for index, (category, severity) in enumerate(pending, start=1):
            if severity == 0 and clean is not None:
                frame = clean.copy()
                frame["category"] = category
                frame["severity"] = 0
            else:
                pairs = list(pool.map(extract_job, _jobs(validation, project, category, severity, model), chunksize=8))
                hog, hsv = _stack(pairs)
                frame = _score_frame(model, validation, hog, hsv, category, severity)
                if severity == 0:
                    clean = frame
            _write_level(frame, _level_path(category, severity))
            row = _metrics(frame, classes)
            rows.append(row)
            elapsed = time.perf_counter() - started
            remaining = (elapsed / index) * (len(pending) - index)
            print(
                f"{CATEGORY_ZH[category]} 严重度 {severity:03d}/100  "
                f"macro-F1 {float(row['macro_f1']):.3f}  准确率 {float(row['accuracy']):.3f}  "
                f"剩余约 {remaining / 60:.1f} 分钟",
                flush=True,
            )
            if index % 20 == 0:
                _write_scores(rows)
    scores = _write_scores(rows)
    picked = (0, 25, 50, 75, 100)
    by_category = {}
    for category in CATEGORIES:
        group = scores[scores["category"] == category]
        by_category[category] = {
            f"s{int(value)}": float(group.loc[group["severity"] == value, "macro_f1"].iloc[0]) for value in picked
        }
    clean_scores = [item["s0"] for item in by_category.values()]
    if max(clean_scores) - min(clean_scores) > 1e-12:
        raise RuntimeError("severity 0 macro-F1 is not the same for every category")
    summary = {
        "n": 375,
        "clean_macro_f1": clean_scores[0],
        "by_category": by_category,
        "used_to_choose_winner": False,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "parameters.json").write_text(json.dumps(parameter_record(), indent=2), encoding="utf-8")
    return scores


def _clean_frame() -> pd.DataFrame | None:
    path = OUT_DIR / "levels" / CLEAN_CATEGORY / "000.csv"
    frame = _read_level(path)
    return frame


def sweep_required(model: dict[str, np.ndarray]) -> pd.DataFrame:
    """Gaussian noise, Gaussian blur, and Brightness. Does not replace the six-category scores."""
    validation, project = validation_table()
    classes = _classes(model)
    rows: list[dict[str, object]] = []
    pending: list[tuple[str, int]] = []
    clean = _clean_frame()
    for category in REQUIRED_CATEGORIES:
        for severity in range(101):
            existing = _read_level(_level_path(category, severity))
            if existing is not None:
                rows.append(_metrics(existing, classes))
            else:
                pending.append((category, severity))
    if pending:
        print(f"三项待测 {len(pending)} 个严重度，已完成 {len(rows)} 个。", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for index, (category, severity) in enumerate(pending, start=1):
            if severity == 0 and clean is not None:
                frame = clean.copy()
                frame["category"] = category
                frame["severity"] = 0
            else:
                pairs = list(pool.map(extract_job, _jobs(validation, project, category, severity, model), chunksize=8))
                hog, hsv = _stack(pairs)
                frame = _score_frame(model, validation, hog, hsv, category, severity)
                if severity == 0:
                    clean = frame
            _write_level(frame, _level_path(category, severity))
            row = _metrics(frame, classes)
            rows.append(row)
            elapsed = time.perf_counter() - started
            remaining = (elapsed / index) * (len(pending) - index)
            print(
                f"{CATEGORY_ZH[category]} 严重度 {severity:03d}/100  "
                f"macro-F1 {float(row['macro_f1']):.3f}  准确率 {float(row['accuracy']):.3f}  "
                f"剩余约 {remaining / 60:.1f} 分钟",
                flush=True,
            )
    scores = pd.DataFrame(rows).sort_values(["category", "severity"])
    scores.to_csv(OUT_DIR / "scores_required.csv", index=False)
    picked = (0, 25, 50, 75, 100)
    by_category = {}
    for category in REQUIRED_CATEGORIES:
        group = scores[scores["category"] == category]
        by_category[category] = {
            f"s{int(value)}": float(group.loc[group["severity"] == value, "macro_f1"].iloc[0]) for value in picked
        }
    record = parameter_record()
    summary = {
        "categories": list(REQUIRED_CATEGORIES),
        "n": 375,
        "by_category": by_category,
        "gaussian_noise": record["gaussian_noise"],
        "gaussian_blur": record["gaussian_blur"],
        "brightness": record["brightness"],
        "used_to_choose_winner": False,
    }
    (OUT_DIR / "summary_required.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return scores


def main() -> None:
    winner_path = SEARCH_DIR / "winner.json"
    winner_text = winner_path.read_text(encoding="utf-8")
    payload = json.loads(winner_text)
    if payload.get("test_was_read") or payload.get("validation_was_read"):
        raise RuntimeError("robustness refuses a winner that already read a held-out split during search")
    if payload.get("replaces_report_round") or payload.get("replaces_s4_winner"):
        raise RuntimeError("robustness refuses a winner that replaces a frozen model")
    model = ensure_model(winner_text)
    if "--required" in sys.argv:
        sweep_required(model)
        render_required(OUT_DIR)
        if winner_path.read_text(encoding="utf-8") != winner_text:
            raise RuntimeError("robustness rewrote winner.json")
        print("REQUIRED_ROBUSTNESS_DONE", flush=True)
        return
    sweep(model)
    render(OUT_DIR)
    if winner_path.read_text(encoding="utf-8") != winner_text:
        raise RuntimeError("robustness rewrote winner.json")
    print("ROBUSTNESS_DONE", flush=True)


if __name__ == "__main__":
    main()
