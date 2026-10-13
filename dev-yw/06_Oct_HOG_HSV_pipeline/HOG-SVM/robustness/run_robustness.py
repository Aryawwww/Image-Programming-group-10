"""Score the frozen S4 HOG+SVM under six corruptions on the validation split."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from corrupt import CATEGORIES, CATEGORY_ZH, parameter_record
from model import (
    PIPELINE_ROOT,
    extract_job,
    fit_frozen,
    load_model,
    output_dir,
    predict_rows,
    resolve_shortcut,
    svm_shortcut,
    true_class_scores,
    validation_table,
)
from plot_robustness import render

CLEAN_CATEGORY = "random_noise"
WORKERS = 4


def _classes(model: dict[str, np.ndarray]) -> list[str]:
    return [str(item) for item in np.asarray(model["classes"]).astype(str)]


def _metrics(frame: pd.DataFrame, classes: list[str]) -> dict[str, object]:
    true_labels = frame["class_name"].to_numpy()
    predicted = frame["predicted_class"].to_numpy()
    precision, recall, f1, _support = precision_recall_fscore_support(
        true_labels, predicted, labels=classes, zero_division=0
    )
    row: dict[str, object] = {
        "category": str(frame["category"].iloc[0]),
        "severity": int(frame["severity"].iloc[0]),
        "n": int(len(frame)),
        "accuracy": float(np.mean(true_labels == predicted)),
        "macro_f1": float(f1_score(true_labels, predicted, labels=classes, average="macro", zero_division=0)),
    }
    for name, prec, rec, score in zip(classes, precision, recall, f1):
        row[f"precision__{name}"] = float(prec)
        row[f"recall__{name}"] = float(rec)
        row[f"f1__{name}"] = float(score)
    return row


def _level_path(out_dir: Path, category: str, severity: int) -> Path:
    return out_dir / "levels" / category / f"{int(severity):03d}.csv"


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


def _jobs(validation: pd.DataFrame, project: Path, category: str, severity: int) -> list[tuple[str, str, int, str]]:
    jobs: list[tuple[str, str, int, str]] = []
    for row in validation.itertuples(index=False):
        jobs.append((str(project / str(row.local_relpath)), category, int(severity), str(row.sha256)))
    if len(jobs) != 375:
        raise RuntimeError(f"expected 375 jobs, got {len(jobs)}")
    return jobs


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
    frame = pd.DataFrame(
        {
            "sha256": validation["sha256"].astype(str).to_numpy(),
            "class_name": true_labels.astype(str),
            "predicted_class": predicted.astype(str),
            "true_class_score": true_class_scores(scores, true_labels, classes),
            "category": category,
            "severity": int(severity),
        }
    )
    return frame


def _stack(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if len(pairs) != 375:
        raise RuntimeError(f"expected 375 feature rows, got {len(pairs)}")
    hog = np.stack([item[0] for item in pairs]).astype(np.float64, copy=False)
    hsv = np.stack([item[1] for item in pairs]).astype(np.float64, copy=False)
    return hog, hsv


def _match_clean(frame: pd.DataFrame, root: Path) -> dict[str, object]:
    reference = pd.read_csv(root / "outputs" / "search_s4" / "validation_scores.csv")
    merged = frame.merge(reference, on="sha256", suffixes=("_new", "_ref"))
    if len(merged) != 375:
        raise RuntimeError(f"clean predictions joined {len(merged)} of 375 validation rows")
    same = merged["predicted_class_new"].astype(str) == merged["predicted_class_ref"].astype(str)
    matched = int(same.sum())
    classes = sorted(frame["class_name"].unique())
    macro = float(
        f1_score(
            frame["class_name"],
            frame["predicted_class"],
            labels=classes,
            average="macro",
            zero_division=0,
        )
    )
    report = {
        "n": 375,
        "matched_predicted_class": matched,
        "macro_f1": macro,
        "winner_round": 99,
        "hsv_alpha": 0.45,
    }
    if matched != 375:
        sample = merged.loc[~same, ["sha256", "predicted_class_new", "predicted_class_ref"]].head(5)
        print(sample.to_string(index=False), flush=True)
        print(f"ROBUSTNESS_FAILED clean match {matched}/375", flush=True)
        raise SystemExit(f"clean predictions differ from validation_scores.csv ({matched}/375)")
    print(f"CLEAN_MATCH {matched}/375 macro_f1={macro:.6f}", flush=True)
    return report


def _clean_frame(model: dict[str, np.ndarray], clean_path: Path) -> pd.DataFrame:
    if clean_path.exists():
        return pd.read_csv(clean_path)
    validation, project = validation_table()
    print("正在提取干净验证特征。", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        pairs = list(pool.map(extract_job, _jobs(validation, project, CLEAN_CATEGORY, 0), chunksize=8))
    hog, hsv = _stack(pairs)
    frame = _score_frame(model, validation, hog, hsv, CLEAN_CATEGORY, 0)
    frame.to_csv(clean_path, index=False)
    return frame


def ensure_model(out_dir: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """有 SVM.lnk 就直接加载系数，不再重拟合，也不再对照验证集。

    快捷方式不在时，才按冻结超参重放一次拟合，并要求 375 张预测与 validation_scores.csv 一致。
    """
    cache_path = out_dir / "model_cache.npz"
    check_path = out_dir / "model_check.json"
    clean_path = out_dir / "clean_predictions.csv"
    link = svm_shortcut()
    if link.exists():
        target = resolve_shortcut(link)
        print(f"沿用快捷方式 {link} -> {target}", flush=True)
        model = load_model(target)
        return model, _clean_frame(model, clean_path)

    winner_path = PIPELINE_ROOT / "outputs" / "search_s4" / "winner.json"
    winner_text = winner_path.read_text(encoding="utf-8")
    if cache_path.exists() and check_path.exists() and clean_path.exists():
        check = json.loads(check_path.read_text(encoding="utf-8"))
        if (
            int(check.get("matched_predicted_class", 0)) == 375
            and check.get("winner_text") == winner_text
        ):
            print("沿用已核对过的 S4 系数。", flush=True)
            return load_model(cache_path), pd.read_csv(clean_path)

    model = fit_frozen(cache_path)
    validation, project = validation_table()
    print("正在核对 375 张干净验证图的预测。", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        pairs = list(pool.map(extract_job, _jobs(validation, project, CLEAN_CATEGORY, 0), chunksize=8))
    hog, hsv = _stack(pairs)
    frame = _score_frame(model, validation, hog, hsv, CLEAN_CATEGORY, 0)
    try:
        report = _match_clean(frame, PIPELINE_ROOT)
    except SystemExit:
        if cache_path.exists():
            cache_path.unlink()
        raise
    report["winner_text"] = winner_text
    check_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    frame.to_csv(clean_path, index=False)
    return model, frame


def _fit_with_heartbeat(out_dir: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    started = time.perf_counter()
    stop = threading.Event()

    def heartbeat() -> None:
        waited = 0
        while not stop.wait(30):
            waited += 30
            print(f"模型准备仍在进行，已等待 {waited} 秒", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return ensure_model(out_dir)
    finally:
        stop.set()
        thread.join(timeout=1)
        print(f"模型准备结束，用时 {time.perf_counter() - started:.1f} 秒。", flush=True)


def _write_scores(out_dir: Path, rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values(["category", "severity"])
    frame.to_csv(out_dir / "scores.csv", index=False)
    return frame


def _summary(scores: pd.DataFrame) -> dict[str, object]:
    picked = (0, 25, 50, 75, 100)
    by_category: dict[str, dict[str, float]] = {}
    for category in CATEGORIES:
        group = scores[scores["category"] == category]
        by_category[category] = {
            f"s{int(value)}": float(group.loc[group["severity"] == value, "macro_f1"].iloc[0]) for value in picked
        }
    clean = [item["s0"] for item in by_category.values()]
    if max(clean) - min(clean) > 1e-12:
        raise RuntimeError("severity 0 macro-F1 is not the same for every category")
    return {"n": 375, "clean_macro_f1": clean[0], "by_category": by_category}


def sweep(out_dir: Path, model: dict[str, np.ndarray], clean: pd.DataFrame) -> pd.DataFrame:
    validation, project = validation_table()
    classes = _classes(model)
    rows: list[dict[str, object]] = []
    pending: list[tuple[str, int]] = []
    for category in CATEGORIES:
        for severity in range(101):
            path = _level_path(out_dir, category, severity)
            existing = _read_level(path)
            if existing is not None:
                rows.append(_metrics(existing, classes))
            else:
                pending.append((category, severity))
    if pending:
        print(f"待测 {len(pending)} 个严重度，已完成 {len(rows)} 个。", flush=True)
    started = time.perf_counter()
    done = len(CATEGORIES) * 101 - len(pending)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for index, (category, severity) in enumerate(pending, start=1):
            if severity == 0:
                frame = clean.copy()
                frame["category"] = category
                frame["severity"] = 0
            else:
                pairs = list(pool.map(extract_job, _jobs(validation, project, category, severity), chunksize=8))
                hog, hsv = _stack(pairs)
                frame = _score_frame(model, validation, hog, hsv, category, severity)
            _write_level(frame, _level_path(out_dir, category, severity))
            row = _metrics(frame, classes)
            rows.append(row)
            done += 1
            elapsed = time.perf_counter() - started
            rate = elapsed / index
            remaining = rate * (len(pending) - index)
            print(
                f"{CATEGORY_ZH[category]} 严重度 {severity:03d}/100  "
                f"macro-F1 {float(row['macro_f1']):.3f}  准确率 {float(row['accuracy']):.3f}  "
                f"剩余约 {remaining / 60:.1f} 分钟",
                flush=True,
            )
            if done % 20 == 0:
                _write_scores(out_dir, rows)
    scores = _write_scores(out_dir, rows)
    summary = _summary(scores)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "parameters.json").write_text(json.dumps(parameter_record(), indent=2), encoding="utf-8")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness sweep for the frozen S4 HOG+SVM.")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out_dir = output_dir()
    if args.plot_only:
        render(out_dir)
        print("ROBUSTNESS_DONE", flush=True)
        return
    _model, clean = _fit_with_heartbeat(out_dir)
    if args.check_only:
        print("ROBUSTNESS_CHECK_OK", flush=True)
        return
    started = time.perf_counter()
    sweep(out_dir, _model, clean)
    render(out_dir)
    print(f"扫描和作图完成，用时 {time.perf_counter() - started:.1f} 秒。", flush=True)
    print("ROBUSTNESS_DONE", flush=True)


if __name__ == "__main__":
    main()
