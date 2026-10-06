"""One-shot validation. Refuses to run unless the step-4 gate was passed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from vein_hog.extract import extract_split
from vein_hog.rounds import SEED

HEALTHY = "Tomato_healthy"


def _load_selected(fit_dir: Path) -> dict:
    payload = json.loads((fit_dir / "metrics.json").read_text(encoding="utf-8"))
    if not payload.get("eligible_for_validation"):
        raise SystemExit(
            "validation refused: selected round did not pass test macro-F1 >= 0.90 and gap <= 0.05"
        )
    return payload


def _predict(hog: np.ndarray, hsv: np.ndarray, alpha: float, model: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    hog_s = (hog - model["hog_mean"]) / model["hog_scale"]
    hsv_s = ((hsv - model["hsv_mean"]) / model["hsv_scale"]) * alpha
    design = np.hstack([hog_s, hsv_s]).astype(np.float32, copy=False)
    scores = design @ model["coef"].T + model["intercept"]
    classes = [str(c) for c in model["classes"]]
    pred = np.asarray(classes)[scores.argmax(axis=1)]
    shifted = scores - scores.max(axis=1, keepdims=True)
    expo = np.exp(shifted)
    proba = expo / expo.sum(axis=1, keepdims=True)
    disease = 1.0 - proba[:, classes.index(HEALTHY)]
    return pred, disease


def run_validation(project_root: Path, repo_root: Path, table: pd.DataFrame) -> Path:
    fit_dir = repo_root / "outputs" / "fit"
    payload = _load_selected(fit_dir)
    cache_dir = repo_root / "outputs" / "cache"
    if not (cache_dir / "validation_features.npz").exists():
        extract_split(
            table,
            project_root,
            "validation",
            cache_dir,
            allow_validation=True,
        )
    meta = pd.read_csv(cache_dir / "validation_meta.csv")
    with np.load(cache_dir / "validation_features.npz") as data:
        hog = data["hog"]
        hsv = data["hsv"]
        raw = data["raw_completeness"]
        hue = data["hue"]
        sat = data["sat"]
        offsets = data["offsets"]
    with np.load(fit_dir / "selected_model" / "linear.npz") as model:
        stored = {key: model[key] for key in model.files}
    pred, disease = _predict(hog, hsv, float(payload["alpha"]), stored)
    box = payload["healthy_hs_box"]
    scale = float(payload["completeness_scale"])
    anomaly = np.zeros(len(meta), dtype=np.float32)
    for index in range(len(meta)):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        pixel_h = hue[start:stop]
        pixel_s = sat[start:stop]
        if len(pixel_h) == 0:
            continue
        outside = (pixel_h < box[0]) | (pixel_h > box[1]) | (pixel_s < box[2]) | (pixel_s > box[3])
        anomaly[index] = float(outside.mean())
    scores = meta.copy()
    scores["vein_completeness"] = np.clip(raw * scale, 0.0, 1.0)
    scores["vein_hsv_anomaly"] = anomaly
    scores["svm_disease_probability"] = disease
    scores["disease_risk"] = 0.5 * disease + 0.5 * anomaly
    scores["predicted_class"] = pred
    out_dir = repo_root / "outputs" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "scores.csv", index=False)

    y = scores["class_name"].to_numpy()
    labels = sorted(set(y) | set(pred))
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame(
        {"class_name": labels, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )
    matrix = confusion_matrix(y, pred, labels=labels)
    binary = (y != HEALTHY).astype(int)
    auroc = float(roc_auc_score(binary, scores["disease_risk"])) if len(set(binary)) > 1 else float("nan")
    summary = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "auroc_healthy_vs_rest": auroc,
        "n": int(len(scores)),
        "selected_round": payload["selected_round"],
        "C": payload["C"],
        "alpha": payload["alpha"],
        "test_macro_f1_monitor": payload["test_macro_f1"],
    }
    grouped = scores.groupby("class_name")[["vein_completeness", "disease_risk"]].agg(["mean", "std"])
    grouped.columns = ["completeness_mean", "completeness_std", "risk_mean", "risk_std"]
    report = _markdown(summary, per_class, matrix, labels, grouped.reset_index(), payload)
    report_path = repo_root / "vein_hog" / "reports" / "validation_accuracy.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    per_class.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return report_path


def _markdown(summary, per_class, matrix, labels, vein_table, payload) -> str:
    lines = [
        "# Validation accuracy",
        "",
        "The test split was a development monitor and was used to choose among the 16 fixed rounds.",
        "This table is the single evaluation on validation after that choice was frozen.",
        "Disease risk is not a clinical grade. PlantVillage has class labels only.",
        "",
        f"Selected round {summary['selected_round']}: C={summary['C']}, HSV alpha={summary['alpha']}.",
        f"Monitor test macro-F1 was {summary['test_macro_f1_monitor']:.4f}.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| accuracy | {summary['accuracy']:.4f} |",
        f"| macro-F1 | {summary['macro_f1']:.4f} |",
        f"| weighted-F1 | {summary['weighted_f1']:.4f} |",
        f"| AUROC, healthy vs rest, disease risk | {summary['auroc_healthy_vs_rest']:.4f} |",
        f"| images | {summary['n']} |",
        "",
        "## Per class",
        "",
        "| class | precision | recall | F1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in per_class.iterrows():
        lines.append(
            f"| {row['class_name']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {int(row['support'])} |"
        )
    lines.extend(["", "## Confusion matrix", "", "Rows are true labels, columns follow the same order:", ""])
    lines.append("| true \\\\ pred | " + " | ".join(labels) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(labels)) + "|")
    for name, row in zip(labels, matrix):
        lines.append("| " + name + " | " + " | ".join(str(int(v)) for v in row) + " |")
    lines.extend(
        [
            "",
            "## Vein scores by class",
            "",
            "| class | completeness mean | completeness sd | risk mean | risk sd |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in vein_table.iterrows():
        lines.append(
            f"| {row['class_name']} | {row['completeness_mean']:.4f} | {row['completeness_std']:.4f} | {row['risk_mean']:.4f} | {row['risk_std']:.4f} |"
        )
    lines.extend(["", "Random seed for the fitted round was " + str(SEED) + ".", ""])
    return "\n".join(lines)
