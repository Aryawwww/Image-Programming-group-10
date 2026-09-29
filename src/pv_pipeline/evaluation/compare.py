from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..artifacts import experiment_layout
from ..config import ProjectContext
from ..io_utils import load_json, read_jsonl, write_csv, write_json
from .runner import PIPELINE_IDS, verify_test_gate


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    matrix = np.bincount(y_true * 4 + y_pred, minlength=16).reshape(4, 4)
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros(4, dtype=np.float64),
        where=denominator != 0,
    )
    return float(f1.mean())


def paired_stratified_bootstrap(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(seed))
    strata = [np.flatnonzero(y_true == class_id) for class_id in range(4)]
    differences = {
        f"{left}-minus-{right}": np.empty(repetitions, dtype=np.float64)
        for left, right in combinations(PIPELINE_IDS, 2)
    }
    for repetition in range(repetitions):
        sampled = np.concatenate(
            [rng.choice(indices, size=indices.size, replace=True) for indices in strata]
        )
        scores = {
            name: _macro_f1(y_true[sampled], values[sampled])
            for name, values in predictions.items()
        }
        for left, right in combinations(PIPELINE_IDS, 2):
            differences[f"{left}-minus-{right}"][repetition] = scores[left] - scores[right]
    return {
        name: {
            "mean_difference": float(values.mean()),
            "ci95": [
                float(value)
                for value in np.quantile(values, [0.025, 0.975], method="linear")
            ],
        }
        for name, values in differences.items()
    }


def compare_models(context: ProjectContext) -> Path:
    verify_test_gate(context)
    comparison_root = context.root / "artifacts" / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    prediction_documents = {}
    metrics = {}
    for pipeline_id in PIPELINE_IDS:
        layout = experiment_layout(context, pipeline_id)
        prediction_path = layout.predictions / "test.jsonl"
        metrics_path = layout.metrics / "test.json"
        if not prediction_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Test outputs are incomplete for {pipeline_id}")
        prediction_documents[pipeline_id] = read_jsonl(prediction_path)
        metrics[pipeline_id] = load_json(metrics_path)

    reference = prediction_documents[PIPELINE_IDS[0]]
    sample_ids = [row["sample_id"] for row in reference]
    y_true = np.asarray([row["true_class_id"] for row in reference], dtype=np.int64)
    predictions = {}
    for pipeline_id, rows in prediction_documents.items():
        if [row["sample_id"] for row in rows] != sample_ids:
            raise ValueError(f"Test sample order differs for {pipeline_id}")
        if [row["true_class_id"] for row in rows] != y_true.tolist():
            raise ValueError(f"Test labels differ for {pipeline_id}")
        predictions[pipeline_id] = np.asarray(
            [row["predicted_class_id"] for row in rows], dtype=np.int64
        )

    table_rows = [
        {
            "pipeline_id": pipeline_id,
            "accuracy": metrics[pipeline_id]["accuracy"],
            "macro_precision": metrics[pipeline_id]["macro_precision"],
            "macro_recall": metrics[pipeline_id]["macro_recall"],
            "macro_f1": metrics[pipeline_id]["macro_f1"],
        }
        for pipeline_id in PIPELINE_IDS
    ]
    write_csv(
        comparison_root / "model_comparison.csv",
        table_rows,
        ("pipeline_id", "accuracy", "macro_precision", "macro_recall", "macro_f1"),
    )
    bootstrap_config = context.config["evaluation"]["paired_stratified_bootstrap"]
    bootstrap = paired_stratified_bootstrap(
        y_true,
        predictions,
        repetitions=int(bootstrap_config["repetitions"]),
        seed=int(bootstrap_config["seed"]),
    )
    practical_threshold = float(
        context.config["evaluation"]["practical_tie_threshold_macro_f1"]
    )
    conclusions = {}
    for left, right in combinations(PIPELINE_IDS, 2):
        key = f"{left}-minus-{right}"
        point_difference = float(metrics[left]["macro_f1"] - metrics[right]["macro_f1"])
        low, high = bootstrap[key]["ci95"]
        if abs(point_difference) < practical_threshold:
            conclusion = "practically_tied"
        elif low <= 0 <= high:
            conclusion = "evidence_inconclusive"
        else:
            conclusion = left if point_difference > 0 else right
        conclusions[key] = {
            "point_difference": point_difference,
            "practical_threshold": practical_threshold,
            "conclusion": conclusion,
        }
    write_json(
        comparison_root / "pairwise_bootstrap.json",
        {"bootstrap": bootstrap, "conclusions": conclusions},
    )

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, pipeline_id in zip(axes, PIPELINE_IDS, strict=True):
        matrix = np.asarray(metrics[pipeline_id]["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(pipeline_id)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_xticks(range(4))
        axis.set_yticks(range(4))
        for row in range(4):
            for column in range(4):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axes.tolist(), shrink=0.75)
    figure.savefig(comparison_root / "comparison_confusion_matrices.png", dpi=160)
    plt.close(figure)

    error_cases = []
    for index, sample_id in enumerate(sample_ids):
        predicted = {name: int(values[index]) for name, values in predictions.items()}
        if any(value != int(y_true[index]) for value in predicted.values()):
            error_cases.append(
                {
                    "sample_id": sample_id,
                    "true_class_id": int(y_true[index]),
                    "predictions": predicted,
                }
            )
    from ..io_utils import write_jsonl

    write_jsonl(comparison_root / "error_cases.jsonl", error_cases)
    return comparison_root
