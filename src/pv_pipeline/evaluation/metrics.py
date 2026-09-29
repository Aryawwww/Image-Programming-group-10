from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ..config import ClassInfo


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    classes: Sequence[ClassInfo],
) -> dict[str, Any]:
    labels = [item.class_id for item in classes]
    true = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        predicted,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": [
            {
                "class_id": item.class_id,
                "class_name": item.class_name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, item in enumerate(classes)
        ],
        "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist(),
        "samples": int(true.size),
    }


def build_prediction_rows(
    *,
    experiment_id: str,
    pipeline_id: str,
    split: str,
    manifest_rows: Sequence[dict[str, Any]],
    predicted: Sequence[int] | np.ndarray,
    scores: np.ndarray,
    score_type: str,
    classes: Sequence[ClassInfo],
) -> list[dict[str, Any]]:
    predicted_array = np.asarray(predicted, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if predicted_array.shape != (len(manifest_rows),):
        raise ValueError("Prediction count does not match manifest rows")
    if scores_array.shape != (len(manifest_rows), len(classes)):
        raise ValueError(f"Invalid score matrix shape: {scores_array.shape}")
    if not np.isfinite(scores_array).all():
        raise ValueError("Prediction scores contain NaN or Infinity")
    class_by_id = {item.class_id: item for item in classes}
    output = []
    for index, row in enumerate(manifest_rows):
        predicted_id = int(predicted_array[index])
        output.append(
            {
                "schema_version": "1.0.0",
                "experiment_id": experiment_id,
                "pipeline_id": pipeline_id,
                "split": split,
                "sample_id": row["sample_id"],
                "true_class_id": int(row["class_id"]),
                "predicted_class_id": predicted_id,
                "predicted_class_name": class_by_id[predicted_id].class_name,
                "score_type": score_type,
                "scores": [float(value) for value in scores_array[index]],
            }
        )
    return output
