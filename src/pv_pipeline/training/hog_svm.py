from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ..artifacts import experiment_layout, finalize_checksums, write_common_artifacts
from ..config import ProjectContext
from ..data.dataset import load_manifest_rows
from ..data.transforms import decode_image, hog_feature
from ..determinism import configure
from ..evaluation.metrics import build_prediction_rows, classification_metrics
from ..hashing import canonical_json_hash, sha256_file
from ..io_utils import write_csv, write_json, write_jsonl


_FEATURE_CONFIG_KEYS = (
    "input_size",
    "resize_interpolation",
    "resize_reducing_gap",
    "grayscale",
    "feature_dtype",
    "hog",
)


def _feature_config_hash(context: ProjectContext) -> str:
    pipeline_config = context.config["hog_svm"]
    return canonical_json_hash(
        {key: pipeline_config[key] for key in _FEATURE_CONFIG_KEYS}
    )


def _feature_cache_path(context: ProjectContext, split: str) -> Path:
    manifest_hash = sha256_file(context.manifest_path)
    feature_hash = _feature_config_hash(context)
    return (
        context.root
        / "artifacts"
        / "cache"
        / "hog"
        / manifest_hash[:16]
        / feature_hash[:16]
        / f"{split}_features.npz"
    )


def build_hog_classifier(context: ProjectContext, c_value: float) -> Pipeline:
    classifier_config = context.config["hog_svm"]["classifier"]
    return Pipeline(
        (
            (
                "scaler",
                StandardScaler(
                    with_mean=bool(context.config["hog_svm"]["scaler"]["with_mean"]),
                    with_std=bool(context.config["hog_svm"]["scaler"]["with_std"]),
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=float(c_value),
                    penalty=str(classifier_config["penalty"]),
                    loss=str(classifier_config["loss"]),
                    dual=bool(classifier_config["dual"]),
                    tol=float(classifier_config["tol"]),
                    max_iter=int(classifier_config["max_iter"]),
                    class_weight=classifier_config["class_weight"],
                    random_state=int(classifier_config["random_state"]),
                ),
            ),
        )
    )


def extract_features(
    context: ProjectContext,
    rows: Sequence[dict[str, Any]],
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = _feature_cache_path(context, split)
    manifest_hash = sha256_file(context.manifest_path)
    feature_hash = _feature_config_hash(context)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as loaded:
            if str(loaded["manifest_hash"].item()) != manifest_hash:
                raise ValueError(f"Stale HOG cache manifest hash: {cache}")
            if str(loaded["feature_config_hash"].item()) != feature_hash:
                raise ValueError(f"Stale HOG cache feature config: {cache}")
            sample_ids = loaded["sample_ids"]
            expected_ids = np.asarray([row["sample_id"] for row in rows], dtype="<U64")
            if not np.array_equal(sample_ids, expected_ids):
                raise ValueError(f"HOG cache sample order differs from manifest: {cache}")
            return loaded["X"], loaded["y"], sample_ids

    features = np.empty(
        (len(rows), int(context.config["hog_svm"]["hog"]["expected_feature_length"])),
        dtype=np.float64,
    )
    labels = np.empty(len(rows), dtype=np.int64)
    sample_ids = np.asarray([row["sample_id"] for row in rows], dtype="<U64")
    for index, row in enumerate(rows):
        image = decode_image(context.extracted_root / Path(row["relative_path"]))
        features[index] = hog_feature(image, context.config["hog_svm"])
        labels[index] = int(row["class_id"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f".{cache.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        X=features,
        y=labels,
        sample_ids=sample_ids,
        feature_config_hash=np.asarray(feature_hash),
        manifest_hash=np.asarray(manifest_hash),
    )
    os.replace(temporary, cache)
    return features, labels, sample_ids


def train_hog_svm(context: ProjectContext, command: Sequence[str]) -> Path:
    configure(int(context.config["seed"]))
    train_rows = load_manifest_rows(context, "train")
    val_rows = load_manifest_rows(context, "val")
    layout = experiment_layout(context, "hog_svm")
    write_common_artifacts(
        context,
        layout,
        device=torch.device("cpu"),
        command=command,
    )
    train_x, train_y, _ = extract_features(context, train_rows, "train")
    val_x, val_y, _ = extract_features(context, val_rows, "val")
    classifier_config = context.config["hog_svm"]["classifier"]
    history = []
    candidates: list[tuple[float, Pipeline]] = []
    for c_value in classifier_config["c_candidates"]:
        model = build_hog_classifier(context, float(c_value))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_x, train_y)
        convergence = [item for item in captured if issubclass(item.category, ConvergenceWarning)]
        if convergence:
            raise RuntimeError(f"LinearSVC failed to converge for C={c_value}")
        val_pred = model.predict(val_x)
        metrics = classification_metrics(val_y, val_pred, context.classes)
        history.append(
            {
                "C": float(c_value),
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
            }
        )
        candidates.append((float(c_value), model))
    history.sort(key=lambda row: row["C"])
    write_csv(
        layout.pipeline_root / "history.csv",
        history,
        ("C", "val_accuracy", "val_macro_f1"),
    )
    epsilon = float(context.config["model_selection"]["score_comparison_epsilon"])
    best_score = max(float(row["val_macro_f1"]) for row in history)
    selected_c = min(
        float(row["C"])
        for row in history
        if best_score - float(row["val_macro_f1"]) <= epsilon
    )
    selected_model = dict(candidates)[selected_c]
    model_path = layout.pipeline_root / "model.joblib"
    joblib.dump(selected_model, model_path, compress=3)
    write_json(layout.pipeline_root / "feature_config.json", context.config["hog_svm"])
    write_json(
        layout.pipeline_root / "model_config.json",
        {"pipeline": "StandardScaler+LinearSVC", "selected_C": selected_c},
    )
    write_json(
        layout.pipeline_root / "selection.json",
        {
            "selection_metric": "val_macro_f1",
            "tie_break": "smallest_C",
            "epsilon": epsilon,
            "selected_C": selected_c,
            "selected_val_macro_f1": best_score,
            "candidates": history,
        },
    )
    margins = np.asarray(selected_model.decision_function(val_x), dtype=np.float64)
    predicted = margins.argmax(axis=1)
    prediction_rows = build_prediction_rows(
        experiment_id=layout.experiment_id,
        pipeline_id="hog_svm",
        split="val",
        manifest_rows=val_rows,
        predicted=predicted,
        scores=margins,
        score_type="decision_margin",
        classes=context.classes,
    )
    write_jsonl(layout.predictions / "val.jsonl", prediction_rows)
    write_json(
        layout.metrics / "val.json",
        classification_metrics(val_y, predicted, context.classes),
    )
    finalize_checksums(layout)
    return layout.pipeline_root
