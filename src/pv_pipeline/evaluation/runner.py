from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch

from ..artifacts import (
    code_state,
    experiment_layout,
    finalize_checksums,
)
from ..config import ProjectContext
from ..data.dataset import load_manifest_rows
from ..determinism import configure, resolve_device
from ..hashing import canonical_json_hash, sha256_file
from ..io_utils import load_json, read_jsonl, write_json, write_jsonl
from ..models.mobilenet_v3 import build_mobilenet
from ..models.serialization import load_model
from ..models.small_cnn import SmallCNN
from ..training.hog_svm import extract_features
from ..training.torch_engine import evaluate_torch
from .metrics import build_prediction_rows, classification_metrics


PIPELINE_IDS = ("hog_svm", "small_cnn", "mobilenet_v3_small")


def _model_path(context: ProjectContext, pipeline_id: str) -> Path:
    layout = experiment_layout(context, pipeline_id)
    suffix = ".joblib" if pipeline_id == "hog_svm" else ".safetensors"
    path = layout.pipeline_root / f"model{suffix}"
    if not path.is_file():
        raise FileNotFoundError(f"Selected model is missing for {pipeline_id}: {path}")
    return path


def _gate_payload(context: ProjectContext) -> dict[str, Any]:
    models = {}
    selections = {}
    for pipeline_id in PIPELINE_IDS:
        layout = experiment_layout(context, pipeline_id)
        model_path = _model_path(context, pipeline_id)
        selection_path = layout.pipeline_root / "selection.json"
        if not selection_path.is_file():
            raise FileNotFoundError(f"Selection record is missing: {selection_path}")
        models[pipeline_id] = sha256_file(model_path)
        selections[pipeline_id] = sha256_file(selection_path)
    return {
        "manifest_sha256": sha256_file(context.manifest_path),
        "resolved_config_sha256": context.config_hash,
        "code": code_state(context),
        "selected_model_sha256": models,
        "selection_record_sha256": selections,
    }


def freeze_test_gate(context: ProjectContext, created_by: str) -> dict[str, Any]:
    payload = _gate_payload(context)
    path = context.root / "artifacts" / "test_gate.json"
    if path.is_file():
        existing = load_json(path)
        if existing.get("locked_payload") != payload:
            raise ValueError("An incompatible frozen test gate already exists")
        if existing.get("locked_payload_sha256") != canonical_json_hash(payload):
            raise ValueError("Existing test gate payload hash is invalid")
        return existing
    gate = {
        "schema_version": "1.0.0",
        "test_gate_frozen": True,
        "created_by": created_by,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "locked_payload": payload,
        "locked_payload_sha256": canonical_json_hash(payload),
    }
    write_json(path, gate)
    return gate


def verify_test_gate(context: ProjectContext) -> dict[str, Any]:
    path = context.root / "artifacts" / "test_gate.json"
    if not path.is_file():
        raise FileNotFoundError("Test gate is not frozen")
    gate = load_json(path)
    expected = _gate_payload(context)
    if gate.get("locked_payload") != expected:
        raise ValueError("Test gate no longer matches manifest, config, code, or selected models")
    if gate.get("locked_payload_sha256") != canonical_json_hash(expected):
        raise ValueError("Test gate payload hash is invalid")
    return gate


def evaluate_pipeline(
    context: ProjectContext,
    *,
    pipeline_id: str,
    split: str,
    device_name: str,
    command: Sequence[str] | None = None,
) -> Path:
    if pipeline_id not in PIPELINE_IDS:
        raise ValueError(f"Unknown pipeline: {pipeline_id}")
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    configure(int(context.config["seed"]))
    if split == "test":
        verify_test_gate(context)
    layout = experiment_layout(context, pipeline_id)
    model_path = _model_path(context, pipeline_id)
    rows = load_manifest_rows(context, split)
    y_true = np.asarray([row["class_id"] for row in rows], dtype=np.int64)

    if pipeline_id == "hog_svm":
        model = joblib.load(model_path)
        features, labels, _ = extract_features(context, rows, split)
        if not np.array_equal(labels, y_true):
            raise ValueError("HOG feature cache labels differ from manifest")
        scores = np.asarray(model.decision_function(features), dtype=np.float64)
        predicted = scores.argmax(axis=1).astype(np.int64)
        score_type = "decision_margin"
        metrics = classification_metrics(y_true, predicted, context.classes)
    else:
        device = resolve_device(device_name)
        if pipeline_id == "small_cnn":
            model = SmallCNN(context.config["small_cnn"])
            batch_size = int(context.config["small_cnn"]["training"]["batch_size"])
        else:
            model, _ = build_mobilenet(context, pretrained=False)
            selection = load_json(layout.pipeline_root / "selection.json")
            selected_stage = selection["selected_stage"]
            stage_config = context.config["mobilenet_v3"][
                "stage_b" if selected_stage == "stage_b" else "stage_a"
            ]
            batch_size = int(stage_config["batch_size"])
        model = model.to(device)
        load_model(model, model_path, device=device)
        evaluation = evaluate_torch(
            context,
            model,
            rows,
            device,
            batch_size=batch_size,
        )
        scores = evaluation.probabilities
        predicted = evaluation.y_pred
        score_type = "probability"
        metrics = dict(evaluation.metrics)
        metrics["loss"] = evaluation.loss

    prediction_rows = build_prediction_rows(
        experiment_id=layout.experiment_id,
        pipeline_id=pipeline_id,
        split=split,
        manifest_rows=rows,
        predicted=predicted,
        scores=scores,
        score_type=score_type,
        classes=context.classes,
    )
    output_path = layout.predictions / f"{split}.jsonl"
    if split == "test" and output_path.is_file():
        existing_hash = sha256_file(output_path)
        temporary_rows_hash = canonical_json_hash(prediction_rows)
        existing_rows_hash = canonical_json_hash(read_jsonl(output_path))
        if existing_rows_hash != temporary_rows_hash:
            raise ValueError(
                f"Existing test predictions differ and were preserved: {output_path} ({existing_hash})"
            )
    else:
        write_jsonl(output_path, prediction_rows)
    write_json(layout.metrics / f"{split}.json", metrics)
    finalize_checksums(layout)
    return output_path


def evaluate_many(
    context: ProjectContext,
    *,
    pipeline: str,
    split: str,
    device_name: str,
    command: Sequence[str] | None = None,
) -> list[Path]:
    pipeline_ids = PIPELINE_IDS if pipeline == "all" else (pipeline,)
    return [
        evaluate_pipeline(
            context,
            pipeline_id=pipeline_id,
            split=split,
            device_name=device_name,
            command=command or sys.argv,
        )
        for pipeline_id in pipeline_ids
    ]
