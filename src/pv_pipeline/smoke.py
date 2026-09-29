from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from jsonschema import Draft202012Validator

from .config import ProjectContext
from .data.dataset import load_manifest_rows
from .data.transforms import (
    decode_image,
    deep_eval_transform,
    hog_feature,
    stateless_train_transform,
)
from .determinism import configure, resolve_device
from .evaluation.metrics import build_prediction_rows
from .hashing import canonical_json_hash
from .io_utils import load_json, write_json, write_jsonl
from .models.mobilenet_v3 import build_mobilenet, configure_stage_a, freeze_batch_norm
from .models.small_cnn import SmallCNN
from .training.hog_svm import build_hog_classifier


def select_smoke_rows(context: ProjectContext) -> dict[str, list[dict[str, Any]]]:
    all_rows = load_manifest_rows(context)
    selected = {"train": [], "val": [], "test": []}
    for split in selected:
        for class_info in context.classes:
            candidates = [
                row
                for row in all_rows
                if row["split"] == split and row["class_id"] == class_info.class_id
            ]
            candidates.sort(key=lambda row: (row["selection_rank"], row["relative_path"]))
            selected[split].extend(candidates[:2])
    return selected


def _image(context: ProjectContext, row: dict[str, Any]):
    return decode_image(context.extracted_root / Path(row["relative_path"]))


def _deep_once(
    context: ProjectContext,
    rows: dict[str, list[dict[str, Any]]],
    device: torch.device,
    pipeline_id: str,
) -> list[dict[str, Any]]:
    configure(int(context.config["seed"]))
    if pipeline_id == "small_cnn":
        model = SmallCNN(context.config["small_cnn"])
        parameters = model.parameters()
        learning_rate = float(context.config["small_cnn"]["training"]["learning_rate"])
        after_train = None
    else:
        model, _ = build_mobilenet(context, pretrained=True)
        configure_stage_a(model)
        parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
        learning_rate = float(context.config["mobilenet_v3"]["stage_a"]["learning_rate"])
        after_train = freeze_batch_norm
    model = model.to(device)
    train_inputs = torch.stack(
        [
            stateless_train_transform(
                _image(context, row),
                row["sample_id"],
                0,
                context.config["common_deep_input"],
                context.config["stateless_train_augmentation"],
                int(context.config["seed"]),
            )
            for row in rows["train"]
        ]
    ).to(device)
    train_labels = torch.tensor(
        [row["class_id"] for row in rows["train"]], dtype=torch.long, device=device
    )
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    model.train()
    if after_train is not None:
        after_train(model)
    optimizer.zero_grad(set_to_none=True)
    logits = model(train_inputs)
    loss = torch.nn.functional.cross_entropy(logits, train_labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    val_inputs = torch.stack(
        [
            deep_eval_transform(_image(context, row), context.config["common_deep_input"])
            for row in rows["val"]
        ]
    ).to(device)
    model.eval()
    with torch.inference_mode():
        val_logits = model(val_inputs)
        probabilities = torch.softmax(val_logits, dim=1).cpu().numpy()
    predicted = probabilities.argmax(axis=1)
    return build_prediction_rows(
        experiment_id="smoke",
        pipeline_id=pipeline_id,
        split="val",
        manifest_rows=rows["val"],
        predicted=predicted,
        scores=probabilities,
        score_type="probability",
        classes=context.classes,
    )


def _hog_once(
    context: ProjectContext,
    rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    train_x = np.stack(
        [hog_feature(_image(context, row), context.config["hog_svm"]) for row in rows["train"]]
    )
    val_x = np.stack(
        [hog_feature(_image(context, row), context.config["hog_svm"]) for row in rows["val"]]
    )
    train_y = np.asarray([row["class_id"] for row in rows["train"]], dtype=np.int64)
    model = build_hog_classifier(context, 0.1)
    model.fit(train_x, train_y)
    scores = np.asarray(model.decision_function(val_x), dtype=np.float64)
    return build_prediction_rows(
        experiment_id="smoke",
        pipeline_id="hog_svm",
        split="val",
        manifest_rows=rows["val"],
        predicted=scores.argmax(axis=1),
        scores=scores,
        score_type="decision_margin",
        classes=context.classes,
    )


def _run_once(
    context: ProjectContext,
    rows: dict[str, list[dict[str, Any]]],
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    configure(int(context.config["seed"]))
    return {
        "hog_svm": _hog_once(context, rows),
        "small_cnn": _deep_once(context, rows, device, "small_cnn"),
        "mobilenet_v3_small": _deep_once(
            context, rows, device, "mobilenet_v3_small"
        ),
    }


def run_smoke_test(context: ProjectContext, device_name: str) -> dict[str, Any]:
    device = resolve_device(device_name)
    rows = select_smoke_rows(context)
    first = _run_once(context, rows, device)
    second = _run_once(context, rows, device)
    first_hash = canonical_json_hash(first)
    second_hash = canonical_json_hash(second)
    schema = load_json(context.root / "schemas" / "prediction_row.schema.json")
    validator = Draft202012Validator(schema)
    schema_errors = [
        error.message
        for pipeline_rows in first.values()
        for row in pipeline_rows
        for error in validator.iter_errors(row)
    ]
    output_root = context.root / "artifacts" / "smoke"
    for run_name, output in (("run1", first), ("run2", second)):
        for pipeline_id, pipeline_rows in output.items():
            write_jsonl(output_root / run_name / f"{pipeline_id}.jsonl", pipeline_rows)
    checks = {
        "sample_count": sum(len(value) for value in rows.values()) == 24,
        "hog_feature_length": int(
            context.config["hog_svm"]["hog"]["expected_feature_length"]
        )
        == 8100,
        "prediction_schema": not schema_errors,
        "deterministic_replay": first_hash == second_hash,
    }
    report = {
        "schema_version": "1.0.0",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "schema_errors": schema_errors,
        "run1_sha256": first_hash,
        "run2_sha256": second_hash,
        "device": str(device),
    }
    write_json(output_root / "smoke_test.json", report)
    if report["status"] != "pass":
        raise RuntimeError(f"Smoke test failed: {report}")
    return report
