from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..artifacts import experiment_layout, finalize_checksums, write_common_artifacts
from ..config import ProjectContext
from ..data.dataset import load_manifest_rows
from ..determinism import configure, resolve_device
from ..evaluation.metrics import build_prediction_rows
from ..io_utils import write_csv, write_json, write_jsonl
from ..models.mobilenet_v3 import (
    build_mobilenet,
    configure_stage_a,
    configure_stage_b,
    freeze_batch_norm,
    trainable_parameter_audit,
)
from ..models.serialization import save_model
from ..models.small_cnn import SmallCNN
from .torch_engine import (
    HISTORY_FIELDS,
    evaluate_torch,
    selection_is_better,
    train_epochs,
)


def _write_validation_outputs(
    context: ProjectContext,
    layout: Any,
    evaluation: Any,
    val_rows: Sequence[dict[str, Any]],
) -> None:
    predictions = build_prediction_rows(
        experiment_id=layout.experiment_id,
        pipeline_id=layout.pipeline_id,
        split="val",
        manifest_rows=val_rows,
        predicted=evaluation.y_pred,
        scores=evaluation.probabilities,
        score_type="probability",
        classes=context.classes,
    )
    write_jsonl(layout.predictions / "val.jsonl", predictions)
    metrics = dict(evaluation.metrics)
    metrics["loss"] = evaluation.loss
    write_json(layout.metrics / "val.json", metrics)


def train_small_cnn(
    context: ProjectContext,
    *,
    device_name: str,
    command: Sequence[str] | None = None,
) -> Path:
    configure(int(context.config["seed"]))
    device = resolve_device(device_name)
    model = SmallCNN(context.config["small_cnn"]).to(device)
    layout = experiment_layout(context, "small_cnn")
    write_common_artifacts(
        context,
        layout,
        device=device,
        command=command or sys.argv,
    )
    train_rows = load_manifest_rows(context, "train")
    val_rows = load_manifest_rows(context, "val")
    training = context.config["small_cnn"]["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training["scheduler_t_max"]),
        eta_min=float(training["scheduler_eta_min"]),
    )
    result = train_epochs(
        context=context,
        model=model,
        train_rows=train_rows,
        val_rows=val_rows,
        device=device,
        stage="small_cnn",
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        history_path=layout.pipeline_root / "history.csv",
    )
    model.load_state_dict(result.best_state, strict=True)
    model_path = layout.pipeline_root / "model.safetensors"
    save_model(model, model_path)
    write_json(layout.pipeline_root / "model_config.json", context.config["small_cnn"])
    write_json(
        layout.pipeline_root / "selection.json",
        {
            "selection_metric": "val_macro_f1",
            "tie_break": ["lower_validation_loss", "earlier_epoch"],
            "selected": result.best_record,
        },
    )
    evaluation = evaluate_torch(
        context,
        model,
        val_rows,
        device,
        batch_size=int(training["batch_size"]),
    )
    _write_validation_outputs(context, layout, evaluation, val_rows)
    finalize_checksums(layout)
    return layout.pipeline_root


def train_mobilenet_v3(
    context: ProjectContext,
    *,
    device_name: str,
    command: Sequence[str] | None = None,
) -> Path:
    configure(int(context.config["seed"]))
    device = resolve_device(device_name)
    model, weights_lock = build_mobilenet(context, pretrained=True)
    model = model.to(device)
    layout = experiment_layout(context, "mobilenet_v3_small")
    write_common_artifacts(
        context,
        layout,
        device=device,
        command=command or sys.argv,
        pretrained_weights=weights_lock,
    )
    train_rows = load_manifest_rows(context, "train")
    val_rows = load_manifest_rows(context, "val")
    config = context.config["mobilenet_v3"]

    configure_stage_a(model)
    stage_a_audit = trainable_parameter_audit(model)
    stage_a = config["stage_a"]
    optimizer_a = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(stage_a["learning_rate"]),
        weight_decay=float(stage_a["weight_decay"]),
    )
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_a,
        T_max=int(stage_a["scheduler_t_max"]),
        eta_min=float(stage_a["scheduler_eta_min"]),
    )
    result_a = train_epochs(
        context=context,
        model=model,
        train_rows=train_rows,
        val_rows=val_rows,
        device=device,
        stage="stage_a",
        epochs=int(stage_a["epochs"]),
        batch_size=int(stage_a["batch_size"]),
        optimizer=optimizer_a,
        scheduler=scheduler_a,
        gradient_clip_norm=float(config["common_training"]["gradient_clip_norm"]),
        history_path=layout.pipeline_root / "history_stage_a.csv",
        after_train_mode=freeze_batch_norm,
    )
    model.load_state_dict(result_a.best_state, strict=True)
    candidates_dir = layout.pipeline_root / "candidates"
    save_model(model, candidates_dir / "stage_a.safetensors")

    stage_b = config["stage_b"]
    feature_start = int(stage_b["trainable_feature_start_index"])
    configure_stage_b(model, feature_start)
    stage_b_audit = trainable_parameter_audit(model)
    optimizer_b = torch.optim.AdamW(
        (
            {
                "params": model.features[feature_start:].parameters(),
                "lr": float(stage_b["feature_learning_rate"]),
            },
            {
                "params": model.classifier.parameters(),
                "lr": float(stage_b["classifier_learning_rate"]),
            },
        ),
        weight_decay=float(stage_b["weight_decay"]),
    )
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b,
        T_max=int(stage_b["scheduler_t_max"]),
        eta_min=float(stage_b["scheduler_eta_min"]),
    )
    result_b = train_epochs(
        context=context,
        model=model,
        train_rows=train_rows,
        val_rows=val_rows,
        device=device,
        stage="stage_b",
        epochs=int(stage_b["epochs"]),
        batch_size=int(stage_b["batch_size"]),
        optimizer=optimizer_b,
        scheduler=scheduler_b,
        gradient_clip_norm=float(config["common_training"]["gradient_clip_norm"]),
        history_path=layout.pipeline_root / "history_stage_b.csv",
        after_train_mode=freeze_batch_norm,
    )
    model.load_state_dict(result_b.best_state, strict=True)
    save_model(model, candidates_dir / "stage_b.safetensors")

    epsilon = float(context.config["model_selection"]["score_comparison_epsilon"])
    stage_b_wins = selection_is_better(result_b.best_record, result_a.best_record, epsilon)
    selected_stage = "stage_b" if stage_b_wins else "stage_a"
    selected_result = result_b if stage_b_wins else result_a
    model.load_state_dict(selected_result.best_state, strict=True)
    save_model(model, layout.pipeline_root / "model.safetensors")
    combined_history = result_a.history + result_b.history
    write_csv(layout.pipeline_root / "history.csv", combined_history, HISTORY_FIELDS)
    write_json(
        layout.pipeline_root / "trainable_parameters.json",
        {"stage_a": stage_a_audit, "stage_b": stage_b_audit},
    )
    write_json(layout.pipeline_root / "model_config.json", config)
    write_json(
        layout.pipeline_root / "selection.json",
        {
            "selection_metric": "val_macro_f1",
            "tie_break": ["lower_validation_loss", "earlier_epoch"],
            "selected_stage": selected_stage,
            "selected": selected_result.best_record,
            "stage_a": result_a.best_record,
            "stage_b": result_b.best_record,
        },
    )
    evaluation = evaluate_torch(
        context,
        model,
        val_rows,
        device,
        batch_size=int(
            stage_b["batch_size"] if stage_b_wins else stage_a["batch_size"]
        ),
    )
    _write_validation_outputs(context, layout, evaluation, val_rows)
    finalize_checksums(layout)
    return layout.pipeline_root
