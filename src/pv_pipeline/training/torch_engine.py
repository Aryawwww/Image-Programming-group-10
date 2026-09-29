from __future__ import annotations

import os
import random
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import ProjectContext
from ..data.dataset import EpochHashSampler, ManifestDataset
from ..evaluation.metrics import classification_metrics
from ..hashing import canonical_json_hash
from ..io_utils import write_csv


@dataclass
class TorchEvaluation:
    loss: float
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    metrics: dict[str, Any]


@dataclass
class TrainingResult:
    best_state: OrderedDict[str, torch.Tensor]
    best_record: dict[str, Any]
    history: list[dict[str, Any]]


HISTORY_FIELDS = (
    "stage",
    "epoch",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_macro_precision",
    "val_macro_recall",
    "val_macro_f1",
    "learning_rates",
)


def clone_state(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in sorted(model.state_dict().items())
    )


def selection_is_better(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    epsilon: float,
) -> bool:
    if incumbent is None:
        return True
    score_difference = float(candidate["val_macro_f1"]) - float(incumbent["val_macro_f1"])
    if score_difference > epsilon:
        return True
    if abs(score_difference) <= epsilon:
        loss_difference = float(candidate["val_loss"]) - float(incumbent["val_loss"])
        if loss_difference < -epsilon:
            return True
        if abs(loss_difference) <= epsilon:
            return int(candidate["epoch"]) < int(incumbent["epoch"])
    return False


def evaluate_torch(
    context: ProjectContext,
    model: nn.Module,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    batch_size: int,
    image_loader: Callable[[dict[str, Any]], Any] | None = None,
) -> TorchEvaluation:
    dataset = ManifestDataset(
        context,
        rows,
        training=False,
        image_loader=image_loader,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    logits_parts = []
    labels_parts = []
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for inputs, labels, _sample_ids in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            if logits.shape != (inputs.shape[0], 4) or not torch.isfinite(logits).all():
                raise ValueError(f"Invalid logits during evaluation: {tuple(logits.shape)}")
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * int(labels.shape[0])
            logits_parts.append(logits.detach().cpu())
            labels_parts.append(labels.detach().cpu())
    logits_array = torch.cat(logits_parts).numpy()
    labels_array = torch.cat(labels_parts).numpy().astype(np.int64, copy=False)
    probabilities = torch.softmax(torch.from_numpy(logits_array), dim=1).numpy()
    predicted = logits_array.argmax(axis=1).astype(np.int64, copy=False)
    metrics = classification_metrics(labels_array, predicted, context.classes)
    return TorchEvaluation(
        loss=total_loss / len(rows),
        y_true=labels_array,
        y_pred=predicted,
        probabilities=probabilities,
        metrics=metrics,
    )


def train_epochs(
    *,
    context: ProjectContext,
    model: nn.Module,
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    device: torch.device,
    stage: str,
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    gradient_clip_norm: float,
    history_path: Path,
    after_train_mode: Callable[[nn.Module], None] | None = None,
) -> TrainingResult:
    train_dataset = ManifestDataset(context, train_rows, training=True)
    sampler = EpochHashSampler(train_rows, int(context.config["seed"]))
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    epsilon = float(context.config["model_selection"]["score_comparison_epsilon"])
    best_record: dict[str, Any] | None = None
    best_state: OrderedDict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    checkpoint_path = history_path.parent / "checkpoints" / f"{stage}.pt"
    checkpoint_identity = {
        "stage": stage,
        "epochs": int(epochs),
        "config_sha256": context.config_hash,
        "train_sample_ids_sha256": canonical_json_hash(
            [row["sample_id"] for row in train_rows]
        ),
        "val_sample_ids_sha256": canonical_json_hash(
            [row["sample_id"] for row in val_rows]
        ),
    }
    start_epoch = 0
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("identity") != checkpoint_identity:
            raise ValueError(f"Checkpoint identity mismatch: {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = list(checkpoint["history"])
        best_record = checkpoint["best_record"]
        best_state = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in checkpoint["best_state"].items()
        )
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        start_epoch = int(checkpoint["next_epoch"])
        if len(history) != start_epoch:
            raise ValueError(f"Checkpoint history length mismatch: {checkpoint_path}")

    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        model.train()
        if after_train_mode is not None:
            after_train_mode(model)
        train_loss_sum = 0.0
        train_samples = 0
        for inputs, labels, _sample_ids in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if logits.shape != (inputs.shape[0], 4) or not torch.isfinite(logits).all():
                raise ValueError(f"Invalid training logits: {tuple(logits.shape)}")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(labels.shape[0])
            train_samples += int(labels.shape[0])
        scheduler.step()

        validation = evaluate_torch(
            context,
            model,
            val_rows,
            device,
            batch_size=batch_size,
        )
        record = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": train_loss_sum / train_samples,
            "val_loss": validation.loss,
            "val_accuracy": validation.metrics["accuracy"],
            "val_macro_precision": validation.metrics["macro_precision"],
            "val_macro_recall": validation.metrics["macro_recall"],
            "val_macro_f1": validation.metrics["macro_f1"],
            "learning_rates": ";".join(
                format(float(group["lr"]), ".17g") for group in optimizer.param_groups
            ),
        }
        history.append(record)
        write_csv(history_path, history, HISTORY_FIELDS)
        if selection_is_better(record, best_record, epsilon):
            best_record = dict(record)
            best_state = clone_state(model)
        if best_record is None or best_state is None:
            raise RuntimeError("Training did not create a best checkpoint")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_checkpoint = checkpoint_path.with_name(
            f".{checkpoint_path.name}.{os.getpid()}.tmp"
        )
        torch.save(
            {
                "identity": checkpoint_identity,
                "next_epoch": epoch + 1,
                "model_state": clone_state(model),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "best_record": best_record,
                "best_state": best_state,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
            temporary_checkpoint,
        )
        os.replace(temporary_checkpoint, checkpoint_path)
    if best_record is None or best_state is None:
        raise RuntimeError("Training produced no candidate checkpoint")
    return TrainingResult(best_state=best_state, best_record=best_record, history=history)
