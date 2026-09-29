from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from ..config import ProjectContext
from ..hashing import canonical_json_hash, sha256_file
from ..io_utils import load_json, write_json


def _weights_checkpoint(context: ProjectContext) -> Path:
    file_name = Path(context.config["mobilenet_v3"]["weights_url"]).name
    return context.root / "data" / "pretrained" / "hub" / "checkpoints" / file_name


def build_mobilenet(
    context: ProjectContext,
    *,
    pretrained: bool,
) -> tuple[nn.Module, dict[str, Any] | None]:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    lock = None
    if pretrained:
        torch_home = context.root / "data" / "pretrained"
        torch_home.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_home)
    model = mobilenet_v3_small(weights=weights)
    if pretrained:
        checkpoint = _weights_checkpoint(context)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Torchvision did not create expected checkpoint: {checkpoint}")
        payload = {
            "weights_enum": context.config["mobilenet_v3"]["weights_enum"],
            "url": context.config["mobilenet_v3"]["weights_url"],
            "file": checkpoint.relative_to(context.root).as_posix(),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        }
        lock = {
            "schema_version": "1.0.0",
            "locked_payload": payload,
            "locked_payload_sha256": canonical_json_hash(payload),
        }
        lock_path = context.root / "data" / "pretrained" / "mobilenet_v3_small.lock.json"
        if lock_path.is_file() and load_json(lock_path) != lock:
            raise ValueError("Pretrained MobileNetV3 weights differ from the existing lock")
        write_json(lock_path, lock)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(
        in_features,
        int(context.config["mobilenet_v3"]["output_classes"]),
    )
    nn.init.xavier_uniform_(model.classifier[3].weight)
    nn.init.constant_(model.classifier[3].bias, 0.0)
    return model, lock


def freeze_batch_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def configure_stage_a(model: nn.Module) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier[3].parameters():
        parameter.requires_grad = True
    freeze_batch_norm(model)
    return trainable_parameter_names(model)


def configure_stage_b(model: nn.Module, feature_start: int) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.features[feature_start:].parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    freeze_batch_norm(model)
    return trainable_parameter_names(model)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def trainable_parameter_audit(model: nn.Module) -> dict[str, Any]:
    names = trainable_parameter_names(model)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"names": names, "parameter_count": int(count)}
