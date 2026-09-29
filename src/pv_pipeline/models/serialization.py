from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def cpu_state_dict(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().cpu().contiguous())
        for name, tensor in sorted(model.state_dict().items())
    )


def save_model(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(cpu_state_dict(model), str(path))


def load_model(model: torch.nn.Module, path: Path, *, device: torch.device) -> None:
    state = load_file(str(path), device=str(device))
    model.load_state_dict(state, strict=True)
