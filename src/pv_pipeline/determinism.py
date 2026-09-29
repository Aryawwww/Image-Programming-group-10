from __future__ import annotations

import os
import platform
import random
from typing import Any

import numpy as np
import torch


DETERMINISTIC_ENVIRONMENT = {
    "PYTHONHASHSEED": "42177",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def assert_process_environment() -> None:
    mismatches = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in DETERMINISTIC_ENVIRONMENT.items()
        if os.environ.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Deterministic process environment is not active. "
            "Run through scripts/run_deterministic.ps1. "
            f"Mismatches: {mismatches}"
        )


def configure(seed: int = 42177, *, require_environment: bool = True) -> None:
    if require_environment:
        assert_process_environment()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "conv"):
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    else:
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but the active PyTorch profile has no CUDA backend")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device(requested)


def runtime_device_record(device: torch.device) -> dict[str, Any]:
    record: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "backend": device.type,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        record.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return record
