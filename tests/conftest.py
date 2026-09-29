from __future__ import annotations

import os
from pathlib import Path

import pytest


for key, value in {
    "PYTHONHASHSEED": "42177",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}.items():
    os.environ[key] = value


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def context(project_root: Path):
    from pv_pipeline.config import load_project

    return load_project(project_root / "configs" / "project_config.json")
