from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import ProjectContext
from .determinism import DETERMINISTIC_ENVIRONMENT, runtime_device_record
from .hashing import combined_hash, sha256_file, source_tree_hash
from .io_utils import load_json, write_checksums, write_json


@dataclass(frozen=True)
class ExperimentLayout:
    experiment_id: str
    pipeline_id: str
    experiment_root: Path
    pipeline_root: Path

    @property
    def predictions(self) -> Path:
        return self.pipeline_root / "predictions"

    @property
    def metrics(self) -> Path:
        return self.pipeline_root / "metrics"

    @property
    def plots(self) -> Path:
        return self.pipeline_root / "plots"


def code_state(context: ProjectContext) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=context.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=context.root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"kind": "git", "revision": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {
            "kind": "source_tree_sha256",
            "revision": source_tree_hash(context.root),
            "dirty": False,
        }


def experiment_layout(context: ProjectContext, pipeline_id: str) -> ExperimentLayout:
    manifest_hash = sha256_file(context.manifest_path)
    revision = code_state(context)["revision"]
    experiment_id = combined_hash(
        (
            manifest_hash,
            context.config_hash,
            revision,
            pipeline_id,
            str(context.config["seed"]),
        )
    )[:16]
    root = context.root / context.config["artifacts"]["root"] / experiment_id
    layout = ExperimentLayout(
        experiment_id=experiment_id,
        pipeline_id=pipeline_id,
        experiment_root=root,
        pipeline_root=root / pipeline_id,
    )
    for path in (layout.pipeline_root, layout.predictions, layout.metrics, layout.plots):
        path.mkdir(parents=True, exist_ok=True)
    return layout


def package_inventory() -> list[dict[str, str]]:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or distribution.name
        packages[name.lower().replace("_", "-")] = distribution.version
    return [{"name": name, "version": packages[name]} for name in sorted(packages)]


def write_common_artifacts(
    context: ProjectContext,
    layout: ExperimentLayout,
    *,
    device: Any,
    command: Iterable[str],
    pretrained_weights: dict[str, Any] | None = None,
) -> None:
    state = code_state(context)
    manifest_lock = load_json(context.manifest_path.parent / "manifest_lock.json")
    dataset_lock = load_json(context.manifest_path.parent / "dataset_lock.json")
    write_json(layout.pipeline_root / "resolved_config.json", context.config)
    provenance = {
        "schema_version": "1.0.0",
        "spec_version": context.config["spec_version"],
        "project_id": context.config["project_id"],
        "pipeline_id": layout.pipeline_id,
        "experiment_id": layout.experiment_id,
        "source": dataset_lock["locked_payload"],
        "manifest_sha256": manifest_lock["manifest_sha256"],
        "classes_sha256": context.classes_hash,
        "config_sha256": context.config_hash,
        "code": state,
        "python_executable": sys.executable,
        "packages": package_inventory(),
        "deterministic_environment": DETERMINISTIC_ENVIRONMENT,
        "device": runtime_device_record(device),
        "pretrained_weights": pretrained_weights,
        "command": list(command),
    }
    write_json(layout.pipeline_root / "provenance.json", provenance)
    runtime = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "username": os.environ.get("USERNAME") or os.environ.get("USER"),
    }
    write_json(layout.pipeline_root / "runtime_metadata.json", runtime)


def finalize_checksums(layout: ExperimentLayout) -> None:
    excluded = {"checksums.sha256", "runtime_metadata.json"}
    paths = [
        path
        for path in layout.pipeline_root.rglob("*")
        if path.is_file()
        and path.name not in excluded
        and "checkpoints" not in path.parts
    ]
    write_checksums(
        layout.pipeline_root,
        paths,
        layout.pipeline_root / "checksums.sha256",
    )
