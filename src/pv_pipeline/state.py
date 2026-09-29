from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import ProjectContext
from .io_utils import load_json, write_json


PHASES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9")


def update_run_state(
    context: ProjectContext,
    *,
    phase: str,
    status: str,
    artifacts: Iterable[str | Path],
    command: Iterable[str],
    note: str | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"Unknown run phase: {phase}")
    path = context.root / "artifacts" / "run_state.json"
    if path.is_file():
        state = load_json(path)
    else:
        state = {
            "schema_version": "1.0.0",
            "project_id": context.config["project_id"],
            "phases": {name: {"status": "pending"} for name in PHASES},
        }
    record = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [str(value) for value in artifacts],
        "command": list(command),
    }
    if note is not None:
        record["note"] = note
    state["phases"][phase] = record
    state["current_phase"] = phase
    state["updated_at_utc"] = record["updated_at_utc"]
    write_json(path, state)
    return state
