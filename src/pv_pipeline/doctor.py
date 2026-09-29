from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import torch

from .config import ProjectContext
from .data.acquire import verify_dataset_lock
from .determinism import assert_process_environment, resolve_device, runtime_device_record
from .hashing import sha256_file, source_tree_hash
from .io_utils import load_json, write_json


def run_doctor(context: ProjectContext, device_name: str) -> dict[str, Any]:
    checks = []

    def check(name: str, action: Any) -> None:
        try:
            detail = action()
            checks.append({"name": name, "passed": True, "detail": detail})
        except Exception as exc:
            checks.append({"name": name, "passed": False, "detail": repr(exc)})

    check("deterministic_process_environment", lambda: assert_process_environment() or "active")
    check("project_config", lambda: context.config_hash)
    check("classes", lambda: context.classes_hash)
    check("source_tree", lambda: source_tree_hash(context.root))
    check("dataset_lock", lambda: verify_dataset_lock(context)["locked_payload_sha256"])
    check(
        "manifest_lock",
        lambda: load_json(context.manifest_path.parent / "manifest_lock.json")[
            "manifest_sha256"
        ],
    )
    check(
        "environment_report",
        lambda: load_json(context.root / "artifacts" / "environment" / "environment_report.json")[
            "status"
        ],
    )
    device = resolve_device(device_name)
    check("device", lambda: runtime_device_record(device))
    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "checks": checks,
    }
    write_json(context.root / "artifacts" / "doctor.json", report)
    return report
