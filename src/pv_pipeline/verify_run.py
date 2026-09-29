from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .artifacts import experiment_layout
from .config import ProjectContext
from .evaluation.runner import PIPELINE_IDS, verify_test_gate
from .io_utils import load_json, read_jsonl, verify_checksums, write_json


def verify_complete_run(context: ProjectContext) -> dict[str, Any]:
    errors = []
    try:
        verify_test_gate(context)
    except Exception as exc:
        errors.append(f"test_gate: {exc}")
    expected_samples = None
    for pipeline_id in PIPELINE_IDS:
        layout = experiment_layout(context, pipeline_id)
        required = [
            "resolved_config.json",
            "provenance.json",
            "model_config.json",
            "selection.json",
            "history.csv",
            "predictions/val.jsonl",
            "predictions/test.jsonl",
            "metrics/val.json",
            "metrics/test.json",
            "metrics/robustness.json",
            "checksums.sha256",
        ]
        for relative in required:
            if not (layout.pipeline_root / relative).is_file():
                errors.append(f"{pipeline_id}: missing {relative}")
        checksum_file = layout.pipeline_root / "checksums.sha256"
        if checksum_file.is_file():
            errors.extend(
                f"{pipeline_id}: {error}"
                for error in verify_checksums(layout.pipeline_root, checksum_file)
            )
        test_predictions = layout.predictions / "test.jsonl"
        if test_predictions.is_file():
            rows = read_jsonl(test_predictions)
            if len(rows) != 540:
                errors.append(f"{pipeline_id}: expected 540 test predictions")
            sample_ids = [row["sample_id"] for row in rows]
            if expected_samples is None:
                expected_samples = sample_ids
            elif sample_ids != expected_samples:
                errors.append(f"{pipeline_id}: test sample order differs")
    comparison_root = context.root / "artifacts" / "comparison"
    for name in (
        "model_comparison.csv",
        "pairwise_bootstrap.json",
        "comparison_confusion_matrices.png",
        "robustness_summary.csv",
        "error_cases.jsonl",
    ):
        if not (comparison_root / name).is_file():
            errors.append(f"comparison: missing {name}")
    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "errors": errors,
    }
    write_json(context.root / "artifacts" / "verify_run.json", report)
    if errors:
        raise RuntimeError(f"Run verification failed with {len(errors)} errors")
    return report
