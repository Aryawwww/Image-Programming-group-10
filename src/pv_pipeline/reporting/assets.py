from __future__ import annotations

import csv
from pathlib import Path

from ..config import ProjectContext
from ..io_utils import load_json, write_bytes


def build_report_assets(context: ProjectContext) -> Path:
    comparison = context.root / "artifacts" / "comparison"
    table_path = comparison / "model_comparison.csv"
    bootstrap_path = comparison / "pairwise_bootstrap.json"
    if not table_path.is_file() or not bootstrap_path.is_file():
        raise FileNotFoundError("Run compare before building report assets")
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    bootstrap = load_json(bootstrap_path)
    lines = [
        "# Canonical Experiment Tables",
        "",
        "## Clean Test Performance",
        "",
        "| Pipeline | Accuracy | Macro precision | Macro recall | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {pipeline_id} | {accuracy:.4f} | {macro_precision:.4f} | "
            "{macro_recall:.4f} | {macro_f1:.4f} |".format(
                pipeline_id=row["pipeline_id"],
                accuracy=float(row["accuracy"]),
                macro_precision=float(row["macro_precision"]),
                macro_recall=float(row["macro_recall"]),
                macro_f1=float(row["macro_f1"]),
            )
        )
    lines.extend(("", "## Pairwise Conclusions", ""))
    for pair, result in bootstrap["conclusions"].items():
        lines.append(
            f"- `{pair}`: {result['conclusion']} "
            f"(point difference {float(result['point_difference']):.4f})."
        )
    lines.extend(
        (
            "",
            "> Scope: these results compare algorithms on the controlled PlantVillage subset; "
            "they do not establish field-diagnostic performance.",
            "",
        )
    )
    output = comparison / "report_tables.md"
    write_bytes(output, "\n".join(lines).encode("utf-8"))
    return output
