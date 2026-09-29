from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .hashing import canonical_json_bytes, sha256_file


def ensure_finite_json(value: Any, location: str = "root") -> None:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"Non-finite float at {location}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            ensure_finite_json(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_finite_json(item, f"{location}[{index}]")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_bytes(path: Path, payload: bytes) -> None:
    _atomic_write(path, payload)


def write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    ensure_finite_json(value)
    if pretty:
        payload = (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    else:
        payload = canonical_json_bytes(value) + b"\n"
    _atomic_write(path, payload)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = []
    for row in rows:
        ensure_finite_json(row)
        lines.append(canonical_json_bytes(row))
    _atomic_write(path, b"\n".join(lines) + b"\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"Blank JSONL line at {path}:{line_number}")
        rows.append(json.loads(line))
    return rows


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_checksums(root: Path, paths: Iterable[Path], output: Path) -> None:
    records = []
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        records.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    _atomic_write(output, ("\n".join(records) + "\n").encode("ascii"))


def verify_checksums(root: Path, checksum_file: Path) -> list[str]:
    errors = []
    for line_number, raw_line in enumerate(
        checksum_file.read_text(encoding="ascii").splitlines(), 1
    ):
        expected, separator, relative = raw_line.partition("  ")
        if not separator:
            errors.append(f"Malformed checksum line {line_number}")
            continue
        path = root / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            errors.append(f"Checksum mismatch: {relative}")
    return errors
