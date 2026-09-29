from __future__ import annotations

import argparse
import email
import hashlib
import json
import platform
import re
import sysconfig
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic wheelhouse manifest")
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requirements-output", type=Path, required=True)
    return parser.parse_args()


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_identity(path: Path) -> tuple[str, str]:
    if path.suffix.lower() != ".whl":
        raise ValueError(f"Only wheel files are permitted: {path.name}")
    with zipfile.ZipFile(path) as archive:
        metadata_paths = sorted(
            name
            for name in archive.namelist()
            if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
        )
        if len(metadata_paths) != 1:
            raise ValueError(
                f"Expected one dist-info/METADATA entry in {path.name}, found {len(metadata_paths)}"
            )
        metadata = email.message_from_bytes(archive.read(metadata_paths[0]))
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ValueError(f"Wheel metadata is missing Name or Version: {path.name}")
    return canonical_name(name), version


def main() -> int:
    args = parse_args()
    wheelhouse = args.wheelhouse.resolve()
    files = []
    distributions: dict[str, dict[str, object]] = {}
    for path in sorted(item for item in wheelhouse.iterdir() if item.is_file()):
        name, version = wheel_identity(path)
        file_hash = sha256_file(path)
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_hash,
            }
        )
        existing = distributions.get(name)
        if existing is not None and existing["version"] != version:
            raise ValueError(
                f"Multiple versions for {name}: {existing['version']} and {version}"
            )
        if existing is None:
            existing = {"name": name, "version": version, "hashes": []}
            distributions[name] = existing
        hashes = existing["hashes"]
        assert isinstance(hashes, list)
        hashes.append(file_hash)
    if not files:
        raise SystemExit("Wheelhouse is empty")

    requirement_rows = []
    requirement_lines = [
        "# Generated from the canonical wheelhouse. Do not edit by hand.",
        "# Every transitive dependency is pinned and protected by SHA-256.",
    ]
    for name in sorted(distributions):
        row = distributions[name]
        hashes = sorted(set(str(value) for value in row["hashes"]))
        requirement_rows.append(
            {"name": name, "version": row["version"], "hashes": hashes}
        )
        hash_arguments = " ".join(f"--hash=sha256:{value}" for value in hashes)
        requirement_lines.append(f"{name}=={row['version']} {hash_arguments}")
    requirements_text = "\n".join(requirement_lines) + "\n"
    requirements_sha256 = hashlib.sha256(requirements_text.encode("utf-8")).hexdigest()

    requirements_path = args.requirements_output.resolve()
    requirements_path.parent.mkdir(parents=True, exist_ok=True)
    requirements_path.write_text(
        requirements_text,
        encoding="utf-8",
        newline="\n",
    )

    locked_payload = {
        "platform_tag": sysconfig.get_platform(),
        "python_version": platform.python_version(),
        "files": files,
        "requirements": requirement_rows,
        "requirements_lock_sha256": requirements_sha256,
    }
    locked_sha256 = hashlib.sha256(
        json.dumps(
            locked_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    output = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "locked_payload": locked_payload,
        "locked_payload_sha256": locked_sha256,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "files": len(files),
                "distributions": len(requirement_rows),
                "manifest": str(output_path),
                "requirements": str(requirements_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
