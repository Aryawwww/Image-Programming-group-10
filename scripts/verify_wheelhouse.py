from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sysconfig
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the canonical wheelhouse")
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--deployment-lock", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wheelhouse = args.wheelhouse.resolve()
    manifest_path = args.manifest.resolve()
    requirements_path = args.requirements.resolve()
    deployment_lock_path = args.deployment_lock.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    deployment_lock = json.loads(deployment_lock_path.read_text(encoding="utf-8"))
    payload = manifest.get("locked_payload", {})
    errors: list[str] = []

    payload_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if payload_hash != manifest.get("locked_payload_sha256"):
        errors.append("manifest locked_payload SHA-256 mismatch")
    if payload.get("python_version") != platform.python_version():
        errors.append(
            f"Python mismatch: manifest={payload.get('python_version')} current={platform.python_version()}"
        )
    if payload.get("platform_tag") != sysconfig.get_platform():
        errors.append(
            f"Platform mismatch: manifest={payload.get('platform_tag')} current={sysconfig.get_platform()}"
        )

    expected_files = {
        row["name"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
        for row in payload.get("files", [])
    }
    actual_names = sorted(path.name for path in wheelhouse.iterdir() if path.is_file())
    if actual_names != sorted(expected_files):
        errors.append(
            f"wheel file set mismatch: expected={sorted(expected_files)} actual={actual_names}"
        )
    for name in sorted(set(actual_names) & set(expected_files)):
        path = wheelhouse / name
        expected = expected_files[name]
        if path.stat().st_size != expected["bytes"]:
            errors.append(f"size mismatch: {name}")
        if sha256_file(path) != expected["sha256"]:
            errors.append(f"SHA-256 mismatch: {name}")

    requirements_hash = sha256_file(requirements_path)
    if requirements_hash != payload.get("requirements_lock_sha256"):
        errors.append("resolved requirements SHA-256 mismatch")

    expected_wheelhouse = deployment_lock.get("wheelhouse", {})
    canonical_values = {
        "python_version": payload.get("python_version"),
        "platform_tag": payload.get("platform_tag"),
        "file_count": len(payload.get("files", [])),
        "distribution_count": len(payload.get("requirements", [])),
        "locked_payload_sha256": manifest.get("locked_payload_sha256"),
        "requirements_lock_sha256": payload.get("requirements_lock_sha256"),
    }
    for key, actual_value in canonical_values.items():
        if expected_wheelhouse.get(key) != actual_value:
            errors.append(
                f"deployment lock mismatch for wheelhouse.{key}: "
                f"expected={expected_wheelhouse.get(key)} actual={actual_value}"
            )

    result = {
        "status": "pass" if not errors else "fail",
        "files": len(actual_names),
        "distributions": len(payload.get("requirements", [])),
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
