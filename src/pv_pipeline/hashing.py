from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CHUNK_SIZE = 1024 * 1024


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
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


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def namespaced_hash(namespace: str, seed: int, *parts: str) -> str:
    return sha256_text("|".join((namespace, str(seed), *parts)))


def sample_id(relative_path: str, file_sha256: str) -> str:
    return sha256_text(f"pv-a1-v1/sample-id|{relative_path}|{file_sha256}")


def source_tree_hash(project_root: Path) -> str:
    roots = ["src", "scripts", "schemas"]
    files: list[Path] = [project_root / "pyproject.toml"]
    included_suffixes = {".py", ".ps1", ".json"}
    for relative_root in roots:
        root = project_root / relative_root
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in included_suffixes
                and "__pycache__" not in path.parts
            )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(project_root).as_posix()):
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\n")
    return digest.hexdigest()


def combined_hash(parts: Iterable[str]) -> str:
    return sha256_text("|".join(parts))
