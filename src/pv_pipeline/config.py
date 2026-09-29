from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import canonical_json_hash, sha256_file


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ClassInfo:
    class_id: int
    class_name: str
    display_name: str


@dataclass(frozen=True)
class ProjectContext:
    root: Path
    config_path: Path
    config: dict[str, Any]
    classes_path: Path
    classes: tuple[ClassInfo, ...]

    @property
    def config_hash(self) -> str:
        return canonical_json_hash(self.config)

    @property
    def classes_hash(self) -> str:
        return sha256_file(self.classes_path)

    @property
    def class_by_name(self) -> dict[str, ClassInfo]:
        return {item.class_name: item for item in self.classes}

    @property
    def class_by_id(self) -> dict[int, ClassInfo]:
        return {item.class_id: item for item in self.classes}

    def resolve(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ConfigError(f"Configured path escapes project root: {relative_path}") from exc
        return candidate

    @property
    def manifest_path(self) -> Path:
        return self.resolve(self.config["dataset"]["canonical_manifest"])

    @property
    def source_snapshot(self) -> Path:
        return self.root / "data" / "source_snapshot"

    @property
    def extracted_root(self) -> Path:
        return self.root / "data" / "extracted"


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigError(f"Could not locate pyproject.toml from {start}")


def _validate(context: ProjectContext) -> None:
    config = context.config
    if config.get("spec_version") != "1.0.0":
        raise ConfigError("Unsupported spec_version")
    if config.get("seed") != 42177:
        raise ConfigError("Canonical seed must be 42177")
    ids = [item.class_id for item in context.classes]
    if ids != [0, 1, 2, 3]:
        raise ConfigError(f"Class IDs must be [0, 1, 2, 3], found {ids}")
    names = [item.class_name for item in context.classes]
    if len(set(names)) != 4:
        raise ConfigError("Class names must be unique")
    counts = config.get("dataset", {}).get("selected_count_per_class", {})
    if counts != {"train": 630, "val": 135, "test": 135}:
        raise ConfigError(f"Unexpected canonical split counts: {counts}")
    labels = config.get("evaluation", {}).get("labels")
    if labels != ids:
        raise ConfigError("Evaluation labels must match class IDs")
    if config.get("small_cnn", {}).get("architecture", {}).get("output_classes") != 4:
        raise ConfigError("Small CNN output size must be 4")
    if config.get("mobilenet_v3", {}).get("output_classes") != 4:
        raise ConfigError("MobileNet output size must be 4")
    exclusions = config.get("dataset", {}).get("holdout_exclusions", [])
    if not isinstance(exclusions, list):
        raise ConfigError("dataset.holdout_exclusions must be a list")
    exclusion_paths = []
    for item in exclusions:
        if not isinstance(item, dict):
            raise ConfigError("Each holdout exclusion must be an object")
        relative_path = item.get("relative_path")
        reason = item.get("reason")
        if not isinstance(relative_path, str) or not relative_path.startswith("raw/color/"):
            raise ConfigError("Holdout exclusion paths must start with raw/color/")
        if not isinstance(reason, str) or not reason.strip():
            raise ConfigError("Each holdout exclusion must include a reason")
        exclusion_paths.append(relative_path)
    if len(exclusion_paths) != len(set(exclusion_paths)):
        raise ConfigError("Holdout exclusion paths must be unique")
    if config.get("hog_svm", {}).get("classifier", {}).get("dual") is not True:
        raise ConfigError(
            "Canonical LinearSVC must use dual=true because training samples "
            "are fewer than HOG features"
        )


def load_project(config_path: str | Path) -> ProjectContext:
    supplied = Path(config_path)
    root = find_project_root(supplied if supplied.is_absolute() else Path.cwd())
    absolute_config = supplied if supplied.is_absolute() else root / supplied
    absolute_config = absolute_config.resolve()
    config = json.loads(absolute_config.read_text(encoding="utf-8"))
    classes_path = (root / config["dataset"]["classes_file"]).resolve()
    class_document = json.loads(classes_path.read_text(encoding="utf-8"))
    classes = tuple(
        ClassInfo(
            class_id=int(row["class_id"]),
            class_name=str(row["class_name"]),
            display_name=str(row["display_name"]),
        )
        for row in class_document["classes"]
    )
    context = ProjectContext(
        root=root,
        config_path=absolute_config,
        config=config,
        classes_path=classes_path,
        classes=classes,
    )
    _validate(context)
    return context
