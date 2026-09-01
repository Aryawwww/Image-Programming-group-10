from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ENV_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ENV_ROOT / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_config(name: str = "default.yaml") -> dict[str, Any]:
    """Load a YAML file from configs/. default.yaml merges the include list."""
    master = _read_yaml(CONFIG_DIR / name)
    includes = master.get("include") or []
    merged: dict[str, Any] = {k: v for k, v in master.items() if k != "include"}
    for rel in includes:
        key = Path(rel).stem
        merged[key] = _read_yaml(CONFIG_DIR / rel)
    merged["_env_root"] = ENV_ROOT
    merged["_config_dir"] = CONFIG_DIR
    # Resolve after YAML merge so Windows/Mac copies of this env share one config.
    from pvch2.paths import resolve_dataset_root

    merged.setdefault("paths", {})
    merged["paths"]["dataset_root"] = str(resolve_dataset_root(merged, env_root=ENV_ROOT))
    merged["paths"]["_resolved_dataset_root"] = merged["paths"]["dataset_root"]
    return merged
