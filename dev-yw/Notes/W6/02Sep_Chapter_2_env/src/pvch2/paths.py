from __future__ import annotations

import os
from pathlib import Path

from pvch2.config import ENV_ROOT, load_config

DATASET_DIRNAME = "PlantVillage-Dataset"
DATASET_ENV_VARS = ("PVCH2_DATASET_ROOT", "PLANTVILLAGE_DATASET")
# Search this env, its parent (Windows layout), and grandparent (Mac W6 layout).
DATASET_SEARCH_UP = 2


def _looks_like_dataset(root: Path) -> bool:
    if not root.is_dir():
        return False
    raw = root / "raw"
    return (raw / "color").is_dir() or (raw / "grayscale").is_dir() or (root / "CITATION.cff").is_file()


def _as_existing_dataset(path: Path, *, relative_to: Path) -> Path | None:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = relative_to / candidate
    if _looks_like_dataset(candidate):
        return candidate.resolve()
    return None


def _search_up(start: Path, name: str, max_up: int) -> Path | None:
    cur = start.resolve()
    for _ in range(max_up + 1):
        found = _as_existing_dataset(cur / name, relative_to=cur)
        if found is not None:
            return found
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return None


def dataset_search_roots(env_root: Path) -> list[Path]:
    """Directories that will be checked for PlantVillage-Dataset: ., .., ../.."""
    env_root = env_root.resolve()
    roots = [env_root]
    cur = env_root
    for _ in range(DATASET_SEARCH_UP):
        parent = cur.parent
        if parent == cur:
            break
        roots.append(parent)
        cur = parent
    return roots


def resolve_dataset_root(cfg: dict | None = None, *, env_root: Path | None = None) -> Path:
    """Locate PlantVillage-Dataset without a machine-specific absolute path.

    Order:
    1. PVCH2_DATASET_ROOT or PLANTVILLAGE_DATASET
    2. configs/paths.yaml dataset_root, if it is not "auto" and exists
    3. PlantVillage-Dataset in ., .., or ../.. of this env (and of cwd)
    """
    cfg = cfg or {}
    env_root = Path(env_root or cfg.get("_env_root") or ENV_ROOT).resolve()

    for var in DATASET_ENV_VARS:
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        found = _as_existing_dataset(Path(raw), relative_to=env_root)
        if found is not None:
            return found
        raise FileNotFoundError(f"{var}={raw!r} is not a PlantVillage dataset directory")

    configured = str((cfg.get("paths") or {}).get("dataset_root") or "").strip()
    if configured and configured.lower() not in {"auto", ""}:
        found = _as_existing_dataset(Path(configured), relative_to=env_root)
        if found is not None:
            return found

    for start in (env_root, Path.cwd()):
        found = _search_up(start, DATASET_DIRNAME, DATASET_SEARCH_UP)
        if found is not None:
            return found

    looked = [str(p / DATASET_DIRNAME) for p in dataset_search_roots(env_root)]
    raise FileNotFoundError(
        "PlantVillage-Dataset not found. Looked in ., .., and ../.. of this env: "
        + "; ".join(looked)
        + ". Set PVCH2_DATASET_ROOT to the dataset directory if it lives elsewhere."
    )


def dataset_root(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    cached = (cfg.get("paths") or {}).get("_resolved_dataset_root")
    if cached:
        return Path(cached)
    root = resolve_dataset_root(cfg)
    cfg.setdefault("paths", {})
    cfg["paths"]["_resolved_dataset_root"] = str(root)
    cfg["paths"]["dataset_root"] = str(root)
    return root


def raw_dir(kind: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    rel = cfg["paths"]["raw"][kind]
    return dataset_root(cfg) / rel


def leaf_map_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return dataset_root(cfg) / cfg["paths"]["leaf_map"]


def output_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    out = Path(cfg["paths"]["output_dir"])
    if not out.is_absolute():
        out = Path(cfg["_env_root"]) / out
    out.mkdir(parents=True, exist_ok=True)
    return out
