"""Run feature caching and the same 16 candidate rounds.

Validation stays closed. The reported round is pre-specified in fit_rounds
and is not the test-set maximum.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vein_hog.extract import DEFAULT_RADIUS, extract_split

REPO = Path(__file__).resolve().parents[3]


def find_project(start: Path) -> Path:
    """Directory that contains the pulled split. The map's paths are not rewritten."""
    for candidate in [start, *start.parents]:
        if (candidate / "dev-yw-main-ref").is_dir():
            return candidate
    raise SystemExit("dev-yw-main-ref was not found; the image split was not moved")


def _gate(repo: Path) -> tuple[int, bool]:
    path = repo / "outputs" / "checks" / "vein_gate.json"
    if not path.exists():
        raise SystemExit("outputs/checks/vein_gate.json is missing; run the overlay check first")
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data.get("radius", DEFAULT_RADIUS)), bool(data.get("passed"))


def _cache_current(cache: Path, split: str, radius: int) -> bool:
    summary_path = cache / f"{split}_summary.json"
    features = cache / f"{split}_features.npz"
    if not summary_path.exists() or not features.exists():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return int(summary.get("interior_radius", -1)) == int(radius)


def main() -> None:
    project = find_project(REPO)
    table = pd.read_csv(REPO / "vein_hog" / "data" / "split_map.csv")
    if set(table["split"]) != {"train", "test", "validation"}:
        raise SystemExit("split map must contain train, test and validation")
    radius, _passed = _gate(REPO)
    cache = REPO / "outputs" / "cache"
    for split in ("train", "test"):
        if not _cache_current(cache, split, radius):
            extract_split(table, project, split, cache, None, radius=radius)
    from vein_hog.fit_rounds import run_fit

    payload = run_fit(cache, REPO / "outputs" / "fit")
    print("step 5 not started: validation stays closed", flush=True)
    if payload.get("eligible_for_validation"):
        raise SystemExit("refusing to open validation")


if __name__ == "__main__":
    main()
