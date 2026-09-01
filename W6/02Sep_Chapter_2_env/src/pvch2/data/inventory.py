from __future__ import annotations

from collections import Counter
from pathlib import Path

from pvch2.paths import raw_dir

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def list_class_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES
    )


def inventory_raw(kind: str) -> dict:
    root = raw_dir(kind)
    if not root.is_dir():
        raise FileNotFoundError(root)
    per_class: dict[str, int] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        per_class[class_dir.name] = sum(
            1 for p in class_dir.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES
        )
    return {
        "kind": kind,
        "root": str(root),
        "n_classes": len(per_class),
        "n_images": int(sum(per_class.values())),
        "per_class": per_class,
        "min_class": min(per_class.values()) if per_class else 0,
        "max_class": max(per_class.values()) if per_class else 0,
    }


def class_imbalance(per_class: dict[str, int]) -> Counter:
    return Counter(per_class)
