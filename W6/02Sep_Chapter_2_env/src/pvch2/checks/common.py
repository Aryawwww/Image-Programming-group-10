from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pvch2.config import load_config
from pvch2.data.inventory import list_class_images
from pvch2.paths import output_dir, raw_dir


@dataclass
class Item:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class CheckResult:
    section: str
    title: str
    items: list[Item] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.items) and all(i.ok for i in self.items)

    def add(self, ok: bool, name: str, detail: str = "") -> None:
        self.items.append(Item(name, bool(ok), detail))


def sample_color_bgr(cfg: dict | None = None) -> tuple[np.ndarray, Path]:
    cfg = cfg or load_config()
    class_dir = raw_dir("color", cfg) / "Apple___Apple_scab"
    images = list_class_images(class_dir)
    if not images:
        raise FileNotFoundError(class_dir)
    path = images[0]
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr, path


def check_dir(section: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    d = output_dir(cfg) / "checks" / section.replace(".", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_u8(path: Path, img: np.ndarray) -> Path:
    if img.ndim == 2:
        vis = img
        if img.dtype != np.uint8:
            x = img.astype(np.float32)
            x = x - x.min()
            vis = np.clip(x / (x.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(path), vis)
    else:
        cv2.imwrite(str(path), img)
    return path
