"""Corrupt one image and return the S6 winner's HOG and HSV rows.

This module is imported by the robustness runner after the winner is frozen.
The search does not import it.
"""

from __future__ import annotations

from pathlib import Path

from corrupt import apply_corruption
from vein_hog.extract import load_bgr
from vein_hog.search_s6 import featurize_leaf


def extract_job(job: tuple[str, str, int, str, int, int, str]) -> tuple:
    path, category, severity, sha256, orientations, pixels_per_cell, layout = job
    image = load_bgr(Path(path))
    corrupted = apply_corruption(image, category, int(severity), sha256)
    return featurize_leaf(corrupted, int(orientations), int(pixels_per_cell), str(layout))
