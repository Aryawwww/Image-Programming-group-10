"""Overlay the named test leaves and measure how far the skeleton sits from the margin.

This does not decide the gate. It only writes the pictures and the distances.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vein_hog.extract import IMAGE_SIZE, _one, load_bgr
from vein_hog.run import REPO, find_project

STEMS = (
    "Tomato_healthy_img_0",
    "Tomato_healthy_img_103",
    "Tomato_Bacterial_spot_img_1",
    "Tomato_Bacterial_spot_img_105",
    "Tomato_Late_blight_img_100",
    "Tomato_Late_blight_img_101",
    "Tomato_Septoria_leaf_spot_img_102",
    "Tomato_Septoria_leaf_spot_img_112",
    "Tomato__Tomato_YellowLeaf__Curl_Virus_img_10",
    "Tomato__Tomato_YellowLeaf__Curl_Virus_img_104",
)

CLASS_LABEL = {
    "Tomato_healthy": "Healthy",
    "Tomato_Bacterial_spot": "Bacterial spot",
    "Tomato_Late_blight": "Late blight",
    "Tomato_Septoria_leaf_spot": "Septoria leaf spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Yellow leaf curl virus",
}


def _rows(radius: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    project = find_project(REPO)
    table = pd.read_csv(REPO / "vein_hog" / "data" / "split_map.csv")
    table = table[table["split"] == "test"].copy()
    table["stem"] = table["remote_relpath"].map(lambda value: Path(str(value)).stem)
    chosen = table[table["stem"].isin(STEMS)]
    if len(chosen) != len(STEMS):
        missing = sorted(set(STEMS) - set(chosen["stem"]))
        raise SystemExit(f"named test leaves not in the split map: {missing}")
    overlay_dir = REPO / "visualize-seaborn" / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    distances: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for _, row in chosen.iterrows():
        sample = _one(load_bgr(project / str(row["local_relpath"])), radius=radius)
        skeleton = np.asarray(sample["skeleton"])
        mask = np.asarray(sample["mask256"]) > 0
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        on = dist[skeleton]
        label = CLASS_LABEL[str(row["class_name"])]
        for value in on:
            distances.append({"Class": label, "Distance to leaf margin (px)": float(value)})
        overlay = np.asarray(sample["bgr256"]).copy()
        overlay[skeleton] = (255, 255, 255)
        stem = f"{row['split']}_{row['class_name']}_{row['stem']}"
        cv2.imwrite(str(overlay_dir / f"{stem}_skeleton.png"), overlay)
        summaries.append(
            {
                "stem": str(row["stem"]),
                "class_name": str(row["class_name"]),
                "skeleton_pixels": int(skeleton.sum()),
                "median_distance_px": float(np.median(on)) if on.size else 0.0,
                "raw_completeness": float(sample["raw_completeness"]),
                "image_size": IMAGE_SIZE,
            }
        )
    return distances, summaries


def main() -> None:
    import sys

    radius = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    distances, summaries = _rows(radius)
    frame = pd.DataFrame(distances)
    out = REPO / "visualize-seaborn"
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    order = [
        "Healthy",
        "Bacterial spot",
        "Late blight",
        "Septoria leaf spot",
        "Yellow leaf curl virus",
    ]
    if not frame.empty:
        sns.boxplot(
            data=frame,
            x="Class",
            y="Distance to leaf margin (px)",
            order=order,
            color="#1f4e79",
            ax=ax,
        )
    ax.axhline(radius, color="#6b3a3a", linestyle="--", linewidth=1)
    ax.set_title(f"Skeleton distance from the leaf margin, interior radius {radius} px")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out / "08_skeleton_distance_to_leaf_margin.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    payload = {"radius": radius, "leaves": summaries}
    path = REPO / "outputs" / "checks" / f"vein_preview_r{radius}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
