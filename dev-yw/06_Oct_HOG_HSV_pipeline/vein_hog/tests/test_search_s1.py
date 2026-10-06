import numpy as np
import pandas as pd
from skimage.feature import hog as skimage_hog

from vein_hog.search_s1 import (
    BLOCKS,
    DISEASES,
    POOLS,
    choose_winner,
    class_aucs,
    crop_energy,
    iter_s1,
    reduce_energy,
)


def test_s1_is_twenty_rounds_outside_the_c_alpha_grid() -> None:
    specs = iter_s1()
    assert len(specs) == 20
    assert [item["round"] for item in specs] == list(range(1, 21))
    assert {item["border"] for item in specs} == {1, 2, 3, 4}
    assert [item["pool"] for item in specs[:5]] == list(POOLS)


def test_cached_hog_layout_is_block_cell_orientation() -> None:
    image = np.zeros((256, 256), dtype=np.float32)
    image[40:80, :] = 1.0
    flat = skimage_hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    grid = skimage_hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=False,
    )
    assert flat.shape == (34596,)
    assert grid.shape == (BLOCKS, BLOCKS, 2, 2, 9)
    assert np.allclose(flat, grid.ravel())


def test_crop_keeps_a_positive_interior() -> None:
    energy = np.zeros((2, 31, 31, 9), dtype=np.float32)
    assert crop_energy(energy, 0).shape == (2, 31, 31, 9)
    assert crop_energy(energy, 4).shape == (2, 23, 23, 9)


def test_one_direction_is_anisotropic_and_coherent() -> None:
    energy = np.zeros((1, 4, 4, 9), dtype=np.float32)
    energy[..., 3] = 2.0
    anisotropy = reduce_energy(energy, "anisotropy")
    entropy = reduce_energy(energy, "entropy")
    vein = reduce_energy(energy, "vein")
    assert anisotropy.shape == (1, 2)
    assert anisotropy[0, 0] == 1.0
    assert entropy[0, 0] == 0.0
    assert vein.shape == (1, 6)
    assert vein[0, 4] == 1.0
    assert reduce_energy(energy, "orient_mean").shape == (1, 9)
    assert reduce_energy(energy, "orient_mean_std").shape == (1, 18)


def test_winner_ignores_a_larger_weaker_round() -> None:
    rows = []
    ranked = [(1, 0.70, 18, 1), (2, 0.70, 2, 4), (3, 0.69, 2, 1)]
    ranked.extend((number, 0.50, 9, 2) for number in range(4, 21))
    for number, auc, features, border in ranked:
        row = {
            "round": number,
            "border": border,
            "pool": "anisotropy",
            "n_features": features,
            "fraction": 1.0,
            "subset": "select",
            "min_class_auc": auc,
            "mean_class_auc": auc,
        }
        for name in DISEASES:
            row[f"auc_{name}"] = auc
        rows.append(row)
    assert choose_winner(pd.DataFrame(rows)) == 2


def test_class_auc_is_one_when_healthy_scores_are_higher() -> None:
    labels = np.array(["Tomato_healthy", "Tomato_healthy", *DISEASES])
    score = np.array([2.0, 1.5, 0.0, -1.0, -0.2, 3.0])
    found = class_aucs(labels, score)
    assert found["Tomato_Late_blight"] == 1.0
    assert min(found.values()) == found["Tomato__Tomato_YellowLeaf__Curl_Virus"]
