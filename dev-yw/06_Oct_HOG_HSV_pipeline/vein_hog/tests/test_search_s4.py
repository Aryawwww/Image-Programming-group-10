from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vein_hog.hog import HOG_DIM
from vein_hog.hsv_window import HSV_ALPHA_FLOOR
from vein_hog.search_s4 import (
    choose_winner,
    descriptor_length,
    fit_healthy_score,
    hog_vector,
    hsv_strengths,
    iter_s4,
    reference_alpha,
)

REFERENCE = 0.4031262697503271


def test_s4_source_does_not_name_a_ridge_filter() -> None:
    import vein_hog.search_s4 as module

    text = Path(module.__file__).read_text(encoding="utf-8").lower()
    assert "frangi" not in text


def test_s4_steps_hsv_from_the_floor_to_one_half() -> None:
    strengths = hsv_strengths()
    assert len(strengths) == 26
    assert strengths[0] == 0.25
    assert strengths[-1] == 0.50
    assert np.allclose(np.diff(strengths), 0.01)
    assert max(strengths) <= 0.50
    specs = iter_s4(REFERENCE)
    assert len(specs) == 4 * 26
    assert [item["round"] for item in specs] == list(range(1, 105))
    assert [item["hog"] for item in specs[:26]] == ["ori9-cell8"] * 26
    alphas = [float(item["hsv_alpha"]) for item in specs[:26]]
    assert alphas == list(strengths)
    assert 0.75 not in alphas
    assert 1.0 not in alphas
    assert descriptor_length(9, 8) == HOG_DIM == 34596
    assert int(specs[0]["hog_dim"]) == 34596
    assert int(specs[26]["hog_dim"]) == descriptor_length(9, 16)
    with pytest.raises(ValueError, match="above the S4 cap"):
        iter_s4(0.75)


def test_strength_below_the_floor_is_not_a_round() -> None:
    with pytest.raises(ValueError, match="below the floor"):
        iter_s4(0.1)


def test_winner_prefers_separation_then_a_shorter_hog() -> None:
    rows = []
    tied = 0
    shorter = next(
        spec
        for spec in iter_s4(REFERENCE)
        if spec["hog"] == "ori6-cell16" and float(spec["hsv_alpha"]) == 0.40
    )
    longer = next(
        spec for spec in iter_s4(REFERENCE) if spec["hog"] == "ori9-cell8" and float(spec["hsv_alpha"]) == 0.50
    )
    for spec in iter_s4(REFERENCE):
        base = 0.55
        if int(spec["round"]) in {int(shorter["round"]), int(longer["round"])}:
            base = 0.90
            tied += 1
        rows.append(
            {
                "round": spec["round"],
                "hog_dim": spec["hog_dim"],
                "hsv_alpha": spec["hsv_alpha"],
                "fraction": 1.0,
                "subset": "select",
                "min_class_auc": base,
                "mean_class_auc": base,
            }
        )
    assert tied == 2
    table = pd.DataFrame(rows)
    assert choose_winner(table, REFERENCE) == int(shorter["round"])
    table["subset"] = "test"
    with pytest.raises(RuntimeError, match="before the test"):
        choose_winner(table, REFERENCE)


def test_reference_record_must_be_the_training_split(tmp_path: Path) -> None:
    path = tmp_path / "strength.json"
    path.write_text(
        '{"hsv_alpha": 0.4, "floor": 0.25, "split": "train", "test_was_read": false, "validation_was_read": true}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="training-split"):
        reference_alpha(path)


def test_fit_rejects_a_weight_under_the_floor() -> None:
    hog = np.zeros((8, 3), dtype=np.float32)
    hsv = np.ones((8, 2), dtype=np.float32)
    labels = np.array(["Tomato_healthy"] * 4 + ["Tomato_Bacterial_spot"] * 4)
    with pytest.raises(ValueError, match="below the floor"):
        fit_healthy_score(hog, hsv, labels, np.arange(8), np.arange(8), 0.0)


def test_published_cell_length_on_a_blank_leaf() -> None:
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    vector = hog_vector(image, 9, 8)
    assert vector.shape == (HOG_DIM,)
    assert vector.dtype == np.float32
