import json
from pathlib import Path

import numpy as np
import pytest

from vein_hog.fit_rounds import REPORT_ALPHA
from vein_hog.hsv_window import (
    HSV_ALPHA_CAP,
    HSV_ALPHA_FLOOR,
    HSVWindowController,
    WINDOW,
    control_strength,
    leaf_contrast,
    load_training_alpha,
    require_hsv_alpha,
    training_frame,
    window_mean_saturation,
    write_strength,
)


def test_floor_is_the_prespecified_report_weight() -> None:
    assert HSV_ALPHA_FLOOR == REPORT_ALPHA == 0.25
    assert HSV_ALPHA_CAP == 1.0


def test_weight_below_the_floor_is_rejected() -> None:
    with pytest.raises(ValueError, match="below the floor"):
        require_hsv_alpha(0.0)
    with pytest.raises(ValueError, match="below the floor"):
        require_hsv_alpha(0.249)
    assert require_hsv_alpha(0.25) == 0.25
    assert require_hsv_alpha(1.0) == 1.0
    with pytest.raises(ValueError, match="above the cap"):
        require_hsv_alpha(1.01)


def test_uniform_windows_stay_on_the_floor() -> None:
    means = np.full(9, 80.0)
    assert leaf_contrast(means) == 0.0
    assert control_strength(np.zeros(4)) == HSV_ALPHA_FLOOR


def test_full_spread_reaches_the_cap_and_median_not_the_max() -> None:
    assert control_strength(np.ones(3)) == HSV_ALPHA_CAP
    # One extreme leaf must not set the weight. The median contrast is 0.2.
    strength = control_strength(np.array([0.0, 0.2, 1.0]))
    assert strength == pytest.approx(HSV_ALPHA_FLOOR + 0.75 * 0.2)


def test_background_pixels_do_not_enter_the_window_mean() -> None:
    saturation = np.zeros((4, 4), dtype=np.float64)
    saturation[:, 2:] = 200.0
    leaf = np.zeros((4, 4), dtype=np.uint8)
    leaf[:, :2] = 255
    means = window_mean_saturation(saturation, leaf, window=2, stride=2, min_leaf_fraction=0.5)
    # Right-hand windows are background. Left-hand leaf windows are saturation 0.
    assert means.size == 2
    assert np.all(means == 0.0)


def test_split_leaf_contrast_uses_only_kept_windows() -> None:
    saturation = np.zeros((4, 4), dtype=np.float64)
    saturation[:, 2:] = 255.0
    leaf = np.full((4, 4), 255, dtype=np.uint8)
    means = window_mean_saturation(saturation, leaf, window=2, stride=2)
    assert means.tolist() == [0.0, 255.0, 0.0, 255.0]
    assert leaf_contrast(means) == pytest.approx(1.0)


def test_one_window_has_no_spread() -> None:
    assert leaf_contrast(np.array([40.0])) == 0.0


def test_block_window_covers_the_hog_lattice() -> None:
    saturation = np.zeros((256, 256), dtype=np.float64)
    leaf = np.full((256, 256), 255, dtype=np.uint8)
    means = window_mean_saturation(saturation, leaf)
    assert means.size == 31 * 31
    assert WINDOW == 16


def test_training_frame_drops_test_and_validation() -> None:
    import pandas as pd

    table = pd.DataFrame(
        {
            "split": ["train", "test", "validation", "train"],
            "local_relpath": ["a", "b", "c", "d"],
        }
    )
    rows = training_frame(table)
    assert rows["split"].tolist() == ["train", "train"]
    assert rows["local_relpath"].tolist() == ["a", "d"]


def test_saved_record_cannot_smuggle_a_weight_under_the_floor(tmp_path: Path) -> None:
    path = tmp_path / "strength.json"
    path.write_text(
        json.dumps({"hsv_alpha": 0.0, "split": "train", "test_was_read": False, "validation_was_read": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="below the floor"):
        load_training_alpha(path)


def test_controller_on_a_uniform_leaf_stays_on_the_floor(tmp_path: Path) -> None:
    # A black border gives the mask an edge. A flat green field has no saturation spread.
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[8:56, 8:56] = (0, 180, 0)
    controller = HSVWindowController()
    assert controller.observe(image) == 0.0
    assert controller.n_windows > 0
    assert controller.strength() == HSV_ALPHA_FLOOR
    payload = write_strength(tmp_path / "strength.json", controller, n_train=1)
    assert payload["hsv_alpha"] == HSV_ALPHA_FLOOR
    assert payload["test_was_read"] is False
    assert payload["validation_was_read"] is False
    assert payload["replaces_report_round"] is False
    assert load_training_alpha(tmp_path / "strength.json") == HSV_ALPHA_FLOOR
