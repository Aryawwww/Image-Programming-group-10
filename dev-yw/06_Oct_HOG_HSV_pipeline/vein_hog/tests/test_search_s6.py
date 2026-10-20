from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vein_hog.hsv import HSV_DIM
from vein_hog.search_s6 import (
    CLASS_REPORT,
    LAYOUTS,
    METRIC_LABELS,
    TRAIN_ROWS,
    assert_train_paths,
    channel_blocks,
    choose_winner,
    classification_metrics,
    iter_s6,
    layout_dim,
    per_class_metrics,
)

REFERENCE = 0.4031262697503271


def _train_paths() -> list[str]:
    return [
        rf"E:\proj\dev-yw-main-ref\dataset\train\cls\img_{index}.jpg" for index in range(TRAIN_ROWS)
    ]


def test_s6_source_does_not_open_held_out_caches() -> None:
    import vein_hog.search_s6 as module

    text = Path(module.__file__).read_text(encoding="utf-8").lower()
    assert "frangi" not in text
    assert "test_features" not in text
    assert "test_meta" not in text
    assert "validation_features" not in text
    assert "score_winner_on_test" not in text


def test_grid_is_384_rounds_inside_the_hsv_cap() -> None:
    specs = iter_s6(REFERENCE)
    assert len(specs) == 12 * 8 * 4 == 384
    assert [int(item["round"]) for item in specs] == list(range(1, 385))
    assert specs[0]["hog"] == "ori6-cell8"
    assert specs[0]["hsv_layout"] == "published"
    assert float(specs[0]["hsv_alpha"]) == 0.25
    assert layout_dim("published") == HSV_DIM == 27
    assert layout_dim("H") == 17
    assert layout_dim("S") == 9
    assert layout_dim("V") == 9
    assert layout_dim("HS") == 26
    assert layout_dim("HV") == 26
    assert layout_dim("SV") == 18
    assert layout_dim("HSV") == 35
    assert list(LAYOUTS) == ["published", "H", "S", "V", "HS", "HV", "SV", "HSV"]
    assert max(float(item["hsv_alpha"]) for item in specs) <= 0.50
    with pytest.raises(ValueError, match="S6 cap"):
        iter_s6(0.75)


def test_published_channels_match_the_auxiliary_block() -> None:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 40
    image[8:36, 10:50] = (20, 180, 30)
    blocks = channel_blocks(image)
    assert blocks["h_hist"].shape == (16,)
    assert blocks["v_hist"].shape == (8,)
    published = np.concatenate(
        [blocks["h_hist"], blocks["s_hist"], blocks["h_mean"], blocks["s_mean"], blocks["v_mean"]]
    )
    assert published.shape == (HSV_DIM,)


def test_train_paths_reject_held_out_segments() -> None:
    paths = _train_paths()
    assert_train_paths(paths)
    leaked = list(paths)
    leaked[4] = rf"E:\proj\dev-yw-main-ref\dataset\validation\cls\img.jpg"
    with pytest.raises(RuntimeError, match="held-out"):
        assert_train_paths(leaked)
    leaked[4] = rf"E:\proj\dev-yw-main-ref\dataset\test\cls\img.jpg"
    with pytest.raises(RuntimeError, match="held-out"):
        assert_train_paths(leaked)


def test_four_metrics_name_accuracy_and_macro_scores() -> None:
    perfect = classification_metrics(list(METRIC_LABELS), list(METRIC_LABELS))
    assert list(perfect) == ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-score"]
    assert perfect["Accuracy"] == 1.0
    assert perfect["Macro F1-score"] == 1.0
    swapped = list(METRIC_LABELS)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    mixed = classification_metrics(list(METRIC_LABELS), swapped)
    assert mixed["Accuracy"] < 1.0
    assert mixed["Macro F1-score"] < 1.0


def test_per_class_precision_recall_and_f1_use_the_requested_names() -> None:
    rows = per_class_metrics(list(METRIC_LABELS), list(METRIC_LABELS))
    assert [row["Class"] for row in rows] == [name for _key, name in CLASS_REPORT]
    assert [row["Class"] for row in rows] == [
        "Bacterial Spot",
        "Late Blight",
        "Septoria Leaf Spot",
        "Tomato Yellow Leaf Curl Virus",
        "Healthy",
    ]
    assert all(row["Precision"] == 1.0 and row["Recall"] == 1.0 and row["F1-score"] == 1.0 for row in rows)


def test_winner_prefers_separation_then_a_shorter_descriptor() -> None:
    specs = iter_s6(REFERENCE)
    shorter = next(
        spec for spec in specs if spec["hog"] == "ori6-cell32" and spec["hsv_layout"] == "HSV" and float(spec["hsv_alpha"]) == 0.50
    )
    longer = next(
        spec for spec in specs if spec["hog"] == "ori18-cell8" and spec["hsv_layout"] == "S" and float(spec["hsv_alpha"]) == 0.40
    )
    rows = []
    for spec in specs:
        base = 0.55
        if int(spec["round"]) in {int(shorter["round"]), int(longer["round"])}:
            base = 0.90
        rows.append(
            {
                "round": spec["round"],
                "hog_dim": spec["hog_dim"],
                "hsv_dim": spec["hsv_dim"],
                "hsv_alpha": spec["hsv_alpha"],
                "fraction": 1.0,
                "subset": "select",
                "min_class_auc": base,
                "mean_class_auc": base,
            }
        )
    table = pd.DataFrame(rows)
    assert choose_winner(table, REFERENCE) == int(shorter["round"])
    table["subset"] = "validation"
    with pytest.raises(RuntimeError, match="before the test"):
        choose_winner(table, REFERENCE)
