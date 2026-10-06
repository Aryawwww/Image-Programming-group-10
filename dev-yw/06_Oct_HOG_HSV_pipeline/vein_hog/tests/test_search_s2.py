import numpy as np
import pandas as pd

from vein_hog.search_s1 import DISEASES
from vein_hog.search_s2 import choose_winner, iter_s2
from vein_hog.vein import vein_skeleton


def test_s2_is_twenty_skeleton_settings_including_the_published_pair() -> None:
    specs = iter_s2()
    assert len(specs) == 20
    assert [item["round"] for item in specs] == list(range(1, 21))
    published = [item for item in specs if item["percentile"] == 90 and item["open_kernel"] == 5]
    assert len(published) == 1


def test_default_skeleton_is_percentile_90_and_kernel_5() -> None:
    magnitude = np.zeros((48, 48), dtype=np.float32)
    magnitude[10:40, 24] = np.linspace(0.2, 1.0, 30)
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[6:42, 6:42] = 255
    default = vein_skeleton(magnitude, mask)
    explicit = vein_skeleton(magnitude, mask, percentile=90, open_kernel=5)
    assert default.shape == magnitude.shape
    assert np.array_equal(default, explicit)


def test_s2_winner_prefers_separation_then_the_published_pair() -> None:
    rows = []
    for spec in iter_s2():
        auc = 0.60
        if spec["percentile"] == 97 and spec["open_kernel"] == 9:
            auc = 0.70
        if spec["percentile"] == 90 and spec["open_kernel"] == 5:
            auc = 0.70
        row = {
            "round": spec["round"],
            "percentile": spec["percentile"],
            "open_kernel": spec["open_kernel"],
            "subset": "select",
            "min_class_auc": auc,
            "mean_class_auc": auc,
        }
        for name in DISEASES:
            row[f"auc_{name}"] = auc
        rows.append(row)
    assert choose_winner(pd.DataFrame(rows)) == 10
