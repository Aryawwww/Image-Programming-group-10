import numpy as np
import pandas as pd

from vein_hog.search_s3 import choose_winner, iter_s3, sigma_label, summarize_bgr


def test_s3_is_twenty_frangi_settings() -> None:
    specs = iter_s3()
    assert len(specs) == 20
    assert [item["round"] for item in specs] == list(range(1, 21))
    assert {item["source"] for item in specs} == {"green", "exg"}
    assert {item["polarity"] for item in specs} == {"dark", "pale"}
    assert sigma_label((1.0, 2.0, 3.0)) == "1-2-3"
    dark_green = [
        item
        for item in specs
        if item["source"] == "green" and item["polarity"] == "dark" and item["sigmas"] == (1.0,)
    ]
    assert len(dark_green) == 1


def test_ridge_summary_has_eighteen_finite_numbers() -> None:
    image = np.zeros((80, 96, 3), dtype=np.uint8)
    image[:, :, 1] = 160
    image[36:44, 10:86] = (30, 30, 30)
    features = summarize_bgr(image)
    assert features.shape == (20, 18)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_s3_winner_prefers_separation_then_fewer_scales() -> None:
    rows = []
    for spec in iter_s3():
        auc = 0.60
        if int(spec["round"]) in (1, 4):
            auc = 0.80
        row = {
            "round": spec["round"],
            "source": spec["source"],
            "polarity": spec["polarity"],
            "sigmas": sigma_label(spec["sigmas"]),
            "n_sigmas": spec["n_sigmas"],
            "fraction": 1.0,
            "subset": "select",
            "min_class_auc": auc,
            "mean_class_auc": auc,
        }
        rows.append(row)
    # Round 1 has one scale. Round 4 has three. The tie keeps round 1.
    assert choose_winner(pd.DataFrame(rows)) == 1
