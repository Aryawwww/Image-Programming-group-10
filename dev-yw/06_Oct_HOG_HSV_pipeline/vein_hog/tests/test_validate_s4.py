import json
from pathlib import Path

import pytest

from vein_hog.validate_s4 import require_frozen_winner


def test_validation_source_does_not_choose_or_cache() -> None:
    import vein_hog.validate_s4 as module

    text = Path(module.__file__).read_text(encoding="utf-8")
    assert "choose_winner(" not in text
    assert "extract_split" not in text
    assert "disease_risk" not in text
    assert "np.savez" not in text
    assert "validation leaked" in text


def test_only_the_frozen_s4_winner_can_be_scored() -> None:
    require_frozen_winner(
        {
            "search": "S4",
            "replaces_report_round": False,
            "winner_round": 99,
            "hog": "ori12-cell16",
            "hsv_alpha": 0.45,
            "orientations": 12,
            "pixels_per_cell": 16,
        }
    )
    with pytest.raises(RuntimeError, match="round 99"):
        require_frozen_winner(
            {
                "search": "S4",
                "replaces_report_round": False,
                "winner_round": 10,
                "hog": "ori9-cell8",
                "hsv_alpha": 0.25,
                "orientations": 9,
                "pixels_per_cell": 8,
            }
        )


def test_winner_file_on_disk_is_the_frozen_round() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "outputs" / "search_s4" / "winner.json").read_text(encoding="utf-8"))
    require_frozen_winner(payload)
    assert payload["validation_was_read"] is False
