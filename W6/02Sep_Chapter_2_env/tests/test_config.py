from pathlib import Path

from pvch2.config import load_config
from pvch2.preprocess.pipelines import available_pipelines


def test_version_and_includes():
    cfg = load_config()
    assert cfg["version"] == "0.2.0"
    data = Path(cfg["paths"]["dataset_root"])
    assert data.name == "PlantVillage-Dataset"
    assert (data / "raw" / "color").is_dir()
    assert cfg["preprocess"]["illumination"]["clahe"]["clip_limit"] == 2.0
    assert cfg["preprocess"]["denoise"]["bilateral"]["d"] == 5
    assert cfg["preprocess"]["leaf_mask"]["morphology"]["open_disk_radius"] == 3
    assert cfg["features"]["hog"]["orientations"] == 9


def test_pipelines_are_independent():
    cfg = load_config()
    assert available_pipelines(cfg) == ["colour", "grayscale", "segmented"]
    assert cfg["pipelines"]["colour"]["apply"]["leaf_mask"] is False
    assert cfg["pipelines"]["grayscale"]["apply"]["leaf_mask"] is False
    assert cfg["pipelines"]["segmented"]["apply"]["leaf_mask"] is True
    assert "HOG" in cfg["pipelines"]["not_in_preprocess"]
