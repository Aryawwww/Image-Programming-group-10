from pathlib import Path

from pvch2.paths import DATASET_DIRNAME, resolve_dataset_root


def _fake_dataset(parent: Path) -> Path:
    ds = parent / DATASET_DIRNAME
    (ds / "raw" / "color").mkdir(parents=True)
    (ds / "CITATION.cff").write_text("cff-version: 1.2.0\n")
    return ds.resolve()


def test_finds_dataset_two_levels_up(tmp_path, monkeypatch):
    ds = _fake_dataset(tmp_path)
    env = tmp_path / "W6" / "02Sep_Chapter_2_env"
    env.mkdir(parents=True)
    monkeypatch.chdir(env)
    monkeypatch.delenv("PVCH2_DATASET_ROOT", raising=False)
    monkeypatch.delenv("PLANTVILLAGE_DATASET", raising=False)
    found = resolve_dataset_root({"paths": {"dataset_root": "auto"}}, env_root=env)
    assert found == ds


def test_finds_dataset_one_level_up(tmp_path, monkeypatch):
    ds = _fake_dataset(tmp_path)
    env = tmp_path / "02Sep_Chapter_2_env"
    env.mkdir()
    monkeypatch.chdir(env)
    monkeypatch.delenv("PVCH2_DATASET_ROOT", raising=False)
    monkeypatch.delenv("PLANTVILLAGE_DATASET", raising=False)
    found = resolve_dataset_root({"paths": {"dataset_root": "auto"}}, env_root=env)
    assert found == ds


def test_env_var_wins_over_auto(tmp_path, monkeypatch):
    env = tmp_path / "env"
    env.mkdir()
    _fake_dataset(tmp_path)
    other = _fake_dataset(tmp_path / "elsewhere")
    monkeypatch.chdir(env)
    monkeypatch.setenv("PVCH2_DATASET_ROOT", str(other))
    found = resolve_dataset_root({"paths": {"dataset_root": "auto"}}, env_root=env)
    assert found == other


def test_stale_windows_path_falls_back_to_search(tmp_path, monkeypatch):
    ds = _fake_dataset(tmp_path)
    env = tmp_path / "W6" / "env"
    env.mkdir(parents=True)
    monkeypatch.chdir(env)
    monkeypatch.delenv("PVCH2_DATASET_ROOT", raising=False)
    monkeypatch.delenv("PLANTVILLAGE_DATASET", raising=False)
    found = resolve_dataset_root(
        {
            "paths": {
                "dataset_root": "E:/IT/Projects/42177 Image ProcessingAndPatternRecognition/PlantVillage-Dataset"
            }
        },
        env_root=env,
    )
    assert found == ds
