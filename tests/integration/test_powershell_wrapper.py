from __future__ import annotations

import shutil
import subprocess

import pytest


def test_powershell_wrapper_forwards_pipeline_option(context):
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("PowerShell is required for the canonical Windows deployment")

    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(context.root / "scripts" / "pv.ps1"),
            "predict",
            "--pipeline",
            "hog_svm",
            "--image",
            "missing-image.jpg",
            "--model",
            "missing-model.joblib",
            "--config",
            "configs/project_config.json",
            "--device",
            "cpu",
        ],
        cwd=context.root,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "following arguments are required: --pipeline" not in combined
    assert "ERROR:" in combined
