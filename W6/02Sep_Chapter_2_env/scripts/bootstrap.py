#!/usr/bin/env python3
"""Create a machine-local .venv with Python 3.13+.

A Windows venv (Scripts/, python.exe) cannot run on Apple Silicon, and a
macOS venv cannot run on Windows. This script rebuilds .venv in place using
whatever 3.13.x is installed on this computer. Patch versions may differ
(3.13.7 vs 3.13.9); that is expected.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
MIN = (3, 13)


def _version_of(exe: str) -> tuple[int, int, int] | None:
    try:
        out = subprocess.check_output(
            [exe, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        parts = tuple(int(p) for p in out.split("."))
        if len(parts) == 3:
            return parts  # type: ignore[return-value]
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None
    return None


def _ok(ver: tuple[int, int, int] | None) -> bool:
    return ver is not None and ver[:2] >= MIN


def _win_py_launcher() -> str | None:
    py = shutil.which("py")
    if not py:
        return None
    try:
        out = subprocess.check_output(
            [py, f"-{MIN[0]}.{MIN[1]}", "-c", "import sys; print(sys.executable)"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


def find_python() -> str:
    forced = os.environ.get("PVCH2_PYTHON", "").strip()
    candidates: list[str] = []
    if forced:
        candidates.append(forced)
    for name in ("python3.13", "python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    launcher = _win_py_launcher()
    if launcher:
        candidates.append(launcher)
    # Prefer a non-venv interpreter on this machine.
    this = sys.executable
    if this and Path(this).resolve().parts[-2:] != ("bin", "python"):
        candidates.append(this)

    seen: set[str] = set()
    for exe in candidates:
        key = str(Path(exe).resolve()) if Path(exe).exists() else exe
        if key in seen:
            continue
        seen.add(key)
        ver = _version_of(exe)
        if _ok(ver):
            return exe
    raise SystemExit(
        f"Need Python {MIN[0]}.{MIN[1]}+ on this machine (any patch, e.g. 3.13.7). "
        "On Apple Silicon: brew install python@3.13. "
        "Or set PVCH2_PYTHON to a 3.13 interpreter."
    )


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def venv_is_foreign() -> bool:
    if not VENV.is_dir():
        return False
    posix_py = (VENV / "bin" / "python").exists()
    win_py = (VENV / "Scripts" / "python.exe").exists()
    if os.name == "nt":
        return win_py is False and posix_py is True
    return posix_py is False and win_py is True


def main() -> None:
    extra = sys.argv[1] if len(sys.argv) > 1 else ".[dev]"
    py = find_python()
    ver = _version_of(py)
    print(f"Python: {py} ({'.'.join(map(str, ver or ()))})", flush=True)
    print(f"Root:   {ROOT}", flush=True)
    print(f"Venv:   {VENV}", flush=True)

    if venv_is_foreign() or (VENV.exists() and not venv_python().exists()):
        print("Removing foreign-OS .venv (Windows/macOS venvs are not interchangeable).")
        shutil.rmtree(VENV)

    if not venv_python().exists():
        subprocess.check_call([py, "-m", "venv", str(VENV)])

    vp = str(venv_python())
    subprocess.check_call([vp, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
    subprocess.check_call([vp, "-m", "pip", "install", "-e", extra], cwd=str(ROOT))
    activate = (
        str(VENV / "Scripts" / "Activate.ps1")
        if os.name == "nt"
        else str(VENV / "bin" / "activate")
    )
    print(f"Done. Activate: {activate}")
    print("Then: python -m pvch2.cli verify")


if __name__ == "__main__":
    main()
