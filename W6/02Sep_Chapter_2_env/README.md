# 02Sep_Chapter_2_env

Version **0.2.0** (2026-09-02). Python **3.13.x** (any patch: 3.13.7 on Apple Silicon is fine; do not require 3.13.9).

PlantVillage Chapter 2 experiment environment. Hyperparameters are taken from `localDoc/PlantVillage_Techniques_Conversation_Morning02Sep.pdf`. Colour, grayscale, and segmented are **independent pipelines**. A leaf mask is used only in the segmented pipeline.

The 54k-image dataset stays where it is. This env does **not** copy it. At runtime it looks for a folder named `PlantVillage-Dataset` in:

1. `PVCH2_DATASET_ROOT` (or `PLANTVILLAGE_DATASET`) if set
2. `.` — next to this env
3. `..` — sibling of this env (Windows layout)
4. `../..` — course root when this env lives under `W6/` (macOS layout)

## Layout

```
configs/          PDF-locked YAML (edit here, not in code)
src/pvch2/        importable package
scripts/          bootstrap.py / bootstrap.ps1
tests/
outputs/          generated (gitignored)
.venv/            local 3.13.x  (never copy this between Windows and macOS)
```

## Setup

Rebuild `.venv` **on the machine you will run on**. A Windows venv cannot run on Apple Silicon.

### macOS (Apple Silicon)

```bash
cd "/path/to/W6/02Sep_Chapter_2_env"
python3 scripts/bootstrap.py
source .venv/bin/activate
python -m pvch2.cli verify
```

Homebrew Python 3.13.x is enough (`brew install python@3.13`). You do not need the Windows 3.13.9 installer.

### Windows

```powershell
cd <this-env>
python scripts/bootstrap.py
.\.venv\Scripts\Activate.ps1
python -m pvch2.cli verify
```

Optional later: `python -m pip install -e ".[dl]"` for PyTorch.

If the dataset is not at `.` / `..` / `../..`:

```bash
export PVCH2_DATASET_ROOT="/path/to/PlantVillage-Dataset"
```

```powershell
$env:PVCH2_DATASET_ROOT = "E:\IT\Projects\42177 Image ProcessingAndPatternRecognition\PlantVillage-Dataset"
```

## Commands

```bash
python -m pvch2.cli verify
python -m pvch2.cli inventory
python -m pvch2.cli preview
python -m pvch2.cli check
python -m pvch2.cli check 3.3.2 3.3.4 3.3.5 3.3.6 3.4
python -m pytest -q
```

Each PDF `3.x.x` has its own checker in `src/pvch2/checks/sections.py`. Artifacts go to `outputs/checks/`.

## Locked 2.1 numbers (PDF §3.7)

| Step | Parameter |
|---|---|
| Resize | 256×256 bicubic |
| Colour | RGB, CIE Lab, HSV; ExG = 2G−R−B |
| Illumination | gray-world; CLAHE on L only, clip 2.0, tile 8×8 |
| Denoise | bilateral d=5, σ_color=50, σ_space=50 |
| Mask (segmented only) | Otsu on ExG; open/close disk r=3; largest component |
| Gradient | Scharr magnitude (optional extra channel) |

HOG / SIFT / corners are not preprocessing. They belong to 2.2.

## Extending

1. Change a number in `configs/*.yaml`.
2. Add a new pipeline key under `configs/pipelines.yaml` with its own `apply` flags.
3. Keep new operators in `src/pvch2/preprocess/ops.py`.
4. Deep-learning training is an extra (`[dl]`), not part of the default install.
