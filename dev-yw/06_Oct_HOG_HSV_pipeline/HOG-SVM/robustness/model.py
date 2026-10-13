"""Refit the frozen S4 round-99 HOG+SVM and score one image."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from corrupt import apply_corruption
from vein_hog.extract import _prepare, load_bgr
from vein_hog.hsv import HSV_DIM, hsv_aux
from vein_hog.hsv_window import require_hsv_alpha
from vein_hog.rounds import SEED, sgd_alpha
from vein_hog.run import find_project
from vein_hog.search_s1 import C_VALUE, SELECT_FRACTION
from vein_hog.search_s4 import (
    SGD_MAX_ITER,
    SGD_TOL,
    _hsv,
    _paths,
    compute_hog,
    hog_vector,
)
from vein_hog.validate_s4 import FROZEN_ALPHA, FROZEN_HOG, FROZEN_ROUND, require_frozen_winner

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
ORIENTATIONS = 12
PIXELS_PER_CELL = 16
HOG_DIM = 10800


def output_dir() -> Path:
    path = Path(__file__).resolve().parent / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def svm_shortcut() -> Path:
    """Windows shortcut kept beside the PlantVillage S4 feature tables.

    The pair is recorded in plantvillage_s4_full/SVM.txt and in
    HOG-SVM/robustness/outputs/model_cache.txt.
    """
    return find_project(PIPELINE_ROOT) / "plantvillage_s4_full" / "SVM.lnk"


def resolve_shortcut(path: Path) -> Path:
    """Target of a .lnk file. A normal path is returned unchanged."""
    if path.suffix.lower() != ".lnk":
        return path
    env = os.environ.copy()
    env["SVM_LNK"] = str(path)
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(New-Object -ComObject WScript.Shell).CreateShortcut($env:SVM_LNK).TargetPath",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    target = completed.stdout.strip()
    if not target:
        raise RuntimeError(f"shortcut has no target: {path}")
    return Path(target)


def featurize(bgr_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HOG 10800 维加 HSV 27 维。掩膜失败时背景不置零、HSV 记为 0。

    HOG 在 vein_hog.search_s4.hog_vector：256 灰度、叶外像素置 0、
    12 方向、16 像素胞元、2×2 块、L2-Hys、平方根压缩。
    HSV 在 vein_hog.hsv.hsv_aux：叶内 16 个 H 箱、8 个 S 箱，再加 H、S、V 均值。
    """
    hog = hog_vector(bgr_u8, ORIENTATIONS, PIXELS_PER_CELL)
    bgr256, mask256, mask_ok = _prepare(bgr_u8)
    colour = hsv_aux(bgr256, mask256 if mask_ok else np.zeros_like(mask256))
    colour = np.asarray(colour, dtype=np.float32).reshape(-1)
    if colour.shape != (HSV_DIM,):
        raise RuntimeError(f"HSV dimension {colour.shape} != {(HSV_DIM,)}")
    return hog, colour


def extract_job(job: tuple[str, str, int, str]) -> tuple[np.ndarray, np.ndarray]:
    """Load one validation image, corrupt it, and return HOG and HSV."""
    path, category, severity, sha256 = job
    image = load_bgr(Path(path))
    corrupted = apply_corruption(image, category, int(severity), sha256)
    return featurize(corrupted)


def predict_rows(hog: np.ndarray, hsv: np.ndarray, model: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """HOG、HSV 各自标准化，HSV 再乘 0.45，拼成 10827 维后做线性打分。

    预测取 hinge 决策分数最大的一类，不是概率。
    """
    hog_s = (hog - model["hog_mean"]) / model["hog_scale"]
    hsv_s = ((hsv - model["hsv_mean"]) / model["hsv_scale"]) * float(model["hsv_alpha"][0])
    design = np.hstack([hog_s, hsv_s]).astype(np.float64, copy=False)
    scores = design @ model["coef"].T + model["intercept"]
    classes = np.asarray(model["classes"]).astype(str)
    predicted = classes[np.argmax(scores, axis=1)]
    return predicted, np.asarray(scores, dtype=np.float64)


def true_class_scores(scores: np.ndarray, true_labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    index = {str(name): column for column, name in enumerate(classes)}
    columns = np.array([index[str(name)] for name in true_labels], dtype=int)
    return scores[np.arange(len(columns)), columns]


def _pack(clf: SGDClassifier, hog_scaler: StandardScaler, hsv_scaler: StandardScaler, alpha: float) -> dict[str, np.ndarray]:
    classes = np.asarray([str(item) for item in clf.classes_])
    coef = np.asarray(clf.coef_, dtype=np.float64)
    intercept = np.asarray(clf.intercept_, dtype=np.float64).reshape(-1)
    if coef.shape != (len(classes), HOG_DIM + HSV_DIM):
        raise RuntimeError(f"coefficient shape {coef.shape} is not {(len(classes), HOG_DIM + HSV_DIM)}")
    return {
        "coef": coef,
        "intercept": intercept,
        "classes": classes,
        "hog_mean": np.asarray(hog_scaler.mean_, dtype=np.float64),
        "hog_scale": np.asarray(hog_scaler.scale_, dtype=np.float64),
        "hsv_mean": np.asarray(hsv_scaler.mean_, dtype=np.float64),
        "hsv_scale": np.asarray(hsv_scaler.scale_, dtype=np.float64),
        "hsv_alpha": np.asarray([alpha], dtype=np.float64),
    }


def load_model(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        model = {key: data[key] for key in data.files}
    if abs(float(model["hsv_alpha"][0]) - FROZEN_ALPHA) > 1e-9:
        raise RuntimeError("cached HSV weight is not 0.45")
    if model["coef"].shape[1] != HOG_DIM + HSV_DIM:
        raise RuntimeError("cached coefficient width is not the S4 feature width")
    return model


def validation_table(root: Path | None = None) -> tuple[pd.DataFrame, Path]:
    root = PIPELINE_ROOT if root is None else Path(root)
    project = find_project(root)
    table = pd.read_csv(root / "vein_hog" / "data" / "split_map.csv")
    validation = table[table["split"] == "validation"].reset_index(drop=True)
    if len(validation) != 375 or set(validation["split"]) != {"validation"}:
        raise RuntimeError(f"expected 375 validation rows, got {len(validation)}")
    return validation, project


def fit_frozen(cache_path: Path) -> dict[str, np.ndarray]:
    """快捷方式缺失时才拟合。训练 HOG 重算，训练 HSV 用已有缓存。

    3000 张训练图取 80%（种子 0）拟合，标准化也只在这 80% 上估计。
    C=1 换成 SGD 的 alpha = 1/(C×2400)。不改 winner.json，不把验证特征写入共享缓存。
    """
    root = PIPELINE_ROOT
    project = find_project(root)
    search_dir = root / "outputs" / "search_s4"
    cache_dir = root / "outputs" / "cache"
    winner_path = search_dir / "winner.json"
    before = winner_path.read_text(encoding="utf-8")
    payload = json.loads(before)
    require_frozen_winner(payload)
    if str(payload["hog"]) != FROZEN_HOG or int(payload["winner_round"]) != FROZEN_ROUND:
        raise RuntimeError("winner.json is not S4 round 99")

    train_meta = pd.read_csv(cache_dir / "train_meta.csv")
    if set(train_meta["split"]) != {"train"}:
        raise RuntimeError("training cache must not contain validation rows")
    labels = train_meta["class_name"].to_numpy()
    fit_index, _select_index = next(
        StratifiedShuffleSplit(n_splits=1, test_size=SELECT_FRACTION, random_state=SEED).split(
            np.zeros(len(labels)), labels
        )
    )
    fit_index = np.asarray(fit_index, dtype=int)
    leaked_before = (cache_dir / "validation_features.npz").exists()
    print("正在按 S4 第 99 轮重算训练 HOG（12 方向，16 像素胞元）。", flush=True)
    train_hog = compute_hog(_paths(train_meta, project), ORIENTATIONS, PIXELS_PER_CELL)
    train_hsv = _hsv(cache_dir, "train")
    if train_hog.shape != (len(train_meta), HOG_DIM) or train_hsv.shape != (len(train_meta), HSV_DIM):
        raise RuntimeError("training HOG and HSV rows do not match the S4 widths")
    alpha = require_hsv_alpha(float(payload["hsv_alpha"]))
    hog_scaler = StandardScaler().fit(train_hog[fit_index])
    hsv_scaler = StandardScaler().fit(train_hsv[fit_index])

    def design(index: np.ndarray) -> np.ndarray:
        scaled_hog = hog_scaler.transform(train_hog[index])
        scaled_hsv = hsv_scaler.transform(train_hsv[index]) * alpha
        return np.hstack([scaled_hog, scaled_hsv]).astype(np.float32, copy=False)

    print(f"正在拟合线性 SVM，拟合子集 {len(fit_index)} 张。", flush=True)
    clf = SGDClassifier(
        loss="hinge",
        alpha=sgd_alpha(C_VALUE, len(fit_index)),
        class_weight="balanced",
        random_state=SEED,
        max_iter=SGD_MAX_ITER,
        tol=SGD_TOL,
    )
    clf.fit(design(fit_index), labels[fit_index])
    model = _pack(clf, hog_scaler, hsv_scaler, alpha)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **model)
    if winner_path.read_text(encoding="utf-8") != before:
        raise RuntimeError("refit rewrote winner.json")
    if (cache_dir / "validation_features.npz").exists() != leaked_before:
        raise RuntimeError("refit wrote a validation feature cache")
    print("拟合完成，系数写在本次测试目录。", flush=True)
    return model
