"""Score the frozen S6 winner on the main-ref test split at five nodes.

The search winner is not changed. Validation and train images are not opened.
Figures use one test image from each class.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from corrupt import brightness_factor, gaussian_blur_ksize, gaussian_noise_sigma, opencv_blur_sigma
from plot_robustness import CLASS_LABEL
from vein_hog.extract import load_bgr
from vein_hog.run import find_project
from vein_hog.search_s6 import classification_metrics, featurize_leaf, load_winner_model, predict_rows

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
SEARCH_DIR = PIPELINE_ROOT / "outputs" / "search_s6"
OUT_DIR = SEARCH_DIR / "robustness_test"
WORKERS = 4
NODES = (
    {"id": "original", "label": "Original"},
    {"id": "gaussian_noise_0.05", "label": "Gaussian Noise (σ = 0.05)"},
    {"id": "gaussian_noise_0.10", "label": "Gaussian Noise (σ = 0.10)"},
    {"id": "gaussian_blur_3", "label": "Gaussian Blur (3 × 3)"},
    {"id": "gaussian_blur_5", "label": "Gaussian Blur (5 × 5)"},
    {"id": "brightness_0.7", "label": "Brightness (0.7)"},
    {"id": "brightness_1.3", "label": "Brightness (1.3)"},
)


def corrupt_node(image: np.ndarray, node_id: str, image_sha256: str) -> np.ndarray:
    if node_id == "original":
        return np.ascontiguousarray(image).copy()
    if node_id == "gaussian_noise_0.05":
        return gaussian_noise_sigma(image, 0.05, image_sha256)
    if node_id == "gaussian_noise_0.10":
        return gaussian_noise_sigma(image, 0.10, image_sha256)
    if node_id == "gaussian_blur_3":
        return gaussian_blur_ksize(image, 3)
    if node_id == "gaussian_blur_5":
        return gaussian_blur_ksize(image, 5)
    if node_id == "brightness_0.7":
        return brightness_factor(image, 0.7)
    if node_id == "brightness_1.3":
        return brightness_factor(image, 1.3)
    raise ValueError(f"unknown test node: {node_id}")


def extract_node(job: tuple) -> tuple[np.ndarray, np.ndarray]:
    path, node_id, sha256, orientations, cell, layout = job
    image = load_bgr(Path(path))
    corrupted = corrupt_node(image, str(node_id), str(sha256))
    return featurize_leaf(corrupted, int(orientations), int(cell), str(layout))


def _assert_test_relpath(relpath: str) -> None:
    parts = [part.lower() for part in Path(relpath).parts]
    if "validation" in parts or "train" in parts:
        raise RuntimeError(f"test node path leaves dataset/test: {relpath}")
    if "dataset" not in parts:
        raise RuntimeError(f"test node path is not under dataset/test: {relpath}")
    dataset_at = parts.index("dataset")
    if dataset_at + 1 >= len(parts) or parts[dataset_at + 1] != "test":
        raise RuntimeError(f"test node path is not dataset/test: {relpath}")


def test_table() -> tuple[pd.DataFrame, Path, dict[str, object]]:
    project = find_project(PIPELINE_ROOT)
    table = pd.read_csv(PIPELINE_ROOT / "vein_hog" / "data" / "split_map.csv")
    counts = {split: set(table.loc[table["split"] == split, "sha256"].astype(str)) for split in ("train", "test", "validation")}
    test = table[table["split"] == "test"].reset_index(drop=True)
    if len(test) != 375 or set(test["split"]) != {"test"}:
        raise RuntimeError(f"expected 375 test rows, got {len(test)}")
    if set(test["class_name"].astype(str)) != set(CLASS_LABEL):
        raise RuntimeError("test split is missing a tomato class")
    sha = set(test["sha256"].astype(str))
    if sha != counts["test"] or len(sha) != 375:
        raise RuntimeError("test sha256 set is not the 375-row test split")
    overlap_train = len(sha & counts["train"])
    overlap_validation = len(sha & counts["validation"])
    if overlap_train or overlap_validation:
        raise RuntimeError(f"test sha256 overlaps train {overlap_train} or validation {overlap_validation}")
    missing: list[str] = []
    for relpath in test["local_relpath"].astype(str):
        _assert_test_relpath(relpath)
        if not (project / relpath).is_file():
            missing.append(relpath)
    if missing:
        raise RuntimeError(f"{len(missing)} test images are missing, first: {missing[0]}")
    per_class = test["class_name"].astype(str).value_counts().to_dict()
    if any(int(per_class.get(name, 0)) != 75 for name in CLASS_LABEL):
        raise RuntimeError(f"test class counts are not 75: {per_class}")
    examples = []
    for name in CLASS_LABEL:
        part = test[test["class_name"].astype(str) == name].copy()
        part["filename"] = part["local_relpath"].astype(str).map(lambda item: Path(item).name)
        chosen = part.sort_values(["filename", "sha256"], kind="mergesort").iloc[0]
        relpath = str(chosen["local_relpath"])
        _assert_test_relpath(relpath)
        examples.append(
            {
                "class_name": name,
                "label": CLASS_LABEL[name],
                "filename": str(chosen["filename"]),
                "local_relpath": relpath,
                "sha256": str(chosen["sha256"]),
            }
        )
    identity = {
        "image_split": "test",
        "n": 375,
        "overlap_train": overlap_train,
        "overlap_validation": overlap_validation,
        "missing_files": 0,
        "per_class": {name: 75 for name in CLASS_LABEL},
        "examples": examples,
    }
    return test, project, identity


def _jobs(test: pd.DataFrame, project: Path, node_id: str, model: dict[str, np.ndarray]) -> list[tuple]:
    orientations = int(model["orientations"][0])
    cell = int(model["pixels_per_cell"][0])
    layout = str(model["hsv_layout"][0])
    jobs = []
    for row in test.itertuples(index=False):
        _assert_test_relpath(str(row.local_relpath))
        jobs.append(
            (
                str(project / str(row.local_relpath)),
                node_id,
                str(row.sha256),
                orientations,
                cell,
                layout,
            )
        )
    if len(jobs) != 375:
        raise RuntimeError(f"expected 375 jobs, got {len(jobs)}")
    return jobs


def _stack(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if len(pairs) != 375:
        raise RuntimeError(f"expected 375 feature rows, got {len(pairs)}")
    hog = np.stack([item[0] for item in pairs]).astype(np.float64, copy=False)
    hsv = np.stack([item[1] for item in pairs]).astype(np.float64, copy=False)
    return hog, hsv


def _score_frame(model: dict[str, np.ndarray], test: pd.DataFrame, hog: np.ndarray, hsv: np.ndarray, node_id: str) -> pd.DataFrame:
    predicted, _scores = predict_rows(hog, hsv, model)
    frame = pd.DataFrame(
        {
            "sha256": test["sha256"].astype(str).to_numpy(),
            "class_name": test["class_name"].astype(str).to_numpy(),
            "predicted_class": predicted.astype(str),
            "node_id": node_id,
        }
    )
    if set(frame["sha256"]) != set(test["sha256"].astype(str)):
        raise RuntimeError(f"{node_id} predictions are not the test sha256 set")
    return frame


def _load_model(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def ensure_model(winner_text: str) -> dict[str, np.ndarray]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "model.npz"
    stamp_path = OUT_DIR / "model_winner.txt"
    if cache_path.exists() and stamp_path.exists() and stamp_path.read_text(encoding="utf-8") == winner_text:
        print("沿用已保存的 S6 系数。", flush=True)
        return _load_model(cache_path)
    project = find_project(PIPELINE_ROOT)
    print("正在按 S6 赢家拟合训练集上的线性 SVM。", flush=True)
    model = load_winner_model(SEARCH_DIR, PIPELINE_ROOT / "outputs" / "cache", project)
    np.savez_compressed(cache_path, **model)
    stamp_path.write_text(winner_text, encoding="utf-8")
    return model


def _prediction_path(node_id: str) -> Path:
    return OUT_DIR / "predictions" / f"{node_id}.csv"


def _read_prediction(path: Path, expected: set[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if len(frame) != 375 or set(frame["sha256"].astype(str)) != expected:
        path.unlink()
        return None
    return frame


def _write_prediction(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.writing.png")
    fig.savefig(temporary, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    temporary.replace(path)


def _theme() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 160,
        }
    )


def _panel(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)


def write_examples(project: Path, examples: list[dict[str, str]]) -> None:
    _theme()
    fig, axes = plt.subplots(len(examples), len(NODES), figsize=(22.4, 13.6))
    for row, example in enumerate(examples):
        relpath = example["local_relpath"]
        _assert_test_relpath(relpath)
        original = load_bgr(project / relpath)
        for column, node in enumerate(NODES):
            view = _panel(corrupt_node(original, node["id"], example["sha256"]))
            ax = axes[row, column]
            ax.imshow(view)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(node["label"], fontsize=9)
            if column == 0:
                ax.set_ylabel(
                    f"{example['label']}\n{example['filename']}",
                    fontsize=7,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=10,
                )
    fig.suptitle("Test split examples  ·  dataset/test", fontsize=13)
    fig.tight_layout(rect=(0.14, 0.0, 1.0, 0.96))
    _save_figure(fig, OUT_DIR / "figures" / "examples.png")


def write_confusion(frame: pd.DataFrame) -> None:
    _theme()
    labels = list(CLASS_LABEL)
    matrix = confusion_matrix(frame["class_name"].astype(str), frame["predicted_class"].astype(str), labels=labels)
    display = [CLASS_LABEL[name] for name in labels]
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    sns.heatmap(
        pd.DataFrame(matrix, index=display, columns=display),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        ax=ax,
    )
    ax.set_title("Confusion Matrix of HOG-SVM Classifier")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.tight_layout()
    _save_figure(fig, OUT_DIR / "figures" / "confusion_matrix.png")


def write_metric_bars(scores: pd.DataFrame) -> None:
    _theme()
    metric_names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-score"]
    long = scores.melt(id_vars=["Condition"], value_vars=metric_names, var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    sns.barplot(data=long, x="Condition", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("S6 HOG-SVM on the test split")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save_figure(fig, OUT_DIR / "figures" / "metrics.png")


def _parameters() -> dict[str, object]:
    return {
        "image_split": "test",
        "original": "copy of the BGR image",
        "gaussian_noise": (
            "pixel/255 + N(0, sigma), clip to [0, 1], times 255. "
            "sigma is 0.05 or 0.10 on the unit interval. "
            "seed is sha256(gaussian_noise|{sigma:.2f}|image_sha256), first 8 bytes, little-endian"
        ),
        "gaussian_blur_3": f"cv2.GaussianBlur ksize (3, 3), sigmaX=0, OpenCV sigma {opencv_blur_sigma(3):.1f}",
        "gaussian_blur_5": f"cv2.GaussianBlur ksize (5, 5), sigmaX=0, OpenCV sigma {opencv_blur_sigma(5):.1f}",
        "brightness": "pixel * factor, clipped to uint8. Factors are 0.7 and 1.3. Factor 1.0 copies the input.",
    }


def main() -> None:
    winner_path = SEARCH_DIR / "winner.json"
    winner_text = winner_path.read_text(encoding="utf-8")
    payload = json.loads(winner_text)
    if payload.get("test_was_read") or payload.get("validation_was_read"):
        raise RuntimeError("test nodes refuse a winner that already read a held-out split during search")
    if payload.get("replaces_report_round") or payload.get("replaces_s4_winner"):
        raise RuntimeError("test nodes refuse a winner that replaces a frozen model")
    if int(payload["winner_round"]) != 384:
        raise RuntimeError(f"frozen S6 winner is not round 384: {payload['winner_round']}")
    test, project, identity = test_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "image_identity.json").write_text(json.dumps(identity, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"TEST_IDENTITY {identity['n']} overlap_train={identity['overlap_train']} "
        f"overlap_validation={identity['overlap_validation']}",
        flush=True,
    )
    model = ensure_model(winner_text)
    if winner_path.read_text(encoding="utf-8") != winner_text:
        raise RuntimeError("model refit rewrote winner.json")
    expected = set(test["sha256"].astype(str))
    frames: list[pd.DataFrame] = []
    for node in NODES:
        path = _prediction_path(node["id"])
        frame = _read_prediction(path, expected)
        if frame is None:
            print(f"正在打分 {node['label']}", flush=True)
            with ProcessPoolExecutor(max_workers=WORKERS) as pool:
                pairs = list(pool.map(extract_node, _jobs(test, project, node["id"], model), chunksize=8))
            hog, hsv = _stack(pairs)
            frame = _score_frame(model, test, hog, hsv, node["id"])
            _write_prediction(frame, path)
        frames.append(frame)
        print(f"完成 {node['label']} {len(frame)}", flush=True)
    rows = []
    original_accuracy = None
    for node, frame in zip(NODES, frames):
        scores = classification_metrics(frame["class_name"], frame["predicted_class"])
        if node["id"] == "original":
            original_accuracy = scores["Accuracy"]
        if original_accuracy is None:
            raise RuntimeError("Original is not the first node")
        rows.append(
            {
                "node_id": node["id"],
                "Condition": node["label"],
                **scores,
                "Accuracy reduction (percentage points)": (original_accuracy - scores["Accuracy"]) * 100.0,
            }
        )
    scores_frame = pd.DataFrame(rows)
    scores_frame.to_csv(OUT_DIR / "scores.csv", index=False)
    summary = {
        "image_split": "test",
        "n": 375,
        "used_to_choose_winner": False,
        "winner_round": int(payload["winner_round"]),
        "hog": str(payload["hog"]),
        "orientations": int(payload["orientations"]),
        "pixels_per_cell": int(payload["pixels_per_cell"]),
        "hsv_layout": str(payload["hsv_layout"]),
        "hsv_alpha": float(payload["hsv_alpha"]),
        "winner_sha256": hashlib.sha256(winner_text.encode("utf-8")).hexdigest(),
        "examples": identity["examples"],
        "conditions": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "parameters.json").write_text(json.dumps(_parameters(), indent=2), encoding="utf-8")
    write_examples(project, identity["examples"])
    original = frames[0]
    if set(original["node_id"].astype(str)) != {"original"}:
        raise RuntimeError("confusion matrix is not the Original node")
    write_confusion(original)
    write_metric_bars(scores_frame)
    if winner_path.read_text(encoding="utf-8") != winner_text:
        raise RuntimeError("test nodes rewrote winner.json")
    for frame in frames:
        if set(frame["sha256"].astype(str)) != expected:
            raise RuntimeError("a prediction table is not the test split")
    print(scores_frame.drop(columns=["node_id"]).to_string(index=False), flush=True)
    print("TEST_NODES_DONE", flush=True)


if __name__ == "__main__":
    main()
