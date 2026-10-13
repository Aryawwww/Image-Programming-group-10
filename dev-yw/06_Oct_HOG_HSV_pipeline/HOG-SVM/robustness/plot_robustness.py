"""Seaborn figures for the HOG+SVM robustness sweep."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from corrupt import CATEGORIES, CATEGORY_EN, apply_corruption
from vein_hog.extract import load_bgr
from vein_hog.run import find_project

CLASS_LABEL = {
    "Tomato_healthy": "Healthy",
    "Tomato_Bacterial_spot": "Bacterial spot",
    "Tomato_Late_blight": "Late blight",
    "Tomato_Septoria_leaf_spot": "Septoria leaf spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Yellow leaf curl virus",
}
PALETTE = {
    "Healthy": "#1d4e89",
    "Bacterial spot": "#d9480f",
    "Late blight": "#6f2dbd",
    "Septoria leaf spot": "#77ff11",
    "Yellow leaf curl virus": "#0e7490",
}
VIOLIN_SEVERITY = (0, 20, 40, 60, 80, 100)
CONFUSION_SEVERITY = (5, 10, 50, 70, 90)
LEGEND_SEVERITY = (0, 25, 50, 75, 100)
BILATERAL_LEGEND_STEP = 2.5
BILATERAL_LEGEND_CANVAS = 16384
CURVE_X = "Perturbation Severity (0 original, 100 unrecognizable)"
LEGEND_RELPATH = "dev-yw-main-ref/dataset/validation/Tomato_healthy/Tomato_healthy_img_1.jpg"
LEGEND_SHA256 = "9192053d985e565755e7ccb5ca1f3e45c77e22a16965cc67a78bdd55e72e86a9"


def _theme() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "axes.titlepad": 6,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.writing.png")
    fig.savefig(temporary, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    temporary.replace(path)


def load_predictions(out_dir: Path) -> pd.DataFrame:
    files = sorted((out_dir / "levels").glob("*/*.csv"))
    expected = len(CATEGORIES) * 101
    if len(files) != expected:
        raise RuntimeError(f"expected {expected} level files, found {len(files)}")
    frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    if frame["severity"].nunique() != 101 or set(frame["category"]) != set(CATEGORIES):
        raise RuntimeError("level files do not cover every category and severity")
    return frame


def _class_order(frame: pd.DataFrame) -> list[str]:
    present = set(frame["class_name"].astype(str))
    order = [name for name in CLASS_LABEL if name in present]
    if len(order) != len(CLASS_LABEL):
        raise RuntimeError(f"missing classes: {sorted(set(CLASS_LABEL) - set(order))}")
    return order


def _counts(frame: pd.DataFrame, classes: list[str]) -> np.ndarray:
    return confusion_matrix(
        frame["class_name"].astype(str),
        frame["predicted_class"].astype(str),
        labels=classes,
    )


def _curve(group: pd.DataFrame, scores: pd.DataFrame, title: str, path: Path) -> None:
    del group
    long_rows: list[dict[str, object]] = []
    for row in scores.itertuples(index=False):
        for name, label in CLASS_LABEL.items():
            long_rows.append(
                {
                    "severity": int(row.severity),
                    "F1": float(getattr(row, f"f1__{name}")),
                    "Class": label,
                }
            )
    long = pd.DataFrame(long_rows)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    sns.lineplot(
        data=long,
        x="severity",
        y="F1",
        hue="Class",
        hue_order=list(CLASS_LABEL.values()),
        palette=PALETTE,
        linewidth=1.15,
        ax=ax,
    )
    ax.plot(
        scores["severity"].to_numpy(),
        scores["macro_f1"].to_numpy(),
        color="#222222",
        linewidth=2.3,
        label="Macro-F1",
        zorder=5,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(CURVE_X)
    ax.set_ylabel("F1")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    order = ["Macro-F1", *CLASS_LABEL.values()]
    paired = {label: handle for handle, label in zip(handles, labels)}
    ax.legend([paired[name] for name in order if name in paired], [name for name in order if name in paired], frameon=False)
    _save(fig, path)


def _violin(group: pd.DataFrame, title: str, path: Path) -> None:
    picked = group[group["severity"].isin(VIOLIN_SEVERITY)].copy()
    picked["Severity"] = picked["severity"].astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    sns.violinplot(
        data=picked,
        x="Severity",
        y="true_class_score",
        order=[str(value) for value in VIOLIN_SEVERITY],
        inner="box",
        cut=0,
        color="#1d4e89",
        ax=ax,
    )
    ax.set_xlabel("Perturbation Severity")
    ax.set_ylabel("True-class decision score")
    ax.set_title(title)
    _save(fig, path)


def _heat(ax: plt.Axes, matrix: np.ndarray, classes: list[str], title: str, center: float | None) -> None:
    labels = [CLASS_LABEL[name] for name in classes]
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    kwargs: dict[str, object] = {"annot": True, "fmt": "d", "cbar": True, "square": True, "ax": ax}
    if center is None:
        sns.heatmap(frame, cmap="Blues", **kwargs)
    else:
        limit = max(int(np.max(np.abs(matrix))), 1)
        sns.heatmap(frame, cmap="vlag", center=center, vmin=-limit, vmax=limit, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")


def _confusion(group: pd.DataFrame, classes: list[str], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(CONFUSION_SEVERITY), figsize=(26.8, 5.2))
    for ax, severity in zip(axes, CONFUSION_SEVERITY):
        part = group[group["severity"] == severity]
        if len(part) != 375:
            raise RuntimeError(f"severity {severity} has {len(part)} rows")
        _heat(ax, _counts(part, classes), classes, f"Perturbation Severity {severity}", None)
    fig.suptitle(title, fontsize=12)
    _save(fig, path)


def _difference(group: pd.DataFrame, classes: list[str], title: str, path: Path) -> None:
    base = _counts(group[group["severity"] == 0], classes)
    fig, axes = plt.subplots(1, 2, figsize=(16.4, 5.0))
    for ax, severity in zip(axes, (50, 100)):
        delta = _counts(group[group["severity"] == severity], classes) - base
        _heat(ax, delta, classes, f"Perturbation Severity {severity} − 0", 0.0)
    fig.suptitle(title, fontsize=12)
    _save(fig, path)


def _legend_levels(category: str) -> tuple[float, ...]:
    if category == "bilateral":
        count = int(round(100 / BILATERAL_LEGEND_STEP)) + 1
        return tuple(index * BILATERAL_LEGEND_STEP for index in range(count))
    return LEGEND_SEVERITY


def _severity_title(severity: float) -> str:
    if float(severity).is_integer():
        return f"Perturbation Severity {int(severity)}"
    return f"Perturbation Severity {severity:.1f}"


def _legend_font(size: int, bold: bool = False):
    from PIL import ImageFont

    filename = "timesbd.ttf" if bold else "times.ttf"
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / filename), size)


def _draw_bilateral_legend_hires(original: np.ndarray, levels: tuple[float, ...]):
    """16384 square legend. Each 256 leaf is enlarged 8× with nearest neighbours."""
    from PIL import Image, ImageDraw

    canvas = BILATERAL_LEGEND_CANVAS
    columns = 8
    leaf = 2048
    title_h = 640
    label_h = 512
    if title_h + 6 * (label_h + leaf) + 384 != canvas or columns * leaf != canvas:
        raise RuntimeError("bilateral legend layout does not fill 16384×16384")
    image = Image.new("RGB", (canvas, canvas), "white")
    draw = ImageDraw.Draw(image)
    title_font = _legend_font(200, bold=True)
    label_font = _legend_font(92)
    foot_font = _legend_font(96)
    draw.text(
        (canvas // 2, title_h // 2),
        "Bilateral filter: reference",
        fill="black",
        font=title_font,
        anchor="mm",
    )
    for index, severity in enumerate(levels):
        row, column = divmod(index, columns)
        corrupted = apply_corruption(original, "bilateral", severity, LEGEND_SHA256)
        rgb = cv2.cvtColor(corrupted, cv2.COLOR_BGR2RGB)
        panel = cv2.resize(rgb, (leaf, leaf), interpolation=cv2.INTER_NEAREST)
        x = column * leaf
        y = title_h + row * (label_h + leaf)
        draw.text(
            (x + leaf // 2, y + label_h // 2),
            _severity_title(severity),
            fill="black",
            font=label_font,
            anchor="mm",
        )
        image.paste(Image.fromarray(panel), (x, y + label_h))
    draw.text(
        (canvas // 2, canvas - 192),
        "Tomato_healthy_img_1.jpg  ·  dev-yw-main-ref/dataset/validation/Tomato_healthy  ·  16384×16384",
        fill="black",
        font=foot_font,
        anchor="mm",
    )
    if image.size != (canvas, canvas):
        raise RuntimeError(f"bilateral legend size {image.size} is not 16384×16384")
    return image


def _draw_legend(original: np.ndarray, category: str, levels: tuple[float, ...]) -> plt.Figure:
    name = CATEGORY_EN[category]
    columns = 8 if category == "bilateral" else len(levels)
    rows = int(np.ceil(len(levels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.55, rows * 1.85 + 0.45))
    flat = np.atleast_1d(axes).ravel()
    for axis, severity in zip(flat, levels):
        corrupted = apply_corruption(original, category, severity, LEGEND_SHA256)
        axis.imshow(cv2.cvtColor(corrupted, cv2.COLOR_BGR2RGB), interpolation="nearest")
        axis.set_title(_severity_title(severity), fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in flat[len(levels) :]:
        axis.axis("off")
    fig.suptitle(f"{name}: reference", fontsize=12)
    fig.text(
        0.5,
        0.01,
        "Tomato_healthy_img_1.jpg  ·  dev-yw-main-ref/dataset/validation/Tomato_healthy",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.06, top=0.90, wspace=0.08, hspace=0.28)
    return fig


def render_severity_legends(out_dir: Path) -> None:
    """Validation leaf across severity. Bilateral uses a step of 2.5."""
    _theme()
    project = find_project(Path(__file__).resolve().parents[2])
    source = project / LEGEND_RELPATH
    if not source.is_file():
        raise RuntimeError(f"validation legend image is missing: {source}")
    original = load_bgr(source)
    figure_root = out_dir / "figures"
    for category in CATEGORIES:
        levels = _legend_levels(category)
        path = figure_root / category / f"05_{category}_reference.png"
        if category == "bilateral":
            temporary = path.with_name(f"{path.stem}.writing.png")
            _draw_bilateral_legend_hires(original, levels).save(temporary, format="PNG", compress_level=3)
            temporary.replace(path)
        else:
            _save(_draw_legend(original, category, levels), path)
        previous = figure_root / category / "05_severity_legend.png"
        if previous.exists() and previous != path:
            previous.unlink()
        print(f"图例已写入 {path}（{len(levels)} 档）", flush=True)


def render(out_dir: Path) -> None:
    _theme()
    predictions = load_predictions(out_dir)
    scores = pd.read_csv(out_dir / "scores.csv")
    classes = _class_order(predictions)
    figure_root = out_dir / "figures"
    for category in CATEGORIES:
        name = CATEGORY_EN[category]
        folder = figure_root / category
        group = predictions[predictions["category"] == category]
        table = scores[scores["category"] == category].sort_values("severity")
        if table["severity"].nunique() != 101:
            raise RuntimeError(f"{category} scores do not cover severity 0..100")
        _curve(group, table, f"{name}: F1 against Perturbation Severity", folder / "01_performance_curve.png")
        _violin(group, f"{name}: true-class decision score", folder / "02_violin_margin.png")
        _confusion(
            group,
            classes,
            f"{name}: confusion at Perturbation Severity 5, 10, 50, 70 and 90",
            folder / "03_confusion_compare.png",
        )
        _difference(group, classes, f"{name}: confusion difference from Perturbation Severity 0", folder / "04_difference_matrix.png")
        print(f"图已写入 {folder}", flush=True)
    render_severity_legends(out_dir)


def main() -> None:
    render(Path(__file__).resolve().parent / "outputs")


if __name__ == "__main__":
    main()
