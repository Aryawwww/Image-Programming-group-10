"""Two seaborn figures for the S6 winner. Both use training rows only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
SEARCH = ROOT.parent / "outputs" / "search_s6"
OUT = ROOT / "s6"

CLASS_LABEL = {
    "Tomato_Bacterial_spot": "Bacterial Spot",
    "Tomato_Late_blight": "Late Blight",
    "Tomato_Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato_healthy": "Healthy",
}
DISEASE_LABEL = {
    "auc_Tomato_Bacterial_spot": "Bacterial Spot",
    "auc_Tomato_Late_blight": "Late Blight",
    "auc_Tomato_Septoria_leaf_spot": "Septoria Leaf Spot",
    "auc_Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
}
PALETTE = {
    "Bacterial Spot": "#d9480f",
    "Late Blight": "#6f2dbd",
    "Septoria Leaf Spot": "#77ff11",
    "Tomato Yellow Leaf Curl Virus": "#0e7490",
    "Healthy": "#1d4e89",
}
CURVE_PALETTE = {"Fit subset": "#1f4e79", "Selection subset": "#b85c38"}


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


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _winner() -> tuple[dict, pd.DataFrame, float]:
    payload = json.loads((SEARCH / "winner.json").read_text(encoding="utf-8"))
    if payload.get("test_was_read") or payload.get("validation_was_read"):
        raise RuntimeError("S6 figures refuse a winner that read test or validation")
    if payload.get("replaces_report_round") or payload.get("replaces_s4_winner"):
        raise RuntimeError("S6 figures refuse a winner that replaces a frozen model")
    rounds = pd.read_csv(SEARCH / "rounds.csv")
    if not set(rounds["subset"]).issubset({"fit", "select"}):
        raise RuntimeError("rounds.csv contains a held-out split")
    chosen = rounds[
        (rounds["round"] == int(payload["winner_round"]))
        & np.isclose(rounds["fraction"].astype(float), 1.0)
        & (rounds["subset"] == "select")
    ]
    if len(chosen) != 1:
        raise RuntimeError("winner select row is missing from rounds.csv")
    select_min = float(chosen.iloc[0]["min_class_auc"])
    if abs(select_min - float(payload["select_min_class_auc"])) > 1e-9:
        raise RuntimeError("winner select AUC does not match rounds.csv")
    return payload, rounds, select_min


def _metric_row(metrics: pd.DataFrame, subset: str) -> pd.Series:
    chosen = metrics[np.isclose(metrics["fraction"].astype(float), 1.0) & (metrics["subset"] == subset)]
    if len(chosen) != 1:
        raise RuntimeError(f"missing fraction-1 {subset} metrics")
    return chosen.iloc[0]


def _format_metrics(row: pd.Series) -> str:
    names = ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1-score")
    return "   ".join(f"{name} {float(row[name]):.3f}" for name in names)


def figure_training_curves(rounds: pd.DataFrame, winner: dict, select_min: float, metrics: pd.DataFrame) -> None:
    shown = rounds[rounds["round"] == int(winner["winner_round"])].copy()
    long = shown.melt(
        id_vars=["fraction", "subset"],
        value_vars=list(DISEASE_LABEL),
        var_name="score",
        value_name="auc",
    )
    long["Set"] = long["subset"].map({"fit": "Fit subset", "select": "Selection subset"})
    long["Disease"] = long["score"].map(DISEASE_LABEL)
    grid = sns.relplot(
        data=long,
        kind="line",
        x="fraction",
        y="auc",
        hue="Set",
        hue_order=["Fit subset", "Selection subset"],
        col="Disease",
        col_order=list(DISEASE_LABEL.values()),
        marker="o",
        palette=CURVE_PALETTE,
        height=2.8,
        aspect=1.05,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("Fraction of the fit subset", "Healthy-versus-disease AUC")
    grid.set_titles(col_template="{col_name}")
    grid.set(ylim=(0.0, 1.02), xlim=(0.15, 1.05))
    grid.figure.suptitle("S6 winner during training", y=1.04, fontsize=14)
    grid.tight_layout()
    grid.figure.text(
        0.01,
        -0.04,
        f"Round {int(winner['winner_round'])}, {winner['hog']}, HSV layout {winner['hsv_layout']}, "
        f"weight {float(winner['hsv_alpha']):.2f}. "
        "Both lines are inside the training split. The fit line can sit near 1 because the descriptor separates those rows. "
        f"The selection line at fraction 1 has minimum AUC {select_min:.3f}, and that number chose the round. "
        f"Fit subset at fraction 1: {_format_metrics(_metric_row(metrics, 'fit'))}. "
        "Test and validation were not read.",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    grid.savefig(OUT / "01_training_curves.png", bbox_inches="tight", facecolor="white")
    plt.close(grid.figure)


def _style_table(table, fontsize: float) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, _col), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#c8c8c8")
        if row == 0:
            cell.set_facecolor("#e8eef4")
            cell.set_text_props(weight="normal")
        else:
            cell.set_facecolor("#ffffff")


def render_fit_accuracy(
    frame: pd.DataFrame,
    winner: dict,
    select_min: float,
    metrics: pd.DataFrame,
    per_class: pd.DataFrame,
) -> plt.Figure:
    """Bars plus the overall and per-class tables. Training rows only."""
    if set(frame["subset"]) != {"fit"}:
        raise RuntimeError("fit predictions are not limited to the fit subset")
    if len(frame) != 2400:
        raise RuntimeError(f"expected 2400 fit rows, got {len(frame)}")
    labels = list(CLASS_LABEL)
    precision, recall, f1, _support = precision_recall_fscore_support(
        frame["class_name"],
        frame["predicted_class"],
        labels=labels,
        zero_division=0,
    )
    rows = []
    for name, prec, rec, score in zip(labels, precision, recall, f1):
        label = CLASS_LABEL[name]
        rows.extend(
            (
                {"Class": label, "Metric": "Precision", "Score": float(prec)},
                {"Class": label, "Metric": "Recall", "Score": float(rec)},
                {"Class": label, "Metric": "F1-score", "Score": float(score)},
            )
        )
    plot = pd.DataFrame(rows)
    fig = plt.figure(figsize=(11.2, 8.6))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[4.6, 1.45, 2.15],
        left=0.08,
        right=0.74,
        top=0.90,
        bottom=0.16,
        hspace=0.42,
    )
    ax = fig.add_subplot(grid[0])
    sns.barplot(
        data=plot,
        x="Metric",
        y="Score",
        hue="Class",
        hue_order=list(CLASS_LABEL.values()),
        palette=PALETTE,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score on the fit subset")
    ax.set_xlabel("")
    ax.set_title("S6 winner after training, on the rows used to fit")
    ax.legend(title="True class", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fit_row = _metric_row(metrics, "fit")
    select_row = _metric_row(metrics, "select")
    cell_text = [
        [name, f"{float(fit_row[name]):.3f}", f"{float(select_row[name]):.3f}"]
        for name in ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1-score")
    ]
    macro_ax = fig.add_subplot(grid[1])
    macro_ax.axis("off")
    table = macro_ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Fit subset", "Selection subset"],
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    _style_table(table, 8.5)
    fit_classes = per_class[np.isclose(per_class["fraction"].astype(float), 1.0) & (per_class["subset"] == "fit")].copy()
    class_order = list(CLASS_LABEL.values())
    if fit_classes["Class"].tolist() != class_order:
        fit_classes = fit_classes.set_index("Class").loc[class_order].reset_index()
    ordered = fit_classes.loc[:, ["Class", "Precision", "Recall", "F1-score"]]
    class_text = [
        [str(klass), f"{float(prec):.3f}", f"{float(rec):.3f}", f"{float(score):.3f}"]
        for klass, prec, rec, score in ordered.itertuples(index=False, name=None)
    ]
    class_ax = fig.add_subplot(grid[2])
    class_ax.axis("off")
    class_table = class_ax.table(
        cellText=class_text,
        colLabels=["Class", "Precision", "Recall", "F1-score"],
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    _style_table(class_table, 8.5)
    fig.text(
        0.08,
        0.015,
        (
            f"2400 training rows, round {int(winner['winner_round'])}, {winner['hog']}, "
            f"HSV layout {winner['hsv_layout']}, weight {float(winner['hsv_alpha']):.2f}. "
            "The fit column is the rows used to place the hyperplane. A value of 1.0 means those rows are linearly separable.\n"
            f"Selection minimum AUC on the other 20% of train is {select_min:.3f}. "
            "Per-class Precision, Recall, and F1-score are for the fit subset. "
            "Fraction-by-fraction scores are in metrics.csv and metrics_per_class.csv.\n"
            "Test, validation, and the robustness sweep were not used. Septoria Leaf Spot is #77ff11."
        ),
        ha="left",
        va="bottom",
        fontsize=8,
    )
    return fig


def figure_fit_accuracy(winner: dict, select_min: float, metrics: pd.DataFrame, per_class: pd.DataFrame) -> None:
    frame = pd.read_csv(SEARCH / "fit_predictions.csv")
    fig = render_fit_accuracy(frame, winner, select_min, metrics, per_class)
    _save(fig, "02_fit_subset_accuracy.png")


def main() -> None:
    from vein_hog.run import find_project
    from vein_hog.search_s6 import _pipeline_root, write_classification_metrics

    _theme()
    OUT.mkdir(parents=True, exist_ok=True)
    root = _pipeline_root()
    write_classification_metrics(root / "outputs" / "cache", SEARCH, find_project(root))
    winner, rounds, select_min = _winner()
    metrics = pd.read_csv(SEARCH / "metrics.csv")
    per_class = pd.read_csv(SEARCH / "metrics_per_class.csv")
    figure_training_curves(rounds, winner, select_min, metrics)
    figure_fit_accuracy(winner, select_min, metrics, per_class)
    written = sorted(path.name for path in OUT.glob("*.png"))
    if written != ["01_training_curves.png", "02_fit_subset_accuracy.png"]:
        raise RuntimeError(f"expected two figures, found {written}")
    print("S6_FIGURES", *written, flush=True)


if __name__ == "__main__":
    main()
