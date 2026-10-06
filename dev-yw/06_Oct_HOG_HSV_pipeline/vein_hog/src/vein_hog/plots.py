"""Seaborn figures for the 16 candidate rounds. Matplotlib is only the canvas."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _theme() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def save_round_comparison(curve: pd.DataFrame, out_dir: Path) -> None:
    """curve columns: round, C, alpha, fraction, split, macro_f1."""
    _theme()
    out_dir.mkdir(parents=True, exist_ok=True)
    full = curve[np.isclose(curve["fraction"], 1.0)].copy()
    full["round_label"] = full.apply(
        lambda row: f"r{int(row['round']):02d}\nC={row['C']:g}\nα={row['alpha']:g}",
        axis=1,
    )

    bar = sns.catplot(
        data=full,
        kind="bar",
        x="round",
        y="macro_f1",
        hue="split",
        height=5,
        aspect=2.4,
        palette="deep",
    )
    bar.set_axis_labels("candidate round", "macro-F1")
    bar.figure.suptitle("Train versus test fit across 16 rounds", y=1.03)
    bar.savefig(out_dir / "rounds_train_vs_test.png", dpi=140, bbox_inches="tight")
    plt.close(bar.figure)

    test = full[full["split"] == "test"]
    heat = test.pivot(index="C", columns="alpha", values="macro_f1")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title("Test macro-F1 by SVM C and HSV alpha")
    ax.set_xlabel("HSV alpha")
    ax.set_ylabel("SVM C")
    fig.tight_layout()
    fig.savefig(out_dir / "rounds_heatmap.png", dpi=140)
    plt.close(fig)

    grid = sns.relplot(
        data=curve,
        kind="line",
        x="fraction",
        y="macro_f1",
        hue="split",
        col="alpha",
        row="C",
        marker="o",
        height=2.3,
        aspect=1.15,
        facet_kws={"margin_titles": True},
        palette="deep",
    )
    grid.set_axis_labels("train fraction", "macro-F1")
    grid.figure.suptitle("Learning curves for all 16 rounds", y=1.02)
    grid.savefig(out_dir / "learning_curve.png", dpi=140, bbox_inches="tight")
    plt.close(grid.figure)


def save_selected_diagnostics(
    confusion: np.ndarray,
    class_names: list[str],
    per_class: pd.DataFrame,
    scores: pd.DataFrame,
    out_dir: Path,
) -> None:
    _theme()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Test confusion matrix of the pre-specified round")
    fig.tight_layout()
    fig.savefig(out_dir / "selected_confusion.png", dpi=140)
    plt.close(fig)

    melted = per_class.melt(
        id_vars=["class_name"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="class_name", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Test precision, recall and F1 by class, pre-specified round")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / "selected_per_class.png", dpi=140)
    plt.close(fig)

    if "vein_completeness_interior" not in scores.columns or "class_score" not in scores.columns:
        return
    drawn = scores
    if "split" in scores.columns:
        drawn = scores[scores["split"] == "test"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=drawn,
        x="vein_completeness_interior",
        y="class_score",
        hue="class_name",
        ax=ax,
        s=28,
        edgecolor="#1a1a1a",
        linewidth=0.3,
    )
    ax.set_xlabel("Uncalibrated interior completeness")
    ax.set_ylabel("Score of the leaf's own class")
    ax.set_title("Interior completeness and per-class score")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_dir / "completeness_vs_class_score.png", dpi=140)
    plt.close(fig)
