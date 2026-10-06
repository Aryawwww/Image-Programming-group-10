"""Seaborn figures for search S1.

The published sixteen-round files are not overwritten. The test-split
picture is drawn only from the already chosen winner.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
ANN = REPO / "outputs" / "annotations"
SEARCH = REPO / "outputs" / "search_s1"
OUT = ROOT / "s1"

CLASS_ORDER = [
    "Tomato_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Late_blight",
    "Tomato_Septoria_leaf_spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
]
CLASS_LABEL = {
    "Tomato_healthy": "Healthy",
    "Tomato_Bacterial_spot": "Bacterial spot",
    "Tomato_Late_blight": "Late blight",
    "Tomato_Septoria_leaf_spot": "Septoria leaf spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Yellow leaf curl virus",
}
DISEASE_LABEL = {
    "auc_Tomato_Bacterial_spot": "Bacterial spot",
    "auc_Tomato_Late_blight": "Late blight",
    "auc_Tomato_Septoria_leaf_spot": "Septoria leaf spot",
    "auc_Tomato__Tomato_YellowLeaf__Curl_Virus": "Yellow leaf curl virus",
}
PALETTE = {
    "Healthy": "#1d4e89",
    "Bacterial spot": "#d9480f",
    "Late blight": "#6f2dbd",
    "Septoria leaf spot": "#77ff11",
    "Yellow leaf curl virus": "#0e7490",
}
CURVE_PALETTE = {"Fit subset": "#1f4e79", "Selection subset": "#b85c38"}
POOL_ORDER = ["orient_mean", "orient_mean_std", "anisotropy", "entropy", "vein"]
BORDER_ORDER = [1, 2, 3, 4]


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


def _label_class(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["True class"] = out["class_name"].map(CLASS_LABEL)
    out["True class"] = pd.Categorical(out["True class"], list(CLASS_LABEL.values()), ordered=True)
    return out


def figure_published_axis() -> None:
    """Same test points as figure 06, but the vertical axis is the healthy score."""
    test = pd.read_csv(ANN / "test.csv")
    plot = _label_class(test)
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    sns.scatterplot(
        data=plot,
        x="vein_completeness_interior",
        y="score_Tomato_healthy",
        hue="True class",
        hue_order=list(CLASS_LABEL.values()),
        palette=PALETTE,
        s=42,
        alpha=0.92,
        edgecolor="#1a1a1a",
        linewidth=0.35,
        ax=ax,
    )
    ax.set_xlim(left=0)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("Interior vein completeness (length / leaf major axis, not scaled)")
    ax.set_ylabel("Decision score of Tomato_healthy")
    ax.set_title("Published round 10, redrawn on the healthy score")
    ax.legend(title="True class", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.55))
    fig.text(
        0.01,
        0.01,
        "No new fit. These are the 375 test images of the pre-specified round C = 1, α = 0.25. "
        "The vertical axis is that round's Tomato_healthy score for every leaf, not the leaf's own class. "
        "Own-class scores put a correctly classified diseased leaf on the same side as a healthy leaf. "
        "Symmetric log, linear threshold 100. Validation was not used.",
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.16, 0.82, 1))
    _save(fig, "00_published_round_healthy_score.png")


def _select_full(rounds: pd.DataFrame) -> pd.DataFrame:
    full = rounds[np.isclose(rounds["fraction"], 1.0) & (rounds["subset"] == "select")].copy()
    full["pool"] = pd.Categorical(full["pool"], POOL_ORDER, ordered=True)
    full["border"] = pd.Categorical(full["border"], BORDER_ORDER, ordered=True)
    return full


def figure_heatmap(rounds: pd.DataFrame, winner: dict) -> None:
    full = _select_full(rounds)
    heat = full.pivot(index="pool", columns="border", values="min_class_auc")
    heat = heat.reindex(index=POOL_ORDER, columns=BORDER_ORDER)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".3f",
        cmap=sns.light_palette("#1d4e89", as_cmap=True),
        vmin=0.5,
        vmax=1.0,
        linewidths=0.6,
        linecolor="white",
        annot_kws={"color": "black", "fontsize": 10},
        cbar_kws={"label": "Minimum Healthy-versus-one-class AUC"},
        ax=ax,
    )
    ax.set_xlabel("Outer blocks dropped on each side")
    ax.set_ylabel("Summary of the cached HOG blocks")
    ax.set_title("S1 selection metric on the training subset")
    fig.text(
        0.01,
        0.01,
        f"Twenty rounds. C = 1 and α = 0 are fixed, so this is not a new cell of the sixteen-round grid. "
        f"The number is the worst of the four Healthy-versus-one-disease AUCs on the held-out 20% of train. "
        f"Winner: round {int(winner['winner_round'])}, {winner['pool']}, border {int(winner['border'])}. "
        "The test split was not used for this table. Validation was not read.",
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    _save(fig, "01_selection_auc.png")


def figure_per_class(rounds: pd.DataFrame, winner: dict) -> None:
    full = _select_full(rounds)
    long = full.melt(
        id_vars=["round", "border", "pool"],
        value_vars=list(DISEASE_LABEL),
        var_name="contrast",
        value_name="AUC",
    )
    long["Disease"] = long["contrast"].map(DISEASE_LABEL)
    long["border"] = long["border"].astype(int)
    grid = sns.relplot(
        data=long,
        kind="line",
        x="border",
        y="AUC",
        hue="Disease",
        hue_order=list(DISEASE_LABEL.values()),
        palette=PALETTE,
        col="pool",
        col_order=POOL_ORDER,
        marker="o",
        height=2.5,
        aspect=0.85,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("Outer blocks dropped", "Healthy-versus-this-class AUC")
    grid.set_titles(col_template="{col_name}")
    grid.set(ylim=(0.45, 1.02), xticks=BORDER_ORDER)
    grid.figure.suptitle("Each disease against Healthy, on the selection subset", y=1.04, fontsize=14)
    grid.tight_layout()
    grid.figure.text(
        0.01,
        -0.04,
        f"The winner is round {int(winner['winner_round'])} because its lowest class curve is the highest of the twenty lows. "
        "A single macro-F1 is not drawn. The test split did not rank these lines.",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    grid.savefig(OUT / "02_per_class_auc.png", bbox_inches="tight", facecolor="white")
    plt.close(grid.figure)


def figure_learning_curves(rounds: pd.DataFrame) -> None:
    shown = rounds[rounds["subset"].isin(["fit", "select"])].copy()
    shown["Set"] = shown["subset"].map({"fit": "Fit subset", "select": "Selection subset"})
    shown["panel"] = shown["pool"].astype(str) + "  border " + shown["border"].astype(int).astype(str)
    order = [f"{pool}  border {border}" for border in BORDER_ORDER for pool in POOL_ORDER]
    shown["panel"] = pd.Categorical(shown["panel"], order, ordered=True)
    grid = sns.relplot(
        data=shown,
        kind="line",
        x="fraction",
        y="min_class_auc",
        hue="Set",
        hue_order=["Fit subset", "Selection subset"],
        col="panel",
        col_wrap=5,
        marker="o",
        palette=CURVE_PALETTE,
        height=2.15,
        aspect=1.05,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("Fraction of the fit subset", "Minimum class AUC")
    grid.set_titles(col_template="{col_name}")
    grid.set(ylim=(0.45, 1.02), xlim=(0.15, 1.05))
    grid.figure.suptitle("Learning curves for the twenty S1 rounds", y=1.03, fontsize=14)
    grid.tight_layout()
    grid.figure.text(
        0.01,
        -0.02,
        "Each panel is one pre-specified summary. The fit subset is the 80% of train used to place the hyperplane. "
        "The selection subset is the other 20% of train. Neither curve is the official test split. "
        "A fit curve near 1 would mean the summary still memorises; these summaries have 2 to 18 numbers, not 34,596.",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    grid.savefig(OUT / "03_learning_curves.png", bbox_inches="tight", facecolor="white")
    plt.close(grid.figure)


def _cloud(frame: pd.DataFrame, score_column: str, path_name: str, title: str, note: str) -> None:
    plot = _label_class(frame)
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    sns.scatterplot(
        data=plot,
        x="vein_completeness_interior",
        y=score_column,
        hue="True class",
        hue_order=list(CLASS_LABEL.values()),
        palette=PALETTE,
        s=42,
        alpha=0.92,
        edgecolor="#1a1a1a",
        linewidth=0.35,
        ax=ax,
    )
    ax.set_xlim(left=0)
    ax.set_xlabel("Interior vein completeness (length / leaf major axis, not scaled)")
    ax.set_ylabel("Decision score of Tomato_healthy")
    ax.set_title(title)
    ax.legend(title="True class", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.55))
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.16, 0.82, 1))
    _save(fig, path_name)


def figure_clouds(winner: dict) -> None:
    scores = pd.read_csv(SEARCH / "select_scores.csv")
    spec = pd.read_csv(SEARCH / "rounds.csv")
    spec = spec[np.isclose(spec["fraction"], 1.0) & (spec["subset"] == "select")][
        ["round", "border", "pool"]
    ].drop_duplicates()
    plot = scores.merge(spec, on="round", how="left")
    plot = _label_class(plot)
    plot["pool"] = pd.Categorical(plot["pool"], POOL_ORDER, ordered=True)
    grid = sns.relplot(
        data=plot,
        kind="scatter",
        x="vein_completeness_interior",
        y="healthy_score",
        hue="True class",
        hue_order=list(CLASS_LABEL.values()),
        palette=PALETTE,
        col="pool",
        row="border",
        col_order=POOL_ORDER,
        row_order=BORDER_ORDER,
        s=12,
        alpha=0.75,
        edgecolor=None,
        height=2.35,
        aspect=1.05,
        facet_kws={"sharex": True, "sharey": False, "margin_titles": True},
    )
    grid.set_axis_labels("Interior completeness", "Healthy score")
    grid.set_titles(row_template="border {row_name}", col_template="{col_name}")
    grid.figure.suptitle("Selection subset: completeness against the healthy score", y=1.02, fontsize=14)
    grid.tight_layout()
    grid.figure.text(
        0.01,
        -0.01,
        "Each point is one leaf from the 20% training holdout. The horizontal axis is the same interior skeleton in every panel. "
        f"The winner is round {int(winner['winner_round'])} ({winner['pool']}, border {int(winner['border'])}). "
        "Septoria leaf spot is #77ff11. This grid was not drawn from the test split.",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    grid.savefig(OUT / "04_selection_clouds.png", bbox_inches="tight", facecolor="white")
    plt.close(grid.figure)

    chosen = scores[scores["round"] == int(winner["winner_round"])]
    _cloud(
        chosen,
        "healthy_score",
        "05_winner_selection_cloud.png",
        f"S1 round {int(winner['winner_round'])} on the selection subset",
        f"Winner {winner['pool']}, border {int(winner['border'])}, {int(winner['n_features'])} features, C = 1, α = 0. "
        "The vertical axis is the Tomato_healthy score, including for diseased leaves. "
        "Completeness is not scaled to 0.80. This subset chose the round. Validation was not read.",
    )
    test = pd.read_csv(SEARCH / "test_scores.csv")
    _cloud(
        test,
        "healthy_score",
        "06_winner_test_diagnostic.png",
        f"S1 round {int(winner['winner_round'])} on the test split, not used to choose",
        "Drawn after winner.json was written. The test macro-F1 of the original sixteen rounds did not rank S1. "
        "A low or overlapping class is left as its own colour. Validation was not read.",
    )


def main() -> None:
    _theme()
    figure_published_axis()
    if not (SEARCH / "winner.json").exists():
        print("S1 figures skipped: winner.json is not written yet", flush=True)
        return
    rounds = pd.read_csv(SEARCH / "rounds.csv")
    winner = json.loads((SEARCH / "winner.json").read_text(encoding="utf-8"))
    figure_heatmap(rounds, winner)
    figure_per_class(rounds, winner)
    figure_learning_curves(rounds)
    figure_clouds(winner)
    print(f"S1 figures in {OUT}", flush=True)


if __name__ == "__main__":
    main()
