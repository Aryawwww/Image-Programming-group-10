"""Seaborn figures for search S2. The published completeness column is not used."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SEARCH = REPO / "outputs" / "search_s2"
OUT = ROOT / "s2"

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
PERCENTILES = [80, 85, 90, 95, 97]
KERNELS = [3, 5, 7, 9]


def _theme() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 160,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _label(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["True class"] = pd.Categorical(
        out["class_name"].map(CLASS_LABEL),
        list(CLASS_LABEL.values()),
        ordered=True,
    )
    return out


def figure_heatmap(rounds: pd.DataFrame, winner: dict) -> None:
    heat = rounds.pivot(index="percentile", columns="open_kernel", values="min_class_auc")
    heat = heat.reindex(index=PERCENTILES, columns=KERNELS)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".3f",
        cmap=sns.light_palette("#1d4e89", as_cmap=True),
        vmin=0.40,
        vmax=1.0,
        linewidths=0.6,
        linecolor="white",
        annot_kws={"color": "black", "fontsize": 10},
        cbar_kws={"label": "Minimum Healthy-versus-one-class AUC"},
        ax=ax,
    )
    ax.set_xlabel("Open-kernel width (px), thick interior blobs removed before the skeleton")
    ax.set_ylabel("Interior gradient percentile kept as an edge")
    ax.set_title("S2 selection metric: completeness on the training subset")
    fig.text(
        0.01,
        0.01,
        f"Twenty skeleton settings. The classifier was not refit. "
        f"Winner: round {int(winner['winner_round'])}, percentile {winner['percentile']:g}, kernel {int(winner['open_kernel'])}. "
        "Percentile 90 and kernel 5 are the published pair; that column was not rewritten. "
        "The test split was not used for this table. Validation was not read.",
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    _save(fig, "01_selection_auc.png")


def figure_per_class(rounds: pd.DataFrame, winner: dict) -> None:
    long = rounds.melt(
        id_vars=["round", "percentile", "open_kernel"],
        value_vars=list(DISEASE_LABEL),
        var_name="contrast",
        value_name="AUC",
    )
    long["Disease"] = long["contrast"].map(DISEASE_LABEL)
    grid = sns.relplot(
        data=long,
        kind="line",
        x="open_kernel",
        y="AUC",
        hue="Disease",
        hue_order=list(DISEASE_LABEL.values()),
        palette=PALETTE,
        col="percentile",
        col_order=PERCENTILES,
        marker="o",
        height=2.5,
        aspect=0.85,
    )
    grid.set_axis_labels("Open-kernel width", "Healthy-versus-this-class AUC")
    grid.set_titles(col_template="percentile {col_name:g}")
    grid.set(ylim=(0.35, 1.02), xticks=KERNELS)
    grid.figure.suptitle("Completeness of each disease against Healthy", y=1.04, fontsize=14)
    grid.tight_layout()
    grid.figure.text(
        0.01,
        -0.04,
        f"Winner round {int(winner['winner_round'])} has the highest worst-class curve. "
        "A higher percentile keeps only the strongest interior edges. Validation was not read.",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    grid.savefig(OUT / "02_per_class_auc.png", bbox_inches="tight", facecolor="white")
    plt.close(grid.figure)


def _cloud(frame: pd.DataFrame, name: str, title: str, note: str) -> None:
    plot = _label(frame)
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    sns.scatterplot(
        data=plot,
        x="vein_completeness_interior",
        y="healthy_score",
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
    ax.set_xlabel("Interior vein completeness under the S2 winner (not scaled)")
    ax.set_ylabel("S1 healthy score (classifier not refit)")
    ax.set_title(title)
    ax.legend(title="True class", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.55))
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.16, 0.82, 1))
    _save(fig, name)


def main() -> None:
    _theme()
    rounds = pd.read_csv(SEARCH / "rounds.csv")
    winner = json.loads((SEARCH / "winner.json").read_text(encoding="utf-8"))
    figure_heatmap(rounds, winner)
    figure_per_class(rounds, winner)
    _cloud(
        pd.read_csv(SEARCH / "select_scores.csv"),
        "03_winner_selection_cloud.png",
        f"S2 round {int(winner['winner_round'])} on the selection subset",
        f"Percentile {winner['percentile']:g}, open kernel {int(winner['open_kernel'])}. "
        "The vertical axis is S1 round "
        f"{int(winner['s1_winner_round'])}'s Tomato_healthy score. "
        "This subset chose the skeleton. The published completeness column is unchanged. Validation was not read.",
    )
    _cloud(
        pd.read_csv(SEARCH / "test_scores.csv"),
        "04_winner_test_diagnostic.png",
        f"S2 round {int(winner['winner_round'])} on the test split, not used to choose",
        "Drawn after winner.json. Completeness AUC on this split did not rank the twenty settings. "
        "Septoria leaf spot is #77ff11. Validation was not read.",
    )
    print(f"S2 figures in {OUT}", flush=True)


if __name__ == "__main__":
    main()
