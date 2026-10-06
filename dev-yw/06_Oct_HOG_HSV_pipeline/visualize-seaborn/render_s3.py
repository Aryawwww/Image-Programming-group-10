"""Seaborn figures for the twenty Frangi rounds. The preview image is separate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SEARCH = REPO / "outputs" / "search_s3"
OUT = ROOT / "s3"

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
ROW_ORDER = ["green dark", "green pale", "exg dark", "exg pale"]
SIGMA_ORDER = ["1", "1-2", "2-3", "1-2-3", "2-3-4"]


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


def _select_full(rounds: pd.DataFrame) -> pd.DataFrame:
    full = rounds[np.isclose(rounds["fraction"], 1.0) & (rounds["subset"] == "select")].copy()
    full["row"] = full["source"].astype(str) + " " + full["polarity"].astype(str)
    full["row"] = pd.Categorical(full["row"], ROW_ORDER, ordered=True)
    full["sigmas"] = pd.Categorical(full["sigmas"], SIGMA_ORDER, ordered=True)
    return full


def figure_heatmap(rounds: pd.DataFrame, winner: dict) -> None:
    full = _select_full(rounds)
    heat = full.pivot(index="row", columns="sigmas", values="min_class_auc")
    heat = heat.reindex(index=ROW_ORDER, columns=SIGMA_ORDER)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".3f",
        cmap=sns.light_palette("#1d4e89", as_cmap=True),
        vmin=0.5,
        vmax=1.0,
        linewidths=0.6,
        linecolor="white",
        annot_kws={"color": "black", "fontsize": 9},
        cbar_kws={"label": "Minimum Healthy-versus-one-class AUC"},
        ax=ax,
    )
    ax.set_xlabel("Frangi scales (pixels)")
    ax.set_ylabel("Source and ridge polarity")
    ax.set_title("S3 selection metric on the training subset")
    reference = float(winner["s1_reference_min_class_auc"])
    fig.text(
        0.01,
        0.01,
        f"Twenty Frangi settings. The fit uses the 18-number S1 summary, not the 34,596-d vector. C = 1, α = 0. "
        f"Winner: round {int(winner['winner_round'])}, {winner['source']} {winner['polarity']}, scales {winner['sigmas']}. "
        f"S1 round 7 on the same holdout was {reference:.3f} and did not vote. "
        "The test split was not used for this table. Validation was not read.",
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    _save(fig, "01_selection_auc.png")


def figure_per_class(rounds: pd.DataFrame, winner: dict) -> None:
    full = _select_full(rounds)
    long = full.melt(
        id_vars=["round", "row", "sigmas"],
        value_vars=list(DISEASE_LABEL),
        var_name="contrast",
        value_name="AUC",
    )
    long["Disease"] = long["contrast"].map(DISEASE_LABEL)
    grid = sns.relplot(
        data=long,
        kind="line",
        x="sigmas",
        y="AUC",
        hue="Disease",
        hue_order=list(DISEASE_LABEL.values()),
        palette=PALETTE,
        col="row",
        col_order=ROW_ORDER,
        marker="o",
        height=2.6,
        aspect=1.05,
    )
    grid.set_axis_labels("Frangi scales", "Healthy-versus-this-class AUC")
    grid.set_titles(col_template="{col_name}")
    grid.set(ylim=(0.45, 1.02))
    grid.figure.suptitle("Each disease against Healthy, on the selection subset", y=1.04, fontsize=14)
    grid.tight_layout()
    grid.figure.text(
        0.01,
        -0.06,
        f"Winner round {int(winner['winner_round'])} has the highest worst-class value. "
        "A macro-F1 is not drawn. The test split did not rank these lines.",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    grid.savefig(OUT / "02_per_class_auc.png", bbox_inches="tight", facecolor="white")
    plt.close(grid.figure)


def figure_learning_curves(rounds: pd.DataFrame) -> None:
    shown = rounds[rounds["subset"].isin(["fit", "select"])].copy()
    shown["Set"] = shown["subset"].map({"fit": "Fit subset", "select": "Selection subset"})
    shown["panel"] = (
        shown["source"].astype(str)
        + " "
        + shown["polarity"].astype(str)
        + " "
        + shown["sigmas"].astype(str)
    )
    order = [f"{row} {sigmas}" for row in ROW_ORDER for sigmas in SIGMA_ORDER]
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
    grid.figure.suptitle("Learning curves for the twenty ridge summaries", y=1.03, fontsize=14)
    grid.tight_layout()
    grid.figure.text(
        0.01,
        -0.02,
        "Each panel is one Frangi setting reduced to 18 numbers. "
        "The selection subset is 20% of train. Neither curve is the official test split. "
        "The raw 34,596-d descriptor was not fit, so a curve near 1 is not assumed.",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    grid.savefig(OUT / "03_learning_curves.png", bbox_inches="tight", facecolor="white")
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
    ax.set_xlabel("Published interior completeness (not recomputed from the ridge)")
    ax.set_ylabel("Healthy score from the ridge summary")
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
    figure_learning_curves(rounds)
    _cloud(
        pd.read_csv(SEARCH / "select_scores.csv").loc[lambda frame: frame["round"] == int(winner["winner_round"])],
        "04_winner_selection_cloud.png",
        f"S3 round {int(winner['winner_round'])} on the selection subset",
        f"{winner['source']} {winner['polarity']}, scales {winner['sigmas']}. "
        "Eighteen features, C = 1, α = 0. The horizontal axis is the published skeleton. "
        "This subset chose the round. Validation was not read.",
    )
    _cloud(
        pd.read_csv(SEARCH / "test_scores.csv"),
        "05_winner_test_diagnostic.png",
        f"S3 round {int(winner['winner_round'])} on the test split, not used to choose",
        "Drawn after winner.json. Septoria leaf spot is #77ff11. "
        "The original sixteen-round test macro-F1 did not rank these settings. Validation was not read.",
    )
    print(f"S3 figures in {OUT}", flush=True)


if __name__ == "__main__":
    main()
