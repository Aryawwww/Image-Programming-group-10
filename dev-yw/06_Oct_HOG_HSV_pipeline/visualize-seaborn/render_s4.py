"""Seaborn figures for search S4. HOG and HSV only. The test picture is last."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SEARCH = REPO / "outputs" / "search_s4"
OUT = ROOT / "s4"

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
HOG_ORDER = ["ori9-cell8", "ori9-cell16", "ori6-cell16", "ori12-cell16"]


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


def _select_full(rounds: pd.DataFrame) -> pd.DataFrame:
    full = rounds[np.isclose(rounds["fraction"].astype(float), 1.0) & (rounds["subset"] == "select")].copy()
    full["hog"] = pd.Categorical(full["hog"], HOG_ORDER, ordered=True)
    full["α"] = full["hsv_alpha"].map(lambda value: f"{float(value):.3f}")
    return full


def figure_heatmap(rounds: pd.DataFrame, winner: dict) -> None:
    full = _select_full(rounds)
    alpha_order = list(dict.fromkeys(full.sort_values("hsv_alpha")["α"].tolist()))
    heat = full.pivot(index="hog", columns="α", values="min_class_auc")
    heat = heat.reindex(index=HOG_ORDER, columns=alpha_order)
    fig, ax = plt.subplots(figsize=(16.5, 4.6))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap=sns.light_palette("#1d4e89", as_cmap=True),
        vmin=0.5,
        vmax=1.0,
        linewidths=0.6,
        linecolor="white",
        annot_kws={"color": "black", "fontsize": 6},
        cbar_kws={"label": "Minimum Healthy-versus-one-class AUC"},
        ax=ax,
    )
    ax.set_xlabel("HSV auxiliary weight")
    ax.set_ylabel("HOG orientations and cell size")
    ax.set_title("S4 selection metric on the training subset")
    fig.text(
        0.01,
        0.01,
        "C = 1. HSV weights run from 0.25 to 0.50 by steps of 0.01, four at a time. "
        f"Winner: round {int(winner['winner_round'])}, {winner['hog']}, α = {float(winner['hsv_alpha']):.3f}. "
        "The test split was not used for this table. Validation was not read.",
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, "01_selection_auc.png")


def figure_per_class(rounds: pd.DataFrame, winner: dict) -> None:
    full = _select_full(rounds)
    long = full.melt(
        id_vars=["round", "hog", "α"],
        value_vars=list(DISEASE_LABEL),
        var_name="contrast",
        value_name="AUC",
    )
    long["Disease"] = long["contrast"].map(DISEASE_LABEL)
    grid = sns.relplot(
        data=long,
        kind="line",
        x="α",
        y="AUC",
        hue="Disease",
        hue_order=list(DISEASE_LABEL.values()),
        palette=PALETTE,
        col="hog",
        col_order=HOG_ORDER,
        marker="o",
        height=2.8,
        aspect=0.95,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("HSV weight", "Healthy-versus-this-class AUC")
    grid.set_titles(col_template="{col_name}")
    grid.set(ylim=(0.45, 1.02))
    grid.figure.suptitle("Each disease against Healthy, on the selection subset", y=1.04, fontsize=14)
    grid.tight_layout()
    grid.figure.text(
        0.01,
        -0.06,
        f"The winner is round {int(winner['winner_round'])} because its lowest class curve is the highest of the lows. "
        "A single macro-F1 is not drawn. The test split did not rank these lines.",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    grid.savefig(OUT / "02_per_class_auc.png", bbox_inches="tight", facecolor="white")
    plt.close(grid.figure)


def figure_learning_curves(rounds: pd.DataFrame, winner: dict) -> None:
    shown = rounds[rounds["subset"].isin(["fit", "select"]) & (rounds["hog"] == winner["hog"])].copy()
    shown["Set"] = shown["subset"].map({"fit": "Fit subset", "select": "Selection subset"})
    shown["panel"] = shown["hog"].astype(str) + "  α " + shown["hsv_alpha"].map(lambda value: f"{float(value):.3f}")
    order = []
    for hog in HOG_ORDER:
        part = shown[shown["hog"] == hog].sort_values("hsv_alpha")
        for label in part["panel"].drop_duplicates().tolist():
            order.append(label)
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
    grid.figure.suptitle("Learning curves for the S4 rounds", y=1.03, fontsize=14)
    grid.tight_layout()
    grid.figure.text(
        0.01,
        -0.02,
        "The fit subset is the 80% of train used to place the hyperplane. The selection subset is the other 20% of train. "
        "Neither curve is the official test split. A fit curve near 1 means that geometry still memorises the training rows.",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    grid.savefig(OUT / "03_learning_curves.png", bbox_inches="tight", facecolor="white")
    plt.close(grid.figure)


def _cloud(frame: pd.DataFrame, path_name: str, title: str, note: str) -> None:
    plot = _label_class(frame)
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
    ax.set_xlabel("Interior vein completeness (length / leaf major axis, not scaled)")
    ax.set_ylabel("Decision score of Tomato_healthy")
    ax.set_title(title)
    ax.legend(title="True class", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.55))
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.16, 0.82, 1))
    _save(fig, path_name)


def figure_clouds(winner: dict) -> None:
    scores = pd.read_csv(SEARCH / "select_scores.csv")
    chosen = scores[scores["round"] == int(winner["winner_round"])]
    _cloud(
        chosen,
        "04_winner_selection_cloud.png",
        f"S4 round {int(winner['winner_round'])} on the selection subset",
        f"Winner {winner['hog']}, {int(winner['orientations'])} orientations, "
        f"{int(winner['pixels_per_cell'])}px cells, HSV α = {float(winner['hsv_alpha']):.3f}. "
        "Septoria leaf spot is #77ff11. This subset chose the round. Validation was not read.",
    )
    test = pd.read_csv(SEARCH / "test_scores.csv")
    _cloud(
        test,
        "05_winner_test_diagnostic.png",
        f"S4 round {int(winner['winner_round'])} on the test split, not used to choose",
        "Drawn after winner.json. The test scores did not rank the rounds. Validation was not read.",
    )


def figure_validation(winner: dict) -> None:
    path = SEARCH / "validation_scores.csv"
    if not path.exists():
        return
    _cloud(
        pd.read_csv(path),
        "06_winner_validation.png",
        f"S4 round {int(winner['winner_round'])} on the validation split",
        f"Winner {winner['hog']}, HSV α = {float(winner['hsv_alpha']):.2f}, frozen before this split was read. "
        "These points did not choose the round. Septoria leaf spot is #77ff11.",
    )


def main() -> None:
    _theme()
    if not (SEARCH / "winner.json").exists():
        print("S4 figures skipped: winner.json is not written yet", flush=True)
        return
    rounds = pd.read_csv(SEARCH / "rounds.csv")
    winner = json.loads((SEARCH / "winner.json").read_text(encoding="utf-8"))
    figure_heatmap(rounds, winner)
    figure_per_class(rounds, winner)
    figure_learning_curves(rounds, winner)
    if (SEARCH / "test_scores.csv").exists():
        figure_clouds(winner)
    figure_validation(winner)
    print(f"S4 figures in {OUT}", flush=True)


if __name__ == "__main__":
    main()
