"""Formal English figures for the refit of the sixteen HOG–HSV rounds.

Numbers are read from the completed run. This script does not refit models
and does not open the validation split. The drawn round is the pre-specified
pair C=1, alpha=0.25, not the highest test score.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
FIT = REPO / "outputs" / "fit"
ANN = REPO / "outputs" / "annotations"
OUT = ROOT

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
SPLIT_LABEL = {"train": "Training set", "test": "Test set"}
C_LEVELS = [0.01, 0.1, 1.0, 10.0]
ALPHA_LEVELS = [0.0, 0.25, 0.5, 1.0]
PALETTE = {"Training set": "#1f4e79", "Test set": "#b85c38"}
RISK_PALETTE = {
    "Healthy": "#1d4e89",
    "Bacterial spot": "#d9480f",
    "Late blight": "#6f2dbd",
    "Septoria leaf spot": "#77ff11",
    "Yellow leaf curl virus": "#0e7490",
}
GATE_F1 = 0.90
REPORT_C = 1.0
REPORT_ALPHA = 0.25


def _theme() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "axes.titlepad": 8,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    curve = pd.read_csv(FIT / "learning_curve.csv")
    curve["Set"] = curve["split"].map(SPLIT_LABEL)
    curve["C"] = curve["C"].astype(float)
    curve["alpha"] = curve["alpha"].astype(float)
    full = curve[np.isclose(curve["fraction"], 1.0)].copy()
    test = pd.read_csv(ANN / "test.csv")
    test["True class"] = test["class_name"].map(CLASS_LABEL)
    test["Predicted class"] = test["predicted_class"].map(CLASS_LABEL)
    metrics = json.loads((FIT / "metrics.json").read_text(encoding="utf-8"))
    return curve, full, test, metrics


def _report_row(full: pd.DataFrame) -> pd.Series:
    rows = full[
        np.isclose(full["C"], REPORT_C)
        & np.isclose(full["alpha"], REPORT_ALPHA)
        & (full["Set"] == "Test set")
    ]
    if len(rows) != 1:
        raise SystemExit("learning curve is missing the pre-specified round")
    return rows.iloc[0]


def figure_train_versus_test(full: pd.DataFrame) -> None:
    wide = (
        full.pivot(index="round", columns="Set", values="macro_f1")
        .reset_index()
        .sort_values("round")
    )
    wide["Absolute gap"] = (wide["Training set"] - wide["Test set"]).abs()
    long = full.copy()
    long["round"] = long["round"].astype(int)
    report = _report_row(full)
    train_report = full[
        np.isclose(full["C"], REPORT_C)
        & np.isclose(full["alpha"], REPORT_ALPHA)
        & (full["Set"] == "Training set")
    ].iloc[0]
    highest = full[full["Set"] == "Test set"]["macro_f1"].max()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), gridspec_kw={"width_ratios": [1.45, 1]})
    sns.barplot(
        data=long,
        x="round",
        y="macro_f1",
        hue="Set",
        hue_order=["Training set", "Test set"],
        palette=PALETTE,
        ax=axes[0],
    )
    axes[0].axhline(GATE_F1, color="#6b3a3a", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_xlabel("Candidate round")
    axes[0].set_ylabel("Macro-averaged F1")
    score_handles, _ = axes[0].get_legend_handles_labels()
    score_handles.append(Line2D([0], [0], color="#6b3a3a", linestyle="--", linewidth=1))
    axes[0].legend(
        score_handles,
        ["Training set", "Test set", "Closed gate, 0.90"],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=3,
        borderaxespad=0,
    )
    sns.barplot(data=wide, x="round", y="Absolute gap", color="#4d4d4d", ax=axes[1])
    axes[1].axhline(0.05, color="#6b3a3a", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, max(0.50, float(wide["Absolute gap"].max()) + 0.05))
    axes[1].set_xlabel("Candidate round")
    axes[1].set_ylabel("Absolute difference in macro-averaged F1")
    axes[1].legend(
        [Line2D([0], [0], color="#6b3a3a", linestyle="--", linewidth=1)],
        ["Closed gate, gap 0.05"],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0,
    )
    fig.suptitle(
        "Sixteen refit rounds of the HOG-primary, HSV-auxiliary linear classifier",
        y=0.98,
        fontsize=14,
    )
    fig.text(
        0.01,
        0.01,
        "Left: macro-averaged F1 on the 3,000 training images and the 375 test images. "
        "Right: the absolute gap. The dashed lines are the closed validation gate, not a claim that it was met. "
        f"The highest test bar on this chart is {highest:.3f}. It was not used to choose a round. "
        f"The class charts use the pre-specified round {int(report['round'])} "
        f"(C = {REPORT_C:g}, α = {REPORT_ALPHA:g}), "
        f"training {float(train_report['macro_f1']):.3f}, test {float(report['macro_f1']):.3f}. "
        "Validation was not read.",
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.24, wspace=0.32)
    _save(fig, "01_training_and_test_macro_f1.png")


def figure_heatmap(full: pd.DataFrame) -> None:
    test = full[full["Set"] == "Test set"]
    heat = test.pivot(index="C", columns="alpha", values="macro_f1")
    heat = heat.reindex(index=C_LEVELS, columns=ALPHA_LEVELS)
    heat.index = [f"{value:g}" for value in heat.index]
    heat.columns = [f"{value:g}" for value in heat.columns]
    lo = float(np.nanmin(heat.to_numpy()))
    hi = float(np.nanmax(heat.to_numpy()))
    if hi - lo < 0.02:
        lo = max(0.0, lo - 0.01)
        hi = min(1.0, hi + 0.01)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".3f",
        cmap=sns.light_palette("#2f5d8a", as_cmap=True),
        vmin=lo,
        vmax=hi,
        linewidths=0.6,
        linecolor="white",
        annot_kws={"color": "black", "fontsize": 10},
        cbar_kws={"label": "Test macro-averaged F1"},
        ax=ax,
    )
    ax.set_xlabel("HSV auxiliary weight, α")
    ax.set_ylabel("Linear SVM regularization, C")
    ax.set_title("Test macro-averaged F1 for every pair of C and α")
    fig.text(
        0.01,
        0.01,
        f"Colour covers the observed interval {lo:.3f}–{hi:.3f}. "
        "No cell is a selected winner. The pre-specified report is C = 1, α = 0.25. "
        "The validation split was not used.",
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _save(fig, "02_test_macro_f1_by_C_and_hsv_weight.png")


def figure_learning_curves(curve: pd.DataFrame) -> None:
    shown = curve.copy()
    shown["C"] = pd.Categorical(shown["C"], categories=C_LEVELS, ordered=True)
    shown["alpha"] = pd.Categorical(shown["alpha"], categories=ALPHA_LEVELS, ordered=True)
    train = shown[shown["Set"] == "Training set"]["macro_f1"]
    near_one = float((train >= 0.99).mean()) if len(train) else 0.0
    ymin = min(0.55, float(shown["macro_f1"].min()) - 0.02)
    grid = sns.relplot(
        data=shown,
        kind="line",
        x="fraction",
        y="macro_f1",
        hue="Set",
        hue_order=["Training set", "Test set"],
        col="alpha",
        row="C",
        marker="o",
        palette=PALETTE,
        height=2.15,
        aspect=1.05,
        facet_kws={"margin_titles": True, "sharex": True, "sharey": True},
    )
    grid.set_axis_labels("Fraction of the training set", "Macro-averaged F1")
    grid.set_titles(row_template="C = {row_name:g}", col_template="α = {col_name:g}")
    grid.set(ylim=(ymin, 1.05), xlim=(0.15, 1.05))
    grid.figure.suptitle("Learning curves for the sixteen refit rounds", y=1.02, fontsize=14)
    grid.tight_layout()
    grid.figure.text(
        0.01,
        -0.02,
        f"Each panel is one of the sixteen (C, α) rounds. "
        f"{near_one:.0%} of the training points sit at or above 0.99: "
        "the descriptor still has 34,596 dimensions, so a training score near 1 is separation, not a vein measurement. "
        "The test curves are the fit picture. Validation was not read, and no panel was chosen for having the highest test score.",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    grid.savefig(OUT / "03_learning_curves.png", bbox_inches="tight", facecolor="white")
    plt.close(grid.figure)


def figure_confusion(test: pd.DataFrame, report: pd.Series) -> None:
    labels = [CLASS_LABEL[name] for name in CLASS_ORDER]
    matrix = pd.crosstab(test["True class"], test["Predicted class"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#1f4e79", as_cmap=True),
        linewidths=0.4,
        linecolor="white",
        annot_kws={"color": "black", "fontsize": 11},
        cbar_kws={"label": "Number of test images"},
        ax=ax,
    )
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Test-set confusion matrix of the pre-specified round")
    fig.text(
        0.01,
        0.01,
        f"Round {int(report['round'])} (C = {REPORT_C:g}, α = {REPORT_ALPHA:g}) was fixed before the test scores were read. "
        "Each true class contributes 75 images. This is not a validation result.",
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "04_test_confusion_matrix.png")


def figure_per_class(test: pd.DataFrame, report: pd.Series) -> None:
    labels = [CLASS_LABEL[name] for name in CLASS_ORDER]
    rows = []
    for label in labels:
        truth = test["True class"] == label
        pred = test["Predicted class"] == label
        tp = int((truth & pred).sum())
        support = int(truth.sum())
        predicted = int(pred.sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append({"Class": label, "Precision": precision, "Recall": recall, "F1": f1})
    table = pd.DataFrame(rows)
    long = table.melt(id_vars="Class", var_name="Measure", value_name="Score")
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    sns.barplot(
        data=long,
        x="Class",
        y="Score",
        hue="Measure",
        hue_order=["Precision", "Recall", "F1"],
        palette=["#1f4e79", "#5b8fa8", "#b85c38"],
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("True class on the test set")
    ax.set_ylabel("Score")
    ax.set_title("Precision, recall, and F1 by class")
    ax.legend(title=None, frameon=False, ncol=3, loc="upper right")
    ax.tick_params(axis="x", rotation=15)
    lowest = table.loc[table["F1"].idxmin()]
    highest = table.loc[table["F1"].idxmax()]
    fig.text(
        0.01,
        0.01,
        f"Round {int(report['round'])} on 375 test images. "
        f"Highest F1 is {highest['Class']} at {highest['F1']:.3f}. "
        f"Lowest F1 is {lowest['Class']} at {lowest['F1']:.3f}. "
        "A low class score is kept; it is not folded into one disease total.",
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, "05_per_class_precision_recall_f1.png")
    table.to_csv(OUT / "05_per_class_scores.csv", index=False)


def figure_cloud(test: pd.DataFrame, report: pd.Series) -> None:
    if "vein_completeness_interior" not in test.columns:
        print("06 skipped: interior completeness was not passed", flush=True)
        return
    labels = [CLASS_LABEL[name] for name in CLASS_ORDER]
    draw_order = [
        "Healthy",
        "Yellow leaf curl virus",
        "Late blight",
        "Bacterial spot",
        "Septoria leaf spot",
    ]
    plot = test.copy()
    plot["True class"] = pd.Categorical(plot["True class"], categories=labels, ordered=True)
    plot["_draw"] = plot["True class"].map({name: i for i, name in enumerate(draw_order)})
    plot = plot.sort_values("_draw", kind="mergesort")
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    sns.scatterplot(
        data=plot,
        x="vein_completeness_interior",
        y="class_score",
        hue="True class",
        hue_order=labels,
        palette=RISK_PALETTE,
        s=42,
        alpha=0.92,
        edgecolor="#1a1a1a",
        linewidth=0.35,
        ax=ax,
    )
    ax.set_xlim(left=0)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("Interior vein completeness (length / leaf major axis, not scaled)")
    ax.set_ylabel("Decision score of the leaf's own class")
    ax.set_title("Interior vein completeness and the per-class score")
    ax.legend(title="True class", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.55))
    fig.text(
        0.01,
        0.01,
        f"Each point is one test image under round {int(report['round'])} "
        f"(C = {REPORT_C:g}, α = {REPORT_ALPHA:g}). "
        "The horizontal axis is the uncalibrated interior skeleton. It is not scaled so that healthy leaves equal 0.80. "
        "The vertical axis is that leaf's own class score on a symmetric log scale, not one disease total. A softmax of these scores collapses to 0 or 1 and is not drawn. "
        "Septoria leaf spot is #77ff11. The previous generation's vein_completeness column is kept in the table and is not plotted. "
        "Validation was not used.",
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.16, 0.82, 1))
    _save(fig, "06_interior_completeness_and_class_score.png")


def main() -> None:
    _theme()
    curve, full, test, _metrics = _prepare()
    if full["round"].nunique() != 16:
        raise SystemExit(f"expected 16 rounds, found {full['round'].nunique()}")
    report = _report_row(full)
    figure_train_versus_test(full)
    figure_heatmap(full)
    figure_learning_curves(curve)
    figure_confusion(test, report)
    figure_per_class(test, report)
    figure_cloud(test, report)
    written = sorted(path.name for path in OUT.glob("*.png"))
    print("\n".join(written))


if __name__ == "__main__":
    main()
