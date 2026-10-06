"""Seaborn figures for the frozen S4 winner on PlantVillage color.

The panels compare HOG alone with HOG plus the frozen HSV weight 0.45.
Weight 0 is an ablation. It did not choose the round.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
OUT = HERE / "s4-winner-results"
SCORES = Path(r"E:\IT\Projects\42177 Image ProcessingAndPatternRecognition\plantvillage_s4_full")

HSV_OFF = "HSV off"
HSV_ON = "HSV on (0.45)"
CONDITION_ORDER = [HSV_OFF, HSV_ON]
CONDITION_COLOR = {HSV_OFF: "#8d99ae", HSV_ON: "#1d4e89"}
NOTE = (
    "*PlantVillage color, 54,305 images, not the course split of 3,750. "
    "Each crop with healthy leaves and at least one disease is refit on a stratified 20% holdout (seed 0). "
    "Minimum AUC is the lowest Healthy-versus-one-disease score in that crop. "
    "Knobs are the frozen S4 winner: 12 orientations, 16-pixel cells, 2×2 L2-Hys, C=1. "
    "Report round 10 is a different model (9 orientations, 8-pixel cells, 34,596 HOG numbers, HSV weight 0.25) and is not shown. "
    "HSV off was not an S4 candidate and did not choose the winner."
)
DASH = " The dashed line is 0.8."
TOMATO_EXTRA = (
    " Colored ticks are the four diseases in the course tomato set. "
    "Septoria leaf spot is #77ff11. Gray ticks are the other tomato diseases. "
    "This tomato model has ten classes, not the course five."
)
JOINT_EXTRA = (
    " This panel is one 38-class linear SVM. "
    "The axis is a change in holdout F1 for naming the class, not the minimum AUC used to freeze the winner. "
    "Macro-F1 is 0.601 with HSV off and 0.617 with HSV at 0.45."
)
COURSE_TICK = {
    "Bacterial spot": "#d9480f",
    "Late blight": "#6f2dbd",
    "Septoria leaf spot": "#77ff11",
    "Yellow leaf curl virus": "#0e7490",
}
CROP_LABEL = {
    "Apple": "Apple",
    "Cherry_(including_sour)": "Cherry",
    "Corn_(maize)": "Corn",
    "Grape": "Grape",
    "Peach": "Peach",
    "Pepper,_bell": "Pepper",
    "Potato": "Potato",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
}
DISEASE_LABEL = {
    "Bacterial_spot": "Bacterial spot",
    "Early_blight": "Early blight",
    "Late_blight": "Late blight",
    "Leaf_Mold": "Leaf mold",
    "Septoria_leaf_spot": "Septoria leaf spot",
    "Spider_mites Two-spotted_spider_mite": "Two-spotted spider mite",
    "Target_Spot": "Target spot",
    "Tomato_Yellow_Leaf_Curl_Virus": "Yellow leaf curl virus",
    "Tomato_mosaic_virus": "Mosaic virus",
}
CLASS_LABEL = {
    "Apple___Apple_scab": "Apple — scab",
    "Apple___Black_rot": "Apple — black rot",
    "Apple___Cedar_apple_rust": "Apple — cedar rust",
    "Apple___healthy": "Apple — healthy",
    "Blueberry___healthy": "Blueberry — healthy",
    "Cherry_(including_sour)___Powdery_mildew": "Cherry — powdery mildew",
    "Cherry_(including_sour)___healthy": "Cherry — healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn — gray leaf spot",
    "Corn_(maize)___Common_rust_": "Corn — common rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn — northern leaf blight",
    "Corn_(maize)___healthy": "Corn — healthy",
    "Grape___Black_rot": "Grape — black rot",
    "Grape___Esca_(Black_Measles)": "Grape — esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape — leaf blight",
    "Grape___healthy": "Grape — healthy",
    "Orange___Haunglongbing_(Citrus_greening)": "Orange — citrus greening",
    "Peach___Bacterial_spot": "Peach — bacterial spot",
    "Peach___healthy": "Peach — healthy",
    "Pepper,_bell___Bacterial_spot": "Pepper — bacterial spot",
    "Pepper,_bell___healthy": "Pepper — healthy",
    "Potato___Early_blight": "Potato — early blight",
    "Potato___Late_blight": "Potato — late blight",
    "Potato___healthy": "Potato — healthy",
    "Raspberry___healthy": "Raspberry — healthy",
    "Soybean___healthy": "Soybean — healthy",
    "Squash___Powdery_mildew": "Squash — powdery mildew",
    "Strawberry___Leaf_scorch": "Strawberry — leaf scorch",
    "Strawberry___healthy": "Strawberry — healthy",
    "Tomato___Bacterial_spot": "Tomato — bacterial spot",
    "Tomato___Early_blight": "Tomato — early blight",
    "Tomato___Late_blight": "Tomato — late blight",
    "Tomato___Leaf_Mold": "Tomato — leaf mold",
    "Tomato___Septoria_leaf_spot": "Tomato — Septoria leaf spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato — spider mite",
    "Tomato___Target_Spot": "Tomato — target spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato — yellow leaf curl",
    "Tomato___Tomato_mosaic_virus": "Tomato — mosaic virus",
    "Tomato___healthy": "Tomato — healthy",
}


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
            "axes.titlepad": 8,
        }
    )


def _caption(figure: plt.Figure, text: str, width: int, top: float = 1.0) -> None:
    wrapped = "\n".join(textwrap.wrap(text, width=width, break_on_hyphens=False))
    lines = wrapped.count("\n") + 1
    bottom = 0.012 + 0.034 * lines
    figure.tight_layout(rect=(0.0, bottom, 1.0, top))
    figure.text(0.0, 0.004, wrapped, fontsize=8, ha="left", va="bottom")


def _save(figure: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _pretty_disease(name: str) -> str:
    if name in DISEASE_LABEL:
        return DISEASE_LABEL[name]
    return name.replace("_", " ")


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    on_crop = pd.read_csv(SCORES / "crop_summary.csv")
    off_crop = pd.read_csv(SCORES / "hsv_off_crop_summary.csv")
    on_dis = pd.read_csv(SCORES / "disease_auc.csv")
    off_dis = pd.read_csv(SCORES / "hsv_off_disease_auc.csv")
    on_joint = pd.read_csv(SCORES / "joint_per_class.csv")
    off_joint = pd.read_csv(SCORES / "hsv_off_joint_per_class.csv")
    joint_on = json.loads((SCORES / "synthesis.json").read_text(encoding="utf-8"))["joint_38"]
    joint_off = json.loads((SCORES / "hsv_off_joint.json").read_text(encoding="utf-8"))

    crops = on_crop.merge(off_crop, on="crop", suffixes=("_on", "_off"))
    crops["Crop"] = crops["crop"].map(CROP_LABEL)
    crops["delta_min"] = crops["min_auc_on"] - crops["min_auc_off"]
    crops["delta_mean"] = crops["mean_auc_on"] - crops["mean_auc_off"]
    if crops["Crop"].isna().any():
        raise RuntimeError("a crop has no display name")

    diseases = on_dis.merge(off_dis, on=["crop", "disease"], suffixes=("_on", "_off"))
    diseases["Disease"] = diseases["disease"].map(_pretty_disease)
    diseases["delta_auc"] = diseases["auc_on"] - diseases["auc_off"]

    joint = on_joint.merge(off_joint, on="class_name", suffixes=("_on", "_off"))
    joint["delta_f1"] = joint["f1_on"] - joint["f1_off"]
    missing = sorted(set(joint["class_name"]) - set(CLASS_LABEL))
    if missing:
        raise RuntimeError(f"class names without a display label: {missing}")
    joint["Class"] = joint["class_name"].map(CLASS_LABEL)
    summary = {
        "joint_macro_on": float(joint_on["macro_f1"]),
        "joint_macro_off": float(joint_off["macro_f1"]),
        "joint_macro_delta": float(joint_on["macro_f1"]) - float(joint_off["macro_f1"]),
        "mean_delta_min": float(crops["delta_min"].mean()),
        "mean_delta_mean": float(crops["delta_mean"].mean()),
    }
    return crops, diseases, joint, summary


def _condition_frame(crops: pd.DataFrame) -> pd.DataFrame:
    order = crops.sort_values("min_auc_on", kind="mergesort")["Crop"]
    rows = []
    for record in crops.itertuples(index=False):
        rows.append({"Crop": record.Crop, "Condition": HSV_OFF, "Minimum AUC": record.min_auc_off})
        rows.append({"Crop": record.Crop, "Condition": HSV_ON, "Minimum AUC": record.min_auc_on})
    frame = pd.DataFrame(rows)
    frame["Crop"] = pd.Categorical(frame["Crop"], list(order), ordered=True)
    frame["Condition"] = pd.Categorical(frame["Condition"], CONDITION_ORDER, ordered=True)
    return frame.sort_values(["Crop", "Condition"], kind="mergesort")


def figure_crop_levels(crops: pd.DataFrame) -> None:
    frame = _condition_frame(crops)
    lower = min(0.75, float(frame["Minimum AUC"].min()) - 0.02)
    figure, ax = plt.subplots(figsize=(8.4, 5.2))
    sns.barplot(
        data=frame,
        y="Crop",
        x="Minimum AUC",
        hue="Condition",
        hue_order=CONDITION_ORDER,
        palette=CONDITION_COLOR,
        errorbar=None,
        ax=ax,
    )
    ax.axvline(0.8, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlim(lower, 1.0)
    ax.set_xlabel("Minimum AUC in each crop*")
    ax.set_ylabel("")
    ax.set_title("Minimum AUC with HSV on (weight 0.45) and off")
    ax.legend(title="", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    _caption(figure, NOTE + DASH, width=108)
    _save(figure, "01_crop_min_auc.png")


def figure_lift(crops: pd.DataFrame) -> None:
    order = list(crops.sort_values("min_auc_on", kind="mergesort")["Crop"])
    long = crops.melt(
        id_vars=["Crop"],
        value_vars=["delta_min", "delta_mean"],
        var_name="Metric",
        value_name="Change in AUC",
    )
    long["Metric"] = long["Metric"].map(
        {"delta_min": "Hardest disease", "delta_mean": "Mean of the diseases"}
    )
    long["Crop"] = pd.Categorical(long["Crop"], order, ordered=True)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 5.2), sharey=True)
    for ax, metric in zip(axes, ["Hardest disease", "Mean of the diseases"]):
        part = long[long["Metric"] == metric]
        sns.barplot(
            data=part,
            y="Crop",
            x="Change in AUC",
            color="#1d4e89",
            errorbar=None,
            ax=ax,
        )
        ax.axvline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(metric)
        ax.set_xlabel("AUC on (0.45) minus AUC off*")
        ax.set_ylabel("")
    figure.suptitle("AUC delta with HSV on (weight 0.45) and off", fontsize=12)
    _caption(figure, NOTE, width=125, top=0.92)
    _save(figure, "02_hsv_lift_by_crop.png")


def figure_tomato(diseases: pd.DataFrame) -> None:
    tomato = diseases[diseases["crop"] == "Tomato"].copy()
    order = list(tomato.sort_values("auc_on", kind="mergesort")["Disease"])
    rows = []
    for record in tomato.itertuples(index=False):
        rows.append({"Disease": record.Disease, "Condition": HSV_OFF, "AUC": record.auc_off})
        rows.append({"Disease": record.Disease, "Condition": HSV_ON, "AUC": record.auc_on})
    frame = pd.DataFrame(rows)
    frame["Disease"] = pd.Categorical(frame["Disease"], order, ordered=True)
    lower = min(0.7, float(frame["AUC"].min()) - 0.02)
    figure, ax = plt.subplots(figsize=(8.4, 5.4))
    sns.barplot(
        data=frame,
        y="Disease",
        x="AUC",
        hue="Condition",
        hue_order=CONDITION_ORDER,
        palette=CONDITION_COLOR,
        errorbar=None,
        ax=ax,
    )
    ax.axvline(0.8, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlim(lower, 1.0)
    ax.set_xlabel("AUC, healthy versus this disease*")
    ax.set_ylabel("")
    ax.set_title("Tomato AUC with HSV on (weight 0.45) and off")
    ax.legend(title="", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    for label in ax.get_yticklabels():
        color = COURSE_TICK.get(label.get_text())
        if color is not None:
            label.set_color(color)
    _caption(figure, NOTE + DASH + TOMATO_EXTRA, width=108)
    _save(figure, "03_tomato_disease_auc.png")


def figure_joint(joint: pd.DataFrame, summary: dict) -> None:
    order = list(joint.sort_values("delta_f1", kind="mergesort")["Class"])
    frame = joint.copy()
    frame["Class"] = pd.Categorical(frame["Class"], order, ordered=True)
    higher = "HSV on (0.45) higher"
    lower = "HSV on (0.45) lower"
    frame["Direction"] = np.where(frame["delta_f1"] >= 0, higher, lower)
    figure, ax = plt.subplots(figsize=(8.8, 9.6))
    sns.barplot(
        data=frame,
        y="Class",
        x="delta_f1",
        hue="Direction",
        hue_order=[higher, lower],
        palette={higher: "#1d4e89", lower: "#9a3412"},
        dodge=False,
        errorbar=None,
        ax=ax,
    )
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("F1 on (0.45) minus F1 off*")
    ax.set_ylabel("")
    ax.set_title("F1 delta with HSV on (weight 0.45) and off")
    ax.legend(title="", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    _caption(figure, NOTE + JOINT_EXTRA, width=115)
    _save(figure, "04_joint_f1_change.png")


def main() -> None:
    _theme()
    crops, diseases, joint, summary = _load()
    figure_crop_levels(crops)
    figure_lift(crops)
    figure_tomato(diseases)
    figure_joint(joint, summary)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"S4 winner figures in {OUT}", flush=True)
    print(
        f"mean delta min AUC {summary['mean_delta_min']:+.3f} "
        f"joint macro {summary['joint_macro_off']:.3f} -> {summary['joint_macro_on']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
