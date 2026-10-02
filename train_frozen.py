"""Train the MobileNetV3-Small frozen-backbone baseline."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from model import build_frozen_model, train_frozen_model
from preprocess import SEED, load_shared_datasets


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_dir / "dataset",
        help="Shared CNN dataset containing train/validation/test folders.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_dir / "outputs" / "frozen"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def save_history_plot(history_df, output_path):
    """Save training and validation accuracy/loss curves without opening a GUI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history_df) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history_df["accuracy"], label="Training")
    axes[0].plot(epochs, history_df["val_accuracy"], label="Validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[1].plot(epochs, history_df["loss"], label="Training")
    axes[1].plot(epochs, history_df["val_loss"], label="Validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        raise ValueError("epochs and batch-size must be positive; patience cannot be negative.")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    image_size = (224, 224)
    (
        train_dataset,
        validation_dataset,
        _,
        class_names,
        dataset_summary,
    ) = load_shared_datasets(args.dataset_dir, image_size, args.batch_size, SEED)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_frozen_model.keras"
    model = build_frozen_model(image_size, len(class_names))
    started_at = time.perf_counter()
    history = train_frozen_model(
        model,
        train_dataset,
        validation_dataset,
        args.epochs,
        args.learning_rate,
        args.patience,
        checkpoint_path,
    )
    training_seconds = time.perf_counter() - started_at

    history_df = pd.DataFrame(history.history)
    history_df.index += 1
    history_df.index.name = "epoch"
    history_df.to_csv(args.output_dir / "training_history.csv")
    save_history_plot(history_df, args.output_dir / "training_history.png")
    metadata = {
        "architecture": "MobileNetV3Small",
        "backbone_trainable": False,
        "weights": "imagenet",
        "image_size": list(image_size),
        "batch_size": args.batch_size,
        "maximum_epochs": args.epochs,
        "epochs_completed": len(history_df),
        "learning_rate": args.learning_rate,
        "early_stopping_patience": args.patience,
        "seed": SEED,
        "class_indices": {name: index for index, name in enumerate(class_names)},
        "shared_dataset_directory": str(args.dataset_dir.resolve()),
        "dataset_counts": dataset_summary,
        "training_seconds": training_seconds,
        "checkpoint": checkpoint_path.name,
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training time: {training_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
