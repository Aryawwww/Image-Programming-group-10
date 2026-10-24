"""Fine-tune the saved MobileNetV3-Small frozen-backbone model."""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from preprocess import SEED, load_shared_datasets


BLOCK_PATTERN = re.compile(r"^(expanded_conv(?:_\d+)?)(?:_|$)")


def parse_args():
	project_dir = Path(__file__).resolve().parent
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--initial-model",
		type=Path,
		default=project_dir / "outputs" / "frozen" / "best_frozen_model.keras",
		help="Frozen-backbone checkpoint to continue training from.",
	)
	parser.add_argument(
		"--dataset-dir",
		type=Path,
		default=project_dir / "dataset",
		help="Shared dataset with train, validation, and test directories.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=project_dir / "outputs" / "fine_tuned",
		help="Directory for separate fine-tuning trial results.",
	)
	parser.add_argument(
		"--run-name",
		default=None,
		help="Trial folder name; defaults to a timestamp.",
	)
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--learning-rate", type=float, default=1e-5)
	parser.add_argument(
		"--unfreeze-blocks",
		type=int,
		default=6,
		help="Number of final MobileNet inverted-residual blocks to unfreeze.",
	)
	parser.add_argument("--patience", type=int, default=10)
	return parser.parse_args()


def unfreeze_final_blocks(model, block_count):
	"""Unfreeze the final MobileNet blocks and keep BatchNorm layers frozen."""
	import tensorflow as tf

	backbones = [
		layer
		for layer in model.layers
		if isinstance(layer, tf.keras.Model)
		and any(BLOCK_PATTERN.match(child.name) for child in layer.layers)
	]
	if len(backbones) != 1:
		raise ValueError(
			"Could not uniquely locate the nested MobileNetV3 backbone in the checkpoint."
		)

	backbone = backbones[0]
	block_names = []
	for layer in backbone.layers:
		match = BLOCK_PATTERN.match(layer.name)
		if match and match.group(1) not in block_names:
			block_names.append(match.group(1))
	if block_count < 1 or block_count > len(block_names):
		raise ValueError(
			f"unfreeze-blocks must be between 1 and {len(block_names)}; got {block_count}."
		)

	selected_blocks = block_names[-block_count:]
	selected_block_names = set(selected_blocks)
	backbone.trainable = True
	for layer in backbone.layers:
		match = BLOCK_PATTERN.match(layer.name)
		layer.trainable = bool(
			match
			and match.group(1) in selected_block_names
			and not isinstance(layer, tf.keras.layers.BatchNormalization)
		)
	return backbone.name, selected_blocks


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
	axes[0].set_ylim(0.7, 1.0)
	axes[0].legend()
	axes[1].plot(epochs, history_df["loss"], label="Training")
	axes[1].plot(epochs, history_df["val_loss"], label="Validation")
	axes[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
	axes[1].set_ylim(0, 0.8)
	axes[1].legend()
	figure.tight_layout()
	figure.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(figure)


def main():
	args = parse_args()
	if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
		raise ValueError("epochs, batch-size, and learning-rate must be positive.")
	if args.patience < 0:
		raise ValueError("patience cannot be negative.")
	if not args.initial_model.is_file():
		raise FileNotFoundError(f"Frozen model checkpoint not found: {args.initial_model}")

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

	model = tf.keras.models.load_model(args.initial_model, compile=False)
	if model.output_shape[-1] != len(class_names):
		raise ValueError(
			f"Checkpoint has {model.output_shape[-1]} outputs, but the dataset has "
			f"{len(class_names)} classes."
		)
	backbone_name, trainable_blocks = unfreeze_final_blocks(
		model, args.unfreeze_blocks
	)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)

	run_name = args.run_name or datetime.now().strftime("finetune_%Y%m%d_%H%M%S")
	run_dir = args.output_dir / run_name
	run_dir.mkdir(parents=True, exist_ok=False)
	checkpoint_path = run_dir / "best_finetuned_model.keras"
	callbacks = [
		tf.keras.callbacks.EarlyStopping(
			monitor="val_loss",
			patience=args.patience,
			restore_best_weights=True,
		),
		tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_path,
			monitor="val_loss",
			save_best_only=True,
		),
	]

	started_at = time.perf_counter()
	history = model.fit(
		train_dataset,
		validation_data=validation_dataset,
		epochs=args.epochs,
		callbacks=callbacks,
	)
	training_seconds = time.perf_counter() - started_at

	history_df = pd.DataFrame(history.history)
	history_df.index += 1
	history_df.index.name = "epoch"
	history_df.to_csv(run_dir / "training_history.csv")
	save_history_plot(history_df, run_dir / "training_history.png")
	best_loss_row = history_df.loc[history_df["val_loss"].idxmin()]
	best_accuracy_row = history_df.loc[history_df["val_accuracy"].idxmax()]
	metadata = {
		"architecture": "MobileNetV3Small",
		"source_checkpoint": str(args.initial_model.resolve()),
		"backbone": backbone_name,
		"trainable_blocks": trainable_blocks,
		"unfreeze_blocks": args.unfreeze_blocks,
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
		"best_validation_loss_epoch": int(best_loss_row.name),
		"best_validation_loss": float(best_loss_row["val_loss"]),
		"validation_accuracy_at_best_loss": float(best_loss_row["val_accuracy"]),
		"best_validation_accuracy_epoch": int(best_accuracy_row.name),
		"best_validation_accuracy": float(best_accuracy_row["val_accuracy"]),
		"training_seconds": training_seconds,
		"checkpoint": checkpoint_path.name,
	}
	(run_dir / "training_config.json").write_text(
		json.dumps(metadata, indent=2), encoding="utf-8"
	)
	print(f"Fine-tuned checkpoint: {checkpoint_path}")
	print(f"Trial results: {run_dir}")
	print(f"Trainable MobileNet blocks: {', '.join(trainable_blocks)}")
	print(f"Training time: {training_seconds:.2f} seconds")


if __name__ == "__main__":
	main()
