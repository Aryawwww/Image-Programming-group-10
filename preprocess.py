"""Load the directory dataset prepared by the Custom CNN pipeline."""

from pathlib import Path


SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_IMAGES_PER_CLASS = {"train": 600, "validation": 75, "test": 75}


def inspect_shared_dataset(dataset_dir):
    """Validate the shared five-class CNN splits and return their file counts."""
    dataset_dir = Path(dataset_dir)
    summary = {}
    class_names = None

    for split, expected_count in EXPECTED_IMAGES_PER_CLASS.items():
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Shared CNN split directory not found: {split_dir}")

        split_classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        if class_names is None:
            class_names = split_classes
            if len(class_names) != 5:
                raise ValueError(
                    f"Expected 5 class folders in {split_dir}, found {len(class_names)}."
                )
        elif split_classes != class_names:
            raise ValueError(
                f"Class folders in {split_dir} do not match the training split."
            )

        summary[split] = {}
        for class_name in class_names:
            class_dir = split_dir / class_name
            count = sum(
                path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                for path in class_dir.iterdir()
            )
            if count != expected_count:
                raise ValueError(
                    f"Expected {expected_count} {split} images for {class_name}, "
                    f"found {count} in {class_dir}."
                )
            summary[split][class_name] = count

    return class_names, summary


def load_shared_datasets(dataset_dir, image_size, batch_size, seed):
    """Load the exact train/validation/test folders prepared for the Custom CNN."""
    import tensorflow as tf

    dataset_dir = Path(dataset_dir)
    class_names, summary = inspect_shared_dataset(dataset_dir)
    common_arguments = {
        "image_size": image_size,
        "color_mode": "rgb",
        "batch_size": batch_size,
        "label_mode": "int",
        "class_names": class_names,
    }
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "train", shuffle=True, seed=seed, **common_arguments
    )
    evaluation_arguments = {**common_arguments, "shuffle": False}
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "validation", **evaluation_arguments
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "test", **evaluation_arguments
    )
    return train_dataset, validation_dataset, test_dataset, class_names, summary
