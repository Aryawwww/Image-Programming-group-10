import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def is_valid_image(path):
    """Return True when PIL can open and verify an image file."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def downsample_images(input_dir, output_dir, output_class_names, target_count, seed):
    """
    Copy the same balanced five-class subset used by the Custom CNN pipeline.

    The file discovery and seeded sampling procedure intentionally match the
    Custom CNN branch. Reuse the generated manifests after the first run.
    """
    os.makedirs(output_dir, exist_ok=True)

    for label_name in output_class_names:
        label_folder = Path(input_dir) / label_name

        if not label_folder.is_dir():
            raise FileNotFoundError(f"Class directory not found: {label_folder}")

        output_label_folder = Path(output_dir) / label_name
        os.makedirs(output_label_folder, exist_ok=True)

        existing_output_files = [
            filename
            for filename in os.listdir(output_label_folder)
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(existing_output_files) == target_count:
            print(f"[{label_name}] Already has {target_count} images - skipping.")
            continue
        if existing_output_files:
            raise RuntimeError(
                f"[{label_name}] Output contains {len(existing_output_files)} images; "
                f"expected either 0 or {target_count}."
            )

        all_files = [
            filename
            for filename in os.listdir(label_folder)
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        valid_files = []
        for filename in all_files:
            full_path = label_folder / filename
            if is_valid_image(full_path):
                valid_files.append(filename)
            else:
                print(f"[{label_name}] Skipped corrupted image: {filename}")

        if len(valid_files) < target_count:
            raise ValueError(
                f"[{label_name}] Found {len(valid_files)} valid images; "
                f"at least {target_count} are required."
            )

        print(f"[{label_name}] Downsampling {len(valid_files)} valid images to {target_count}.")
        random.seed(seed)
        selected_files = random.sample(valid_files, target_count)
        for filename in selected_files:
            shutil.copy(label_folder / filename, output_label_folder / filename)


def load_image_paths(folder_path):
    """Collect sorted image paths and class labels without loading pixels."""
    filepaths = []
    labels = []

    label_names = sorted(
        name
        for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name))
    )

    for label_name in label_names:
        label_folder = Path(folder_path) / label_name
        for image_name in sorted(os.listdir(label_folder)):
            if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(str(label_folder / image_name))
                labels.append(label_name)

    dataframe = pd.DataFrame({"filepath": filepaths, "label": labels})

    print(f"\nTotal image paths: {len(dataframe)}")
    print(dataframe["label"].value_counts().sort_index())
    return dataframe, label_names


def make_subsets(dataframe, seed):
    """Create the Custom CNN branch's stratified 80/10/10 split."""
    train_df, temporary_df = train_test_split(
        dataframe,
        train_size=0.8,
        random_state=seed,
        stratify=dataframe["label"],
    )
    validation_df, test_df = train_test_split(
        temporary_df,
        train_size=0.5,
        random_state=seed,
        stratify=temporary_df["label"],
    )

    print(f"\nTraining images: {len(train_df)}")
    print(f"Validation images: {len(validation_df)}")
    print(f"Testing images: {len(test_df)}")
    return train_df, validation_df, test_df


def save_split_manifests(train_df, validation_df, test_df, subset_dir, output_dir):
    """Save portable relative paths so every model can use the same images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, dataframe in (
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ):
        manifest = dataframe.copy()
        manifest["filepath"] = manifest["filepath"].map(
            lambda path: Path(path).relative_to(subset_dir).as_posix()
        )
        manifest.insert(0, "split", split_name)
        manifest.to_csv(output_dir / f"{split_name}_split.csv", index=False)


def create_generators(train_df, validation_df, test_df, image_size, batch_size, seed):
    """
    Create MobileNetV3-Small image generators.

    Pixel values remain in the 0-255 range. Create MobileNetV3-Small with its
    default include_preprocessing=True setting.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    evaluation_datagen = ImageDataGenerator()

    common_arguments = {
        "x_col": "filepath",
        "y_col": "label",
        "target_size": image_size,
        "color_mode": "rgb",
        "class_mode": "sparse",
        "batch_size": batch_size,
    }

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        shuffle=True,
        seed=seed,
        **common_arguments,
    )
    validation_generator = evaluation_datagen.flow_from_dataframe(
        dataframe=validation_df,
        shuffle=False,
        **common_arguments,
    )
    test_generator = evaluation_datagen.flow_from_dataframe(
        dataframe=test_df,
        shuffle=False,
        **common_arguments,
    )
    return train_generator, validation_generator, test_generator


if __name__ == "__main__":
    dataset_path = Path(r"E:\UTS\42177 Image Processing\A2\PlantVillage-Tomato-5")
    subset_path = Path(r"E:\UTS\42177 Image Processing\A2\PlantVillage-Tomato-5-subset-750")
    split_output_path = Path(__file__).resolve().parent / "data_splits"

    subset_class_names = [
        "Tomato___healthy",
        "Tomato___Late_blight",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Bacterial_spot",
    ]

    target_count = 750

    downsample_images(dataset_path, subset_path, subset_class_names, target_count, SEED)
    dataframe, _ = load_image_paths(subset_path)
    train_df, validation_df, test_df = make_subsets(dataframe, SEED)
    save_split_manifests(
        train_df,
        validation_df,
        test_df,
        subset_path,
        split_output_path,
    )
    print(f"\nSaved split manifests to: {split_output_path}")
