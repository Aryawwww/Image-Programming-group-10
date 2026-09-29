import os
import numpy as np
import random
import pandas as pd
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_model, train_model
from evaluate import plot_history, calculate_metrics, plot_confusion_matrix

tf.keras.utils.set_random_seed(42)

def is_valid_image(path):
    """
    Checks if an image is valid and readable using PIL.
    Returns True if valid, False if corrupted or unreadable.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # Only checks integrity
        return True
    except Exception:
        return False

def downsample_images(input_dir, output_dir, output_class_names, target_count, seed):
    """
    Downsamples or copies only valid images from input_dir to output_dir.
    Ensures no class has more than target_count images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Looping through each class 
    for label_name in output_class_names:
        label_folder = Path(input_dir) / label_name

        if not os.path.isdir(label_folder):
            continue

        output_label_folder = Path(output_dir) / label_name
        os.makedirs(output_label_folder, exist_ok=True)

        # Skip if already has target number of valid images
        existing_output_files = [
            f for f in os.listdir(output_label_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(existing_output_files) == target_count:
            print(f"[{label_name}] Already has {target_count} images — skipping.")
            continue

        # Collect valid image files only
        all_files = [
            f for f in os.listdir(label_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        valid_files = []
        for f in all_files:
            full_path = os.path.join(label_folder, f)
            if is_valid_image(full_path):
                valid_files.append(f)
            else:
                print(f"[{label_name}] Skipped corrupted image: {f}")

        if len(valid_files) == 0:
            print(f"[{label_name}] No valid images found — skipping.")
            continue

        # Copying valid images or downsampling 
        if len(valid_files) >= target_count:
            print(f"[{label_name}] Number of invalid and valid images: {len(all_files)}.")
            print(f"[{label_name}] Downsampling from {len(valid_files)} to {target_count} valid images.")
            random.seed(seed) 
            selected_files = random.sample(valid_files, target_count) 
            for f in selected_files:
                src_path = os.path.join(label_folder, f)
                dst_path = os.path.join(output_label_folder, f)
                shutil.copy(src_path, dst_path)
        else: 
            # Print error message if the number of valid files is less than the target count
            print(f"[{label_name}] Skipped: Found only {len(valid_files)} valid images; needs at least {target_count}")
            break

def load_image_paths(folder_path):
    """
    Collects image file paths and class labels without loading
    all image pixel data into memory.
    """

    filepaths = []
    labels = []

    label_names = sorted([
        name for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name))
    ])

    for label_name in label_names:
        label_folder = Path(folder_path) / label_name

        for img_name in sorted(os.listdir(label_folder)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(label_folder, img_name)
                filepaths.append(img_path)
                labels.append(label_name)

    df = pd.DataFrame({
        "filepath": filepaths,
        "label": labels
    })

    print()
    print(f"Total image paths: {len(df)}")
    print(df["label"].value_counts().sort_index())

    return df, label_names


def make_subsets(df, seed):
    """
    Splits image paths into 80% training, 10% validation,
    and 10% testing subsets while preserving class balance.
    """

    train_df, temp_df = train_test_split(
        df,
        train_size=0.8,
        random_state=seed,
        stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        train_size=0.5,
        random_state=seed,
        stratify=temp_df["label"]
    )

    print()
    print("Training images:", len(train_df))
    print("Validation images:", len(val_df))
    print("Testing images:", len(test_df))

    return train_df, val_df, test_df


def create_generators(train_df, val_df, test_df, image_size, batch_size, seed):
    """
    Creates generators that load images from disk batch by batch
    instead of loading the entire dataset into memory.
    """

    # Data augmentation for training images only
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Validation and test images are only rescaled
    eval_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=image_size,
        color_mode="rgb",
        class_mode="sparse",
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    val_generator = eval_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col="label",
        target_size=image_size,
        color_mode="rgb",
        class_mode="sparse",
        batch_size=batch_size,
        shuffle=False
    )

    test_generator = eval_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col="label",
        target_size=image_size,
        color_mode="rgb",
        class_mode="sparse",
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, val_generator, test_generator

if __name__ == "__main__":

    # Defining paths  
    dataset_path = "/Users/djy/Desktop/Image Processing and Pattern Recognition/A2/PlantVillage-Dataset-master/raw/color"
    output_path = "/Users/djy/Desktop/Image Processing and Pattern Recognition/A2/Custom CNN/subset"

    # Provisional five classes for initial pipeline testing
    subset_class_names = [
        "Tomato___healthy",
        "Tomato___Late_blight",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Bacterial_spot"
    ]

    # Defining key variables
    image_size = (224, 224)
    target_count = 750 # Change to 750 images later 
    seed = 42
    batch_size = 32
    num_epochs = 50 # Change to 50 later
    best_model_file = "best_model.keras"

    # Downsampling each class to equal the target_count and saving into subset_path
    downsample_images(dataset_path, output_path, subset_class_names, target_count, seed)

    # Collecting image paths without loading all images into memory
    df, label_names = load_image_paths(output_path)

    # Creating training, validation and testing subsets
    train_df, val_df, test_df = make_subsets(df, seed)

    # Creating generators that load images batch by batch
    train_generator, val_generator, test_generator = create_generators(
        train_df, val_df, test_df, image_size, batch_size, seed
    )

    # Integer labels for confusion matrix
    int_labels = [i for i in range(len(label_names))]

    # True test labels in the same order as test_generator
    y_test = test_generator.classes

    # Build and train the model
    model = build_model(image_size)
    history = train_model(model, train_generator, val_generator, num_epochs, best_model_file)

    # Saving history 
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("training_history.csv", index=False)

    # Loading history and best model 
    history_df = pd.read_csv("training_history.csv")
    model = tf.keras.models.load_model(best_model_file)
    
    # Evaluating metrics and plotting confusion matrix
    y_pred = calculate_metrics(test_generator, y_test, model, label_names)
    plot_confusion_matrix(y_test, y_pred, label_names, int_labels)

    # Plotting accuracy and validation accuracy and loss curves 
    plot_history(history_df)






