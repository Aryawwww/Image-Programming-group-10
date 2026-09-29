import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf


# Setting fixed random seeds to improve reproducibility and ensure
# that the same dataset split and Gaussian noise can be recreated
# consistently across different runs
tf.keras.utils.set_random_seed(42)
np.random.seed(42)


# ---------------------------------------------------------
# Defining file paths and key variables
# ---------------------------------------------------------

# Path to the balanced dataset subset containing the five
# selected tomato leaf classes used for CNN training
dataset_path = (
    "/Users/djy/Desktop/Image Processing and Pattern Recognition/"
    "A2/Custom CNN/subset"
)

# Path to the final CNN model selected during hyperparameter tuning
# The selected model uses Adam with learning rate 0.001,
# dropout 0.5, batch size 32, and training data augmentation
final_model_path = (
    "/Users/djy/Desktop/Image Processing and Pattern Recognition/"
    "A2/Custom CNN/results/"
    "seed42_lr_0.001_dropout_0.5_augmentation_750_per_class_50epochs_earlystop/"
    "best_model.keras"
)

# Defining the same image size and random seed used during
# CNN training, validation and testing
image_size = (224, 224)
seed = 42


# ---------------------------------------------------------
# Loading image paths and class labels
# ---------------------------------------------------------

def load_image_paths(folder_path):
    """
    Collects the file paths and corresponding class labels for all
    images in the balanced tomato leaf dataset.

    Images are not loaded into memory at this stage. Only their file
    paths and labels are stored in a DataFrame to reduce memory usage.

    Class names are sorted alphabetically to maintain the same class
    ordering used during CNN training and evaluation.
    """

    filepaths = []
    labels = []

    # Identifying and alphabetically sorting the class folders
    label_names = sorted([
        name for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name))
    ])

    # Looping through each class folder
    for label_name in label_names:

        label_folder = Path(folder_path) / label_name

        # Looping through all image files within the current class
        for img_name in sorted(os.listdir(label_folder)):

            # Including supported image file formats only
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):

                # Recording the complete image file path
                filepaths.append(
                    str(label_folder / img_name)
                )

                # Recording the corresponding class label
                labels.append(label_name)

    # Storing image paths and labels together in a DataFrame
    df = pd.DataFrame({
        "filepath": filepaths,
        "label": labels
    })

    return df, label_names


# ---------------------------------------------------------
# Recreating the original testing subset
# ---------------------------------------------------------

def make_test_subset(df, seed):
    """
    Recreates the same 80% training, 10% validation and 10% testing
    split that was used during CNN training.

    The same random seed and stratified splitting procedure are used
    so that robustness testing is performed on the same test images
    used during the original CNN evaluation.

    Only the testing subset is returned because the trained CNN is
    not retrained during robustness testing.
    """

    # First split: assigning 80% of the dataset to training and
    # retaining the remaining 20% for validation and testing
    train_df, temp_df = train_test_split(
        df,
        train_size=0.8,
        random_state=seed,
        stratify=df["label"]
    )

    # Second split: dividing the remaining 20% equally to produce
    # 10% validation data and 10% testing data
    val_df, test_df = train_test_split(
        temp_df,
        train_size=0.5,
        random_state=seed,
        stratify=temp_df["label"]
    )

    return test_df


# ---------------------------------------------------------
# Loading and preprocessing individual test images
# ---------------------------------------------------------

def load_and_preprocess_image(filepath):
    """
    Loads an individual image from disk, converts it to RGB format,
    and resizes it to 224 x 224 pixels.

    This matches the image format and dimensions used during
    CNN training and original model evaluation.
    """

    # Opening the image and ensuring that it contains three RGB channels
    image = Image.open(filepath).convert("RGB")

    # Resizing the image to match the CNN input dimensions
    image = image.resize(
        image_size,
        resample=Image.Resampling.NEAREST
    )

    return image


# ---------------------------------------------------------
# Applying Gaussian blur
# ---------------------------------------------------------

def apply_gaussian_blur(image, radius=2):
    """
    Applies Gaussian blur to a test image to simulate image degradation
    caused by reduced image sharpness or focus.

    The transformation is applied only during robustness evaluation.
    The CNN model is not retrained using blurred images.

    The radius parameter controls the strength of Gaussian blur.
    A larger radius produces a stronger blurring effect.
    """

    # Applying Gaussian blur using the specified radius
    blurred_image = image.filter(
        ImageFilter.GaussianBlur(radius=radius)
    )

    return blurred_image


# ---------------------------------------------------------
# Applying brightness adjustment
# ---------------------------------------------------------

def apply_brightness(image, factor=0.7):
    """
    Reduces the brightness of a test image to examine whether the CNN
    remains accurate when image illumination differs from the original
    PlantVillage images.

    The transformation is applied only during robustness evaluation.
    The CNN model is not retrained using brightness-adjusted images.

    A brightness factor below 1.0 produces a darker image, while a
    factor above 1.0 produces a brighter image.
    """

    # Creating a PIL brightness adjustment object
    enhancer = ImageEnhance.Brightness(image)

    # Applying the selected brightness factor
    brightness_adjusted_image = enhancer.enhance(factor)

    return brightness_adjusted_image


# ---------------------------------------------------------
# Applying Gaussian noise
# ---------------------------------------------------------

def apply_gaussian_noise(image, mean=0.0, std=0.05):
    """
    Applies Gaussian noise to a test image to simulate random
    pixel-level variation that may occur during image acquisition
    or transmission.

    The transformation is applied only during robustness evaluation.
    The CNN model is not retrained using noisy images.

    The mean parameter controls the centre of the Gaussian noise
    distribution, while the standard deviation controls the strength
    of the added noise. A larger standard deviation produces stronger
    visible noise.
    """

    # Converting the PIL image to a NumPy array and scaling
    # pixel values from 0-255 to 0-1
    image_array = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    # Generating Gaussian noise with the same height, width
    # and number of channels as the original RGB image
    noise = np.random.normal(
        loc=mean,
        scale=std,
        size=image_array.shape
    )

    # Adding the generated Gaussian noise to the original image
    noisy_image = image_array + noise

    # Restricting all resulting pixel values to the valid
    # range between 0 and 1
    noisy_image = np.clip(
        noisy_image,
        0.0,
        1.0
    )

    # Converting the pixel values back from the 0-1 range
    # to the standard 0-255 image range
    noisy_image = (
        noisy_image * 255
    ).astype(np.uint8)

    # Converting the NumPy array back into a PIL RGB image
    noisy_image = Image.fromarray(
        noisy_image,
        mode="RGB"
    )

    return noisy_image


# ---------------------------------------------------------
# Evaluating the CNN under one testing condition
# ---------------------------------------------------------

def evaluate_condition(
    model,
    test_df,
    label_names,
    condition_name,
    transform=None
):
    """
    Evaluates the trained CNN on the complete testing subset under
    one specified image condition.

    If a transformation is provided, it is applied to each image
    before prediction. If no transformation is provided, the original
    clean test images are evaluated.

    Accuracy, precision, recall and F1-score are calculated after all
    test images have been classified. Macro-averaged metrics are used
    so that all five tomato classes contribute equally to the overall
    evaluation.
    """

    # Creating lists for storing the true and predicted class labels
    y_true = []
    y_pred = []

    # Mapping the alphabetically sorted class names to integer labels
    # corresponding to the CNN output classes
    class_to_index = {
        label: index
        for index, label in enumerate(label_names)
    }

    # Looping through every image in the testing subset
    for _, row in test_df.iterrows():

        # Loading and resizing the original test image
        image = load_and_preprocess_image(
            row["filepath"]
        )

        # Applying the selected robustness transformation when required
        # Clean test images remain unchanged when transform is None
        if transform is not None:
            image = transform(image)

        # Converting the PIL image into a NumPy array
        image_array = np.array(
            image,
            dtype=np.float32
        )

        # Scaling pixel values from 0-255 to 0-1 to match
        # the preprocessing used during CNN training
        image_array = image_array / 255.0

        # Adding a batch dimension because the CNN expects input
        # in the form: batch x height x width x channels
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Generating class probabilities using the trained CNN
        prediction = model.predict(
            image_array,
            verbose=0
        )

        # Selecting the class with the highest predicted probability
        predicted_class = int(
            np.argmax(
                prediction,
                axis=1
            )[0]
        )

        # Recording the true integer class label
        y_true.append(
            class_to_index[row["label"]]
        )

        # Recording the predicted integer class label
        y_pred.append(
            predicted_class
        )

    # Calculating class-specific and overall evaluation metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    # Printing the complete classification report for the
    # current robustness testing condition
    print()
    print("=" * 60)
    print(condition_name)
    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            zero_division=0
        )
    )

    # Returning the main overall evaluation metrics so that
    # performance can be compared across testing conditions
    return {
        "Condition": condition_name,
        "Accuracy": report["accuracy"],
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"]
    }


# ---------------------------------------------------------
# Running robustness evaluation
# ---------------------------------------------------------

if __name__ == "__main__":

    # Loading the final CNN model selected using validation performance
    # The saved model is evaluated directly and is not retrained
    print("Loading final CNN model...")

    model = tf.keras.models.load_model(
        final_model_path
    )

    # Loading all image paths and their corresponding class labels
    df, label_names = load_image_paths(
        dataset_path
    )

    # Recreating the same testing subset used during the
    # original CNN evaluation
    test_df = make_test_subset(
        df,
        seed
    )

    # Confirming the total number of test images
    print()
    print("Test images:", len(test_df))

    # Confirming that the testing subset remains balanced
    # across all five tomato leaf classes
    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    # Creating a list for storing the overall evaluation
    # metrics produced under each testing condition
    results = []


    # -----------------------------------------------------
    # Evaluating the original clean test images
    # -----------------------------------------------------

    # The clean condition provides the baseline model performance
    # against which all degraded image conditions can be compared
    results.append(
        evaluate_condition(
            model,
            test_df,
            label_names,
            condition_name="Clean"
        )
    )


    # -----------------------------------------------------
    # Evaluating images after Gaussian blur
    # -----------------------------------------------------

    # Gaussian blur is applied only to the testing images to
    # examine the effect of reduced image sharpness.
    # The trained CNN weights remain unchanged.
    results.append(
        evaluate_condition(
            model,
            test_df,
            label_names,
            condition_name="Gaussian Blur",
            transform=lambda image: apply_gaussian_blur(
                image,
                radius=2
            )
        )
    )


    # -----------------------------------------------------
    # Evaluating images after brightness reduction
    # -----------------------------------------------------

    # Image brightness is reduced to 70% of the original level
    # to examine CNN performance under reduced illumination.
    # The trained CNN weights remain unchanged.
    results.append(
        evaluate_condition(
            model,
            test_df,
            label_names,
            condition_name="Reduced Brightness",
            transform=lambda image: apply_brightness(
                image,
                factor=0.7
            )
        )
    )


    # -----------------------------------------------------
    # Evaluating images after Gaussian noise
    # -----------------------------------------------------

    # Gaussian noise is added only to the testing images to
    # examine CNN performance under random pixel-level degradation.
    # The trained CNN weights remain unchanged.
    results.append(
        evaluate_condition(
            model,
            test_df,
            label_names,
            condition_name="Gaussian Noise",
            transform=lambda image: apply_gaussian_noise(
                image,
                mean=0.0,
                std=0.05
            )
        )
    )


    # -----------------------------------------------------
    # Summarising robustness results
    # -----------------------------------------------------

    # Combining the evaluation results from all four conditions
    # into a single DataFrame for direct comparison
    results_df = pd.DataFrame(
        results
    )

    # Printing the overall robustness comparison table
    print()
    print("=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # Saving the robustness results as a CSV file so that
    # they can later be used for analysis, figures and reporting
    results_df.to_csv(
        "robustness_results.csv",
        index=False
    )

    print()
    print("Saved: robustness_results.csv")