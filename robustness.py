import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Setting random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Defining paths and parameters
model_path = "results/dropout0.5_lr0.001_batch64/best_model.keras"
test_path = "subset/test"
image_size = (224, 224)
batch_size = 64

# Creating folder for robustness results
output_dir = Path("results/robustness")
output_dir.mkdir(parents=True, exist_ok=True)

# Loading original test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False
)

# Getting class names from the test dataset
class_names = test_dataset.class_names

print("\nClass order:")
for i, name in enumerate(class_names):
    print(i, name)

# Loading the final tuned CNN model
model = tf.keras.models.load_model(model_path)

print("\nFinal tuned CNN loaded successfully.")
print("Model:", model_path)
print("Robustness results will be saved to:", output_dir)

# Adding Gaussian noise to images
def add_gaussian_noise(images, sigma):
    noise = tf.random.normal(
        shape=tf.shape(images),
        mean=0.0,
        stddev=sigma * 255.0
    )
    noisy_images = images + noise
    return tf.clip_by_value(noisy_images, 0.0, 255.0)


# Applying Gaussian blur to images
def apply_gaussian_blur(images, kernel_size):
    sigma = 1.0

    x = tf.range(
        -(kernel_size // 2),
        kernel_size // 2 + 1,
        dtype=tf.float32
    )

    gaussian = tf.exp(-(x ** 2) / (2.0 * sigma ** 2))
    gaussian = gaussian / tf.reduce_sum(gaussian)

    kernel = tf.tensordot(gaussian, gaussian, axes=0)
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, 3, 1])

    blurred_images = tf.nn.depthwise_conv2d(
        images,
        kernel,
        strides=[1, 1, 1, 1],
        padding="SAME"
    )

    return tf.clip_by_value(blurred_images, 0.0, 255.0)


# Adjusting image brightness
def adjust_brightness(images, factor):
    bright_images = images * factor
    return tf.clip_by_value(bright_images, 0.0, 255.0)

# Evaluating model performance under each robustness condition
def evaluate_robustness(dataset, model):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return accuracy, precision, recall, f1, y_true, y_pred

# Creating corrupted test datasets
def create_corrupted_dataset(dataset, corruption_type, value):
    def corrupt_images(images, labels):
        if corruption_type == "noise":
            images = add_gaussian_noise(images, value)

        elif corruption_type == "blur":
            images = apply_gaussian_blur(images, value)

        elif corruption_type == "brightness":
            images = adjust_brightness(images, value)

        return images, labels

    return dataset.map(corrupt_images)

# Defining robustness test conditions
conditions = [
    ("Original", None, None),
    ("Gaussian Noise 0.05", "noise", 0.05),
    ("Gaussian Noise 0.10", "noise", 0.10),
    ("Gaussian Blur 3x3", "blur", 3),
    ("Gaussian Blur 5x5", "blur", 5),
    ("Brightness 0.7", "brightness", 0.7),
    ("Brightness 1.3", "brightness", 1.3)
]

# Storing robustness results
results = []
predictions = {}

# Evaluating original test dataset
print("\nEvaluating robustness conditions...")

for condition_name, corruption_type, value in conditions:
    print("\nCondition:", condition_name)

    if corruption_type is None:
        condition_dataset = test_dataset
    else:
        condition_dataset = create_corrupted_dataset(test_dataset, corruption_type, value)

    accuracy, precision, recall, f1, y_true, y_pred = evaluate_robustness(condition_dataset, model)

    results.append({
        "Condition": condition_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1
    })

    predictions[condition_name] = (y_true, y_pred)

    print("Accuracy:", round(accuracy, 4))
    print("Macro Precision:", round(precision, 4))
    print("Macro Recall:", round(recall, 4))
    print("Macro F1:", round(f1, 4))


# Converting results to DataFrame
results_df = pd.DataFrame(results)

# Calculating accuracy drop relative to original test data
original_accuracy = results_df.loc[results_df["Condition"] == "Original", "Accuracy"].iloc[0]

results_df["Accuracy Drop"] = original_accuracy - results_df["Accuracy"]

# Saving robustness results
results_file = output_dir / "robustness_results.csv"
results_df.to_csv(results_file, index=False)

print("\nRobustness testing completed.")
print("\nRobustness results:")
print(results_df.to_string(index=False))

print("\nResults saved to:")
print(results_file)

# Plotting confusion matrices for all robustness conditions
for condition_name, (y_true, y_pred) in predictions.items():
    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    plt.figure(figsize=(12, 10))

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", square=True, xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.title(f"Confusion Matrix - {condition_name}", fontsize=18)

    file_name = condition_name.lower().replace(" ", "_").replace(".", "")

    confusion_matrix_file = (output_dir / f"confusion_matrix_{file_name}.png")

    plt.savefig(confusion_matrix_file, dpi=300, bbox_inches="tight")

    plt.close()

    print("Saved confusion matrix:", confusion_matrix_file)

# Plotting robustness performance comparison
plot_df = results_df.copy()

# Shortening condition names for the figure
plot_df["Condition"] = [
    "Original",
    "Noise\n0.05",
    "Noise\n0.10",
    "Blur\n3×3",
    "Blur\n5×5",
    "Brightness\n0.7",
    "Brightness\n1.3"
]

metrics = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1"
]

x = np.arange(len(plot_df))
width = 0.19

fig, ax = plt.subplots(figsize=(14, 7.5))

for i, metric in enumerate(metrics):
    bars = ax.bar(x + (i - 1.5) * width, plot_df[metric], width, label=metric)

    # Adding value labels above bars
    for bar in bars:
        height = bar.get_height()

        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.010, f"{height:.4f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

# Formatting the figure
ax.set_title("Custom CNN Performance Under Robustness Conditions", fontsize=16, fontweight="bold", pad=20)

ax.set_xlabel("Robustness Condition", fontsize=11, labelpad=10)

ax.set_ylabel("Performance Score", fontsize=11,  labelpad=10)

ax.set_xticks(x)
ax.set_xticklabels(plot_df["Condition"], fontsize=10)

ax.set_ylim(0, 1.10)

ax.grid(axis="y", linestyle="--", alpha=0.25)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=10)

plt.tight_layout()

# Saving robustness performance comparison
performance_plot_file = (output_dir / "robustness_performance_comparison.png")

plt.savefig(performance_plot_file, dpi=300, bbox_inches="tight")

plt.close()

print("Saved robustness performance plot:", performance_plot_file)


