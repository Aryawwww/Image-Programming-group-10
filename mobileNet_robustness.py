"""
MobileNetV3-Small robustness test using the SAME corruption settings and the
SAME evaluation pipeline as the Custom CNN robustness script, so the two
models can be compared like for like.

Conditions (7 = Original + the 6 corruption experiments):
  Original
  Gaussian noise  sigma 0.05, 0.10   (std = sigma * 255 on 0-255 pixels)
  Gaussian blur   3x3, 5x5 kernel    (separable Gaussian, sigma = 1.0)
  Brightness      x0.7, x1.3         (pixels * factor, clipped to 0-255)

Images are loaded with tf.keras.utils.image_dataset_from_directory (224x224,
float32 0-255, no extra rescaling - the MobileNet model contains its own
preprocessing), evaluated in batches on the test split.

Usage (from the repo root):
  python mobileNet_robustness.py \
      --model-path outputs/fine_tuned/finetune_3/best_finetuned_model.keras \
      --output-dir outputs/robustness/fine_tuned
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

IMAGE_SIZE = (224, 224)

CONDITIONS = [
    ("Original", None, None),
    ("Gaussian Noise 0.05", "noise", 0.05),
    ("Gaussian Noise 0.10", "noise", 0.10),
    ("Gaussian Blur 3x3", "blur", 3),
    ("Gaussian Blur 5x5", "blur", 5),
    ("Brightness 0.7", "brightness", 0.7),
    ("Brightness 1.3", "brightness", 1.3),
]


# ---- corruptions (identical to the Custom CNN script) -----------------------
def add_gaussian_noise(images, sigma):
    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=sigma * 255.0)
    return tf.clip_by_value(images + noise, 0.0, 255.0)


def apply_gaussian_blur(images, kernel_size):
    sigma = 1.0
    x = tf.range(-(kernel_size // 2), kernel_size // 2 + 1, dtype=tf.float32)
    gaussian = tf.exp(-(x ** 2) / (2.0 * sigma ** 2))
    gaussian = gaussian / tf.reduce_sum(gaussian)
    kernel = tf.tensordot(gaussian, gaussian, axes=0)
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    blurred = tf.nn.depthwise_conv2d(images, kernel, strides=[1, 1, 1, 1], padding="SAME")
    return tf.clip_by_value(blurred, 0.0, 255.0)


def adjust_brightness(images, factor):
    return tf.clip_by_value(images * factor, 0.0, 255.0)


def create_corrupted_dataset(dataset, corruption_type, value):
    def corrupt(images, labels):
        if corruption_type == "noise":
            images = add_gaussian_noise(images, value)
        elif corruption_type == "blur":
            images = apply_gaussian_blur(images, value)
        elif corruption_type == "brightness":
            images = adjust_brightness(images, value)
        return images, labels
    return dataset.map(corrupt)


def evaluate(dataset, model):
    y_true, y_pred = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_score(y_true, y_pred, average="macro", zero_division=0),
            f1_score(y_true, y_pred, average="macro", zero_division=0),
            y_true, y_pred)


PRETTY = {
    "Tomato_Bacterial_spot": "Bacterial spot",
    "Tomato_Late_blight": "Late blight",
    "Tomato_Septoria_leaf_spot": "Septoria leaf spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato yellow leaf\ncurl virus",
    "Tomato_healthy": "Healthy",
}


def plot_confusion(cm, class_names, title, path):
    """Report-style matrix: blue heatmap, counts in cells, 'Number of images' colour bar."""
    cm = np.asarray(cm)
    labels = [PRETTY.get(n, n.replace("Tomato_", "").replace("_", " ")) for n in class_names]
    fig, ax = plt.subplots(figsize=(8, 6.3))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Number of images")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_performance(df, path, title):
    labels = ["Original", "Noise\n0.05", "Noise\n0.10", "Blur\n3×3", "Blur\n5×5",
              "Brightness\n0.7", "Brightness\n1.3"]
    metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    x = np.arange(len(df)); width = 0.19
    fig, ax = plt.subplots(figsize=(14, 7.5))
    for i, m in enumerate(metrics):
        bars = ax.bar(x + (i - 1.5) * width, df[m], width, label=m)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.4f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Robustness Condition"); ax.set_ylabel("Performance Score")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.10)
    ax.grid(axis="y", linestyle="--", alpha=0.25); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="MobileNetV3-Small robustness test with the Custom CNN settings.")
    p.add_argument("--model-path", default="outputs/fine_tuned/finetune_3/best_finetuned_model.keras")
    p.add_argument("--test-dir", default="dataset/test")
    p.add_argument("--output-dir", default="outputs/robustness/fine_tuned")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--model-title", default="MobileNetV3-Small")
    a = p.parse_args()

    tf.random.set_seed(42); np.random.seed(42)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        a.test_dir, image_size=IMAGE_SIZE, batch_size=a.batch_size, shuffle=False)
    class_names = test_dataset.class_names
    print("\nClass order:")
    for i, n in enumerate(class_names): print(i, n)

    model = tf.keras.models.load_model(a.model_path)
    if model.output_shape[-1] != len(class_names):
        raise ValueError(f"Model has {model.output_shape[-1]} outputs but test data has {len(class_names)} classes.")
    print("\nModel:", a.model_path, "\nResults will be saved to:", out)

    results, predictions = [], {}
    for name, ctype, value in CONDITIONS:
        print("\nCondition:", name)
        ds = test_dataset if ctype is None else create_corrupted_dataset(test_dataset, ctype, value)
        acc, prec, rec, f1, y_true, y_pred = evaluate(ds, model)
        results.append({"Condition": name, "Accuracy": acc, "Macro Precision": prec,
                        "Macro Recall": rec, "Macro F1": f1})
        predictions[name] = (y_true, y_pred)
        print(f"Accuracy {acc:.4f} | Macro P {prec:.4f} | Macro R {rec:.4f} | Macro F1 {f1:.4f}")
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
        fname = name.lower().replace(" ", "_").replace(".", "")
        (out / f"classification_report_{fname}.txt").write_text(report)

    df = pd.DataFrame(results)
    df["Accuracy Drop"] = df.loc[df["Condition"] == "Original", "Accuracy"].iloc[0] - df["Accuracy"]
    df.to_csv(out / "robustness_results.csv", index=False)
    print("\nRobustness results:\n", df.to_string(index=False))
    print("\nNote: Original accuracy with the PIL loader used earlier was 0.9920 (372/375) for Fine-tune 3.")

    for name, (y_true, y_pred) in predictions.items():
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        fname = name.lower().replace(" ", "_").replace(".", "")
        acc_c = np.trace(cm) / cm.sum()
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out / f"confusion_matrix_{fname}.csv")
        plot_confusion(cm, class_names,
                       f"{a.model_title}\n{name} test set (n = {int(cm.sum())}) | Accuracy: {acc_c*100:.2f}%",
                       out / f"confusion_matrix_{fname}.png")
    plot_performance(df, out / "robustness_performance_comparison.png",
                     f"{a.model_title} Performance Under Robustness Conditions")
    print("\nSaved all outputs to", out)


if __name__ == "__main__":
    main()
