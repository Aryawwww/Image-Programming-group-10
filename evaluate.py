
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_history(history):
    """
    This function plots the training and validation accuracy and loss.
    """
    epoch_range = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], "r", label="Training accuracy")
    plt.plot(history['val_accuracy'], "b", label="Validation accuracy")
    plt.title("Training and validation accuracy", fontsize = 16)
    plt.xlabel("Epochs", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.xticks(epoch_range)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], "r", label="Training loss")
    plt.plot(history['val_loss'], "b", label="Validation loss")
    plt.title("Training and validation loss", fontsize = 16)
    plt.xlabel("Epochs", fontsize = 12)
    plt.ylabel("Loss", fontsize = 12)
    plt.xticks(epoch_range)
    plt.legend()

    plt.savefig("history.png", dpi=300, bbox_inches='tight')

    plt.show()

def calculate_metrics(test_dataset, y_test, model, label_names):
    """
    This function calculates the evaluation metrics - accuracy, precision, recall, F1 score.
    """    

    # Predicting class probabilities
    y_pred_probs = model.predict(test_dataset)

    # Converting probabilities to class predictions
    y_pred = np.argmax(y_pred_probs, axis=1)  

    # Computing evaluation metrics per class
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)

    # Converting to a DataFrame for better readability
    df_report = pd.DataFrame(report).transpose()

    # Display the table
    print("\n==================== Evaluation Metrics Report ====================")
    print(df_report)
    print("\n")

    # Saving to CSV file
    df_report.to_csv("evaluation_metrics.csv")

    return y_pred


def plot_confusion_matrix(y_true, y_pred, label_names, int_labels):
    """
    This function plots the confusion matrix.
    """

    # Computing confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels = int_labels)

    # Plotting the confusion matrix
    plt.figure(figsize=(12,10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", square=True, xticklabels= label_names, yticklabels= label_names)
    plt.xlabel("Predicted Labels", fontsize = 12)
    plt.ylabel("True Labels", fontsize = 12)
    plt.title("Confusion Matrix of Custom CNN Classifier", fontsize = 18)

    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')

    plt.show()

def plot_misclassified_samples(X_test, y_pred, y_true, class_names, seed):

    # Finding indices where predictions are not equal to the true labels
    misclassified_indices = np.where(y_pred != y_true)[0]

    # Randomly choose 4 images; seed ensures reproducibility
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(misclassified_indices, size=4, replace=False)

    plt.figure(figsize=(12, 12))

    for i, index in enumerate(selected_indices):
        
        # Extract the image data
        img = X_test[index].astype("uint8")
        
        # Get the true text label and predicted text label
        true_label = y_true[index]
        pred_label = y_pred[index]
        
        # Create a grid subplot 
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        
        true_label_name = class_names[true_label]
        predict_label_name = class_names[pred_label]

        plt.title(f"True label: {true_label_name}\nPredicted label: {predict_label_name}", fontsize=12)
        plt.axis("off")

    plt.suptitle("Sample Predictions", fontsize=18)
    plt.tight_layout()
    plt.show()
