import time
import numpy as np
import pandas as pd
import tensorflow as tf

from model import build_model, train_model
from evaluate import plot_history, calculate_metrics, plot_confusion_matrix, plot_misclassified_samples

def load_images_from_folders(dataset_path, image_size, batch_size, seed):
    """
    Loads images from subfolders 
    """

    # Load the training split
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path+'/train',
        image_size = image_size,
        color_mode="rgb",  # Converts to RGB 
        batch_size=batch_size, 
        shuffle=True,  
        seed=seed          # Ensures the random shuffle is reproducible
    )

    # Load the validation split
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path+'/validation',
        image_size = image_size,
        color_mode="rgb",  
        batch_size=batch_size, 
        shuffle=False,  
    )

    # Load the test split
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path+'/test',
        image_size = image_size,
        color_mode="rgb",  
        batch_size=batch_size, 
        shuffle=False,  
    )

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":

    # Defining paths  
    dataset_path = "./subset"

    # Defining key variables
    seed = 42
    batch_size = 64
    image_size = (224, 224)
    num_epochs = 50 # Change to 50 later
    best_model_file = "best_model.keras"

    # Loading images from dataset_path
    train_dataset, val_dataset, test_dataset = load_images_from_folders(dataset_path, image_size, batch_size, seed)

    # Use TensorFlow's actual class order
    class_names = test_dataset.class_names

    # Initialising integer labels
    int_labels = [i for i in range(len(class_names))]

    print("Class order:", class_names)
    
    # Build and train the model
    model = build_model(image_size)

    # Count model parameters
    total_params = model.count_params()
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )

    print("\n===== Model Parameters =====")
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    # Measure training time
    start_time = time.perf_counter()

    history = train_model(
        model, train_dataset, val_dataset, num_epochs, best_model_file
    )

    training_time = time.perf_counter() - start_time

    print("\n===== Training Time =====")
    print(f"Training time: {training_time:.2f} seconds")

    # Saving history 
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("training_history.csv", index=False)

    # Loading history and best model 
    history_df = pd.read_csv("training_history.csv")
    model = tf.keras.models.load_model(best_model_file)
    
    # Evaluating metrics and plotting confusion matrix
    y_test = np.concatenate([y for x, y in test_dataset], axis=0)
    X_test = np.concatenate([x for x, y in test_dataset], axis=0)

    y_pred = calculate_metrics(test_dataset, y_test, model, class_names)
    plot_confusion_matrix(y_test, y_pred, class_names, int_labels)

    # Plotting accuracy and validation accuracy and loss curves 
    plot_history(history_df)

    # Visualising miclassified samples 
    plot_misclassified_samples(X_test, y_pred, y_test, class_names, seed)






