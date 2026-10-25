import os
import numpy as np
import random
import cv2
from PIL import Image
import shutil
from pathlib import Path
import uuid
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img


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

def load_images_from_folders(folder_path):
    """
    Loads images from subfolders and returns X (image array), y (class labels), and class names.
    """

    X = []
    y = []

    # List of class folders
    label_names = sorted(os.listdir(folder_path))  # Sorted for consistent label ordering

    for label, label_name in enumerate(label_names):
        label_folder = Path(folder_path) / label_name

        if not os.path.isdir(label_folder):
            continue

        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)

            try:
                # Loads images 
                img = load_img(img_path) 
                img_array = img_to_array(img)
                X.append(img_array)
                y.append(label)
            except Exception as e:
                print(f"Skipped {img_path} due to error: {e}")
                continue

    # Print statements for checking 
    print()
    print(f"Length of X is {len(X)}")
    print(f"Length of y is {len(y)}")

    return np.array(X), np.array(y)

def augment_images(output_dir, class_names, target_count, seed):
    """
    Augments each class folder until it reaches target_count.
    """

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    random.seed(seed)

    # Iterate through each class
    for class_name in class_names:

        # Path to current class
        class_path = Path(output_dir) / class_name

        # Get all images
        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # Only original images can be used for augmentation
        original_images = [f for f in images if '_aug_' not in f] 

        generate_count = target_count - len(images)

        if generate_count <= 0:
            print(f"[{class_name}] No augmentation needed.")
            continue

        print(f"[{class_name}] Augmenting {generate_count} images...")

        for i in range(generate_count):

            # Cycle through randomly shuffled original images
            if i % len(original_images) == 0:
                random.shuffle(original_images)

            img_name = original_images[i % len(original_images)]
            img_path = os.path.join(class_path, img_name)

            try:
                # Load image and convert to batch
                img = load_img(img_path)
                x = np.expand_dims(img_to_array(img), axis=0)

                # Generate one augmented image
                batch = next(datagen.flow(x, batch_size=1))
                augmented_img = array_to_img(batch[0])

                # Create filename
                original_base = os.path.splitext(img_name)[0]
                unique_id = uuid.uuid4().hex[:8]
                filename = f"{class_name}_aug_{unique_id}.jpeg"

                # Save image
                augmented_img.save(os.path.join(class_path, filename))

                print(
                    f"[{class_name}] Generated {i + 1}/{generate_count} "
                    f"→ Saved {len(images) + i + 1}"
                )

            except Exception as e:
                print(f"[{class_name}] Skipped {img_name}: {e}")

def create_dataset_splits(X, y, seed, test_val_size):

    """
    This function splits the data into 400 images for the training set, 75 images for validation set, 75 images for the test set
    """

    # First, split the dataset into temp and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_val_size, random_state= seed, stratify=y)

    # Now split the temp set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=test_val_size, random_state=seed, stratify=y_temp)

    # Print statements for checking 
    print()
    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:  ", y_val.shape)
    print("y_test shape: ", y_test.shape)
    print()

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_split_data(X, y, split_name, output_path, class_names):

    """
    Saves images into nested folders based on their split and class label.
    """

    for idx, (image, label) in enumerate(zip(X, y)):
        # Create folder
        leaf_class = class_names[label]
        leaf_class_dir = Path(output_path)/ split_name / leaf_class
        os.makedirs(leaf_class_dir, exist_ok=True)

        # Define the output file path
        filepath = os.path.join(leaf_class_dir, f"{leaf_class}_img_{idx}.jpg")

        # Saving the image 
        image = np.clip(image, 0, 255).astype(np.uint8)
        cv2.imwrite(filepath, image)

    print(f"Saved {len(X)} images in {split_name} folder")
        
if __name__ == "__main__":

    # Defining paths  
    dataset_path = "D:/11. 2026/2026 Semester 2/Image processing/archive(1)/PlantVillage"
    downsample_subset_path = "D:/11. 2026/2026 Semester 2/Image processing/downsample_subset"
    subset_path = "D:/11. 2026/2026 Semester 2/Image processing/subset"

    # Defining key variables 
    seed = 42
    subset_class_names = ["Tomato_healthy", "Tomato_Late_blight", "Tomato_Septoria_leaf_spot", "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato_Bacterial_spot"]

    # Downsample target count is 550 images per class
    downsample_target_count = 550 

    # Downsampling each class to equal the target_count and saving into downsample_subset_path
    downsample_images(dataset_path, downsample_subset_path, subset_class_names, downsample_target_count, seed)

    # Loading images from subset_path
    X, y = load_images_from_folders(downsample_subset_path)

    # Validation and test sets both have the same size - 75 images per class
    # This is multiplied by 5 since there are 5 classes, for making the training, validation and test sets
    test_val_size = 75*5 

    # Creating training, validation and testing sets
    X_train, X_val, X_test, y_train, y_val, y_test = create_dataset_splits(X, y, seed, test_val_size)

    # Creating training, validation and test folders
    for split_name in ["train", "validation", "test"]:
        output_dir = Path(subset_path) / split_name
        os.makedirs(output_dir, exist_ok=True)

    # Creating nested folders for training, validation and test set 
    save_split_data(X_train, y_train, "train", subset_path, subset_class_names)
    save_split_data(X_val, y_val, "validation", subset_path, subset_class_names)
    save_split_data(X_test, y_test, "test", subset_path, subset_class_names)

    # Augmenting images to ensure each class meets the subset_target_count 
    # Target count is 600 images per class
    subset_target_count = 600 
    training_path = Path(subset_path) / "train"
    augment_images(training_path, subset_class_names, subset_target_count, seed)








