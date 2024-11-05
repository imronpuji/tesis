import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed


def process_image(image_path, input_dir, output_dir, subset):
    """Process and save the image to the specified subset directory."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to 224x224

    if image is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return

    relative_path = os.path.relpath(os.path.dirname(image_path), input_dir)
    
    # Create the output directory for the subset
    output_subdir = os.path.join(output_dir, subset, relative_path)
    os.makedirs(output_subdir, exist_ok=True)

    # Save the processed image
    save_path = os.path.join(output_subdir, f"{os.path.splitext(os.path.basename(image_path))[0]}.jpg")
    cv2.imwrite(save_path, image)


def process_and_save_images(input_dir, output_dir, n_jobs=-1):
    """Process and save images into training, validation, and testing subsets."""
    image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    # Split data into train (60%), validation (15%), and test (25%)
    train_paths, temp_paths = train_test_split(image_paths, test_size=0.4, random_state=42)  # 60% train
    val_paths, test_paths = train_test_split(temp_paths, test_size=0.625, random_state=42)  # 15% val, 25% test

    # Process and save images in parallel
    Parallel(n_jobs=n_jobs)(
        delayed(process_image)(image_path, input_dir, output_dir, "train")
        for image_path in train_paths
    )

    Parallel(n_jobs=n_jobs)(
        delayed(process_image)(image_path, input_dir, output_dir, "val")
        for image_path in val_paths
    )

    Parallel(n_jobs=n_jobs)(
        delayed(process_image)(image_path, input_dir, output_dir, "test")
        for image_path in test_paths
    )


# Example usage
input_directory = "./corns"  # Path to your input images
output_directory = "resized_images_corns"  # Path to save processed images
process_and_save_images(input_directory, output_directory, n_jobs=-1)
