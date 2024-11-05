import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed


def kmeans_segmentation(image, k):
    """Apply K-means clustering for image segmentation."""
    pixel_values = image.reshape((-1, 3)).astype(np.float32)
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(pixel_values)
    segmented_image = kmeans.cluster_centers_[labels].reshape(image.shape).astype(np.uint8)
    return segmented_image


def process_image(image_path, k_values, input_dir, output_dir, subset, is_training=True):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    if image is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return

    relative_path = os.path.relpath(os.path.dirname(image_path), input_dir)

    if is_training:
        for k in k_values:
            print(f"Processing image {os.path.basename(image_path)} with K={k} for {subset}")
            output_subdir = os.path.join(output_dir, subset, f"k{k}", relative_path)
            os.makedirs(output_subdir, exist_ok=True)

            segmented_image = kmeans_segmentation(image, k)
            save_path = os.path.join(output_subdir, f"{os.path.splitext(os.path.basename(image_path))[0]}_k{k}.jpg")
            cv2.imwrite(save_path, segmented_image)
    else:
        print(f"Processing image {os.path.basename(image_path)} for {subset}")
        output_subdir = os.path.join(output_dir, subset, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        save_path = os.path.join(output_subdir, f"{os.path.splitext(os.path.basename(image_path))[0]}.jpg")
        cv2.imwrite(save_path, image)


def process_and_save_images(input_dir, output_dir, k_values, n_jobs=-1):
    """Process and save images with different K values using parallel processing."""
    image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            image_path = os.path.join(root, file)
            image_paths.append(image_path)

    train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

    # Validate dataset in parallel
    Parallel(n_jobs=n_jobs)(
        delayed(process_image)(image_path, k_values, input_dir, output_dir, "val", is_training=False)
        for image_path in val_paths
    )

    # Train dataset in parallel
    Parallel(n_jobs=n_jobs)(
        delayed(process_image)(image_path, k_values, input_dir, output_dir, "train", is_training=True)
        for image_path in train_paths
    )


# Example usage
input_directory = "./corns"
output_directory = "segmented_images"
k_values = [2]
process_and_save_images(input_directory, output_directory, k_values, n_jobs=-1)
