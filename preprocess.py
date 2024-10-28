from sklearn.cluster import KMeans
import cv2
import numpy as np
import os
from joblib import Parallel, delayed

def kmeans_segmentation(image, k):
    """Apply K-means clustering for image segmentation."""
    pixel_values = image.reshape((-1, 3)).astype(np.float32)
    # Removed 'n_jobs' since it's not a valid argument for KMeans
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(pixel_values)
    segmented_image = kmeans.cluster_centers_[labels].reshape(image.shape).astype(np.uint8)
    return segmented_image

def process_image(file, root, input_dir, output_dir, k_values):
    image_path = os.path.join(root, file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return

    relative_path = os.path.relpath(root, input_dir)

    for k in k_values:
        print(f"Processing image {file} with K={k}")
        output_subdir = os.path.join(output_dir, f"k{k}", relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        segmented_image = kmeans_segmentation(image, k)
        save_path = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}_k{k}.jpg")
        cv2.imwrite(save_path, segmented_image)

def process_and_save_images(input_dir, output_dir, k_values, n_jobs=-1):
    """Process and save images with different K values using parallel processing."""
    for root, dirs, files in os.walk(input_dir):
        Parallel(n_jobs=n_jobs)(
            delayed(process_image)(file, root, input_dir, output_dir, k_values) for file in files
        )

# Example usage
input_directory = "./data"
output_directory = "segmented_images"
k_values = [2, 4, 8, 16, 32, 64]
process_and_save_images(input_directory, output_directory, k_values, n_jobs=-1)
