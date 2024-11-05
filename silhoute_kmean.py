import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_images_from_nested_folders(root_folder):
    """Load all images from nested folders and flatten them into feature vectors."""
    images = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
            if img is not None:
                img_resized = cv2.resize(img, (244, 244))  # Resize to a standard size
                img_flattened = img_resized.flatten()  # Flatten the image into a 1D vector
                images.append(img_flattened)
                print(f"Loaded image: {img_path}")
    print(f"Total images loaded: {len(images)}")
    return np.array(images)

def find_optimal_k_with_silhouette(data, min_k=3, max_k=64, sample_fraction=0.2):
    """Find the optimal K using Silhouette Score on a subset of data."""
    n_samples = len(data)
    max_k = min(max_k, n_samples - 1)

    # Split data into sample for silhouette scoring and a set aside portion
    sample_data, _ = train_test_split(data, test_size=(1 - sample_fraction), random_state=0)
    print(f"Using {len(sample_data)} samples out of {n_samples} for silhouette scoring")

    best_k = min_k
    best_score = -1
    silhouette_scores = []

    for k in range(min_k, max_k + 1):
        # Perform K-means clustering with K clusters
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(sample_data)
        
        # Calculate silhouette score for current K
        if len(set(labels)) > 1:  # Ensure that there are at least 2 clusters
            score = silhouette_score(sample_data, labels)
            silhouette_scores.append(score)
            print(f"K={k}, Silhouette Score={score}")
            
            # Update best K if current score is higher
            if score > best_score:
                best_k = k
                best_score = score
        else:
            silhouette_scores.append(-1)
            print(f"K={k}, Silhouette Score=-1 (less than 2 clusters)")

    # Plot the Silhouette Scores for different values of K
    plt.plot(range(min_k, max_k + 1), silhouette_scores, 'bo-')
    plt.title('Silhouette Method for Optimal K (20% Sample)')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.show()

    return best_k, best_score

# Example usage
if __name__ == "__main__":
    # Path to root folder containing subfolders with images
    root_folder = "./corns/"

    # Load image data and preprocess
    data = load_images_from_nested_folders(root_folder)

    # Find the optimal K using 20% sample data for Silhouette Score
    optimal_k, optimal_score = find_optimal_k_with_silhouette(data, min_k=3, max_k=64, sample_fraction=0.1)
    print(f"Optimal K: {optimal_k}, with Silhouette Score: {optimal_score}")
