import numpy as np
import cv2
import os
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def load_images_with_labels(root_folder):
    """Load images from nested folders, assign labels based on folder names, and flatten them into feature vectors."""
    images = []
    labels = []
    class_names = sorted(os.listdir(root_folder))  # Assuming each subfolder corresponds to a class

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(root_folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        for file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
            if img is not None:
                img_resized = cv2.resize(img, (244, 244))  # Resize to a standard size (244x244 in this case)
                img_flattened = img_resized.flatten()  # Flatten the image into a 1D vector
                images.append(img_flattened)
                labels.append(label)
                print(f"Loaded image: {img_path} from class: {class_name}")
    
    print(f"Total images loaded: {len(images)}")
    return np.array(images), np.array(labels), class_names

def sample_images(images, labels, n_samples_per_class=500):
    """Sample a fixed number of images per class."""
    sampled_images = []
    sampled_labels = []

    unique_classes = np.unique(labels)
    for cls in unique_classes:
        # Get all images of the current class
        class_indices = np.where(labels == cls)[0]
        # Randomly sample n_samples_per_class from the current class
        sampled_class_indices = np.random.choice(class_indices, n_samples_per_class, replace=False)
        sampled_images.extend(images[sampled_class_indices])
        sampled_labels.extend(labels[sampled_class_indices])

    return np.array(sampled_images), np.array(sampled_labels)

def distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def get_centroid(points):
    """Calculate the centroid of a list of points."""
    return np.mean(points, axis=0)

def assign_to_clusters(data, centroids):
    """Assign each data point to the nearest centroid."""
    clusters = [[] for _ in centroids]
    labels = []
    for point in data:
        distances = [distance(point, centroid) for centroid in centroids]
        closest_centroid_index = np.argmin(distances)
        clusters[closest_centroid_index].append(point)
        labels.append(closest_centroid_index)
    return clusters, np.array(labels)

def update_centroids(clusters):
    """Update centroids as the mean of all points assigned to each cluster."""
    return [get_centroid(cluster) for cluster in clusters]

def convergence_reached(old_centroids, new_centroids, tolerance=1e-4):
    """Check if the centroids have converged."""
    return all(distance(old, new) < tolerance for old, new in zip(old_centroids, new_centroids))

def canopy_clustering(data, threshold1, threshold2):
    """Perform canopy clustering on the dataset."""
    canopies = []
    for point in data:
        if not any(in_canopy(point, canopy, threshold1) for canopy in canopies):
            new_canopy = [point]
            for other_point in data:
                if distance(point, other_point) < threshold2:
                    new_canopy.append(other_point)
            canopies.append(new_canopy)
    return canopies

def in_canopy(point, canopy, threshold):
    """Check if a point is within the canopy defined by the threshold."""
    return any(distance(point, center) < threshold for center in canopy)

def improved_kmeans(data, k, threshold1, threshold2, max_iterations=100):
    """Perform K-means clustering with initial centroids from canopy clustering."""
    canopies = canopy_clustering(data, threshold1, threshold2)
    initial_centroids = [get_centroid(canopy) for canopy in canopies[:k]]
    return kmeans(data, initial_centroids, max_iterations)

def kmeans(data, initial_centroids, max_iterations=100):
    """Standard K-means clustering algorithm."""
    centroids = initial_centroids
    labels = np.zeros(len(data))
    for _ in range(max_iterations):
        clusters, labels = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if convergence_reached(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids, labels

def calculate_silhouette_score(data, labels):
    """Calculate the Silhouette Score for the clustering result."""
    return silhouette_score(data, labels)

def find_optimal_k_with_silhouette(data, min_k=3, max_k=64, threshold1=0.5, threshold2=0.8):
    """Find the optimal K using Silhouette Score."""
    n_samples = len(data)  # Total number of data points
    max_k = min(max_k, n_samples - 1)  # Ensure K does not exceed n_samples - 1

    best_k = min_k
    best_score = -1
    silhouette_scores = []

    for k in range(min_k, max_k + 1):
        # Perform clustering with K clusters
        clusters, centroids, labels = improved_kmeans(data, k, threshold1, threshold2)
        
        # Calculate silhouette score for current K
        if len(set(labels)) > 1:  # Ensure that there are at least 2 clusters
            score = calculate_silhouette_score(data, labels)
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
    plt.title('Silhouette Method for Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.show()

    return best_k, best_score

if __name__ == "__main__":
    # Path to root folder containing subfolders with images
    root_folder = "./corns"

    # Load image data and labels
    data, labels, class_names = load_images_with_labels(root_folder)

    # Sample 500 images from each class
    sampled_data, sampled_labels = sample_images(data, labels, n_samples_per_class=10)

    # Find the optimal K using Silhouette Score in the range 3 to 64
    optimal_k, optimal_score = find_optimal_k_with_silhouette(sampled_data, min_k=1, max_k=64)
    print(f"Optimal K: {optimal_k}, with Silhouette Score: {optimal_score}")
