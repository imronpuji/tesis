import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tqdm import tqdm
import random

# Function to perform quantile normalization on an image array
def quantile_normalize(image_array, lower_quantile=0.01, upper_quantile=0.99):
    lower_bound = np.quantile(image_array, lower_quantile)
    upper_bound = np.quantile(image_array, upper_quantile)
    image_array = np.clip(image_array, lower_bound, upper_bound)
    image_array = (image_array - lower_bound) / (upper_bound - lower_bound)
    image_array *= 255
    return image_array

# Function to extract features from a sample of images using VGG16
def extract_features(main_folder, sample_ratio=0.05):
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    features = []
    image_paths = []

    # Collect image paths from subfolders
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for img_file in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, img_file)
                image_paths.append(img_path)

    # Randomly sample image paths based on sample_ratio
    sampled_paths = random.sample(image_paths, int(len(image_paths) * sample_ratio))

    for img_path in tqdm(sampled_paths, desc="Extracting Features"):
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = quantile_normalize(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        feature = model.predict(img_array)
        features.append(feature.flatten())

    return np.array(features)

# Optimized function to find the best k using sampling and KMeans
def find_optimal_k(data):
    sample_data = data
    N = len(sample_data)
    max_k = int(np.sqrt(N))
    print(f"Max K: {max_k}")
    print(f"Total sample size: {N}")

    silhouette_map = {}

    # Calculate silhouette scores at intervals
    for k in range(2, max_k + 1, 5):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(sample_data)
        score = silhouette_score(sample_data, kmeans.labels_)
        silhouette_map[k] = score

    # Sort silhouette scores to find best initial k
    sorted_silhouette_map = dict(sorted(silhouette_map.items(), key=lambda item: item[1], reverse=True))
    
    # Fine-tune around the best k
    additional_scores = {}
    for k in list(sorted_silhouette_map.keys())[:min(3, len(sorted_silhouette_map))]:  # check top 3 best scores
        for adj_k in [k - 1, k + 1]:
            if adj_k >= 2 and adj_k not in silhouette_map:
                kmeans = KMeans(n_clusters=adj_k, random_state=42).fit(sample_data)
                score = silhouette_score(sample_data, kmeans.labels_)
                additional_scores[adj_k] = score

    # Merge and re-sort final silhouette scores
    silhouette_map.update(additional_scores)
    final_sorted_map = dict(sorted(silhouette_map.items(), key=lambda item: item[1], reverse=True))
    optimal_k = list(final_sorted_map.keys())[0]

    return optimal_k, final_sorted_map

# Main function
if __name__ == "__main__":
    main_folder = './corns'  # Path to main folder with subfolders of images
    
    print("Extracting features from sampled images...")
    data = extract_features(main_folder, sample_ratio=0.05)  # Using 5% sampling rate

    print("Running optimized algorithm to find best k...")
    optimal_k, silhouette_scores = find_optimal_k(data)

    print(f"Optimal number of clusters: {optimal_k}")
    print("Silhouette scores for each k:", silhouette_scores)
