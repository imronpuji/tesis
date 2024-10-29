import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import random

def kmeans_segmentation(image, k):
    """Apply K-means clustering for image segmentation."""
    pixel_values = image.reshape((-1, 3)).astype(np.float32)
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(pixel_values)
    segmented_image = kmeans.cluster_centers_[labels].reshape(image.shape).astype(np.uint8)
    return segmented_image, labels

def evaluate_kmeans(image, k):
    """Evaluate K-means clustering using silhouette score."""
    _, labels = kmeans_segmentation(image, k)
    pixel_values = image.reshape((-1, 3)).astype(np.float32)
    score = silhouette_score(pixel_values, labels)
    return score

def pso_find_best_k(image, k_range, num_particles=10, num_iterations=20):
    """Use Particle Swarm Optimization to find the best K value for K-means clustering."""
    # Initialize particles
    particles = [random.choice(k_range) for _ in range(num_particles)]
    velocities = [0] * num_particles
    personal_best_positions = particles[:]
    personal_best_scores = [evaluate_kmeans(image, k) for k in particles]
    global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
    global_best_score = max(personal_best_scores)

    for _ in range(num_iterations):
        for i in range(num_particles):
            # Update velocity and position
            r1, r2 = random.random(), random.random()
            velocities[i] = 0.5 * velocities[i] + r1 * (personal_best_positions[i] - particles[i]) + r2 * (global_best_position - particles[i])
            particles[i] = int(particles[i] + velocities[i])
            particles[i] = max(min(particles[i], max(k_range)), min(k_range))  # Ensure particles stay within bounds

            # Evaluate new position
            score = evaluate_kmeans(image, particles[i])
            if score > personal_best_scores[i]:
                personal_best_positions[i] = particles[i]
                personal_best_scores[i] = score
                if score > global_best_score:
                    global_best_position = particles[i]
                    global_best_score = score

    return global_best_position

def process_and_save_images(input_dir, output_dir, k_range):
    """Process and save images with the best K value found by PSO."""
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue
            
            relative_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            
            print(f"Finding best K for image {file}")
            best_k = pso_find_best_k(image, k_range)
            print(f"Best K for image {file} is {best_k}")
            segmented_image, _ = kmeans_segmentation(image, best_k)
            save_path = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}_k{best_k}.jpg")
            cv2.imwrite(save_path, segmented_image)

# Example usage
input_directory = "./data"
output_directory = "segmented_images"
k_range = range(2, 65)  # Define the range of possible K values
process_and_save_images(input_directory, output_directory, k_range)