import os
import random
import shutil
from pathlib import Path

def split_dataset(source_dir, output_dir, train_ratio=0.7):
    """
    Split dataset into training and testing sets.
    
    Args:
        source_dir (str): Directory containing the original images organized by class
        output_dir (str): Directory where to create train/test split
        train_ratio (float): Ratio of images to use for training (default: 0.7)
    """
    # Create output directory structure
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    
    # Get list of classes (subdirectories)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"\n[INFO] Splitting dataset with {train_ratio:.0%} training ratio")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    
    for class_name in classes:
        print(f"\nProcessing class: {class_name}")
        
        # Create class directories in train and test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get list of images for this class
        class_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Randomly shuffle images
        random.shuffle(images)
        
        # Calculate split index
        n_train = int(len(images) * train_ratio)
        
        # Split images into train and test sets
        train_images = images[:n_train]
        test_images = images[n_train:]
        
        print(f"Total images: {len(images)}")
        print(f"Training images: {len(train_images)}")
        print(f"Testing images: {len(test_images)}")
        
        # Copy images to respective directories
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)
            
        for img in test_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(test_dir, class_name, img)
            shutil.copy2(src, dst)
    
    print("\n[SUCCESS] Dataset split complete!")
    
    # Print final statistics
    print("\nFinal Dataset Statistics:")
    print("="*50)
    for split in ['train', 'test']:
        split_dir = os.path.join(output_dir, split)
        total = 0
        print(f"\n{split.upper()} Set:")
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            n_images = len(os.listdir(class_dir))
            total += n_images
            print(f"{class_name}: {n_images} images")
        print(f"Total: {total} images")

# Example usage
if __name__ == "__main__":
    # Define directories
    source_directory = "./data"  # Your original dataset directory
    output_directory = "./dataset"  # Where to create the split dataset
    
    # Your original dataset should be organized like this:
    # original_dataset/
    # ├── blight/
    # │   ├── image1.jpg
    # │   ├── image2.jpg
    # ├── common_rust/
    # │   ├── image1.jpg
    # │   ├── image2.jpg
    # └── ...
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Split dataset
    split_dataset(source_directory, output_directory, train_ratio=0.7)