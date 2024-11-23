import os
from PIL import Image


def split_grid_to_individual(grid_image_path, output_dir, grid_size):
    """
    Split a grid image into individual images and save them in the output directory.

    Args:
        grid_image_path (str): Path to the grid image file.
        output_dir (str): Directory to save individual images.
        grid_size (int): Size of the grid (e.g., 8 for an 8x8 grid).
    """
    try:
        grid = Image.open(grid_image_path)
        width, height = grid.size

        # Calculate the size of each individual image
        img_width = width // grid_size
        img_height = height // grid_size

        print(f"Processing: {grid_image_path}")
        print(f"Grid dimensions: {grid_size}x{grid_size}")
        print(f"Individual image size: {img_width}x{img_height}")

        # Split images
        counter = 0
        for row in range(grid_size):
            for col in range(grid_size):
                # Cropping coordinates
                left = col * img_width
                top = row * img_height
                right = left + img_width
                bottom = top + img_height

                # Crop and save individual image
                img = grid.crop((left, top, right, bottom))
                output_filename = (
                    f"{os.path.splitext(os.path.basename(grid_image_path))[0]}_{counter}.png"
                )
                output_path = os.path.join(output_dir, output_filename)
                img.save(output_path)
                counter += 1

        print(f"Successfully split into {counter} individual images\n")
        return counter

    except Exception as e:
        print(f"Error processing {grid_image_path}: {str(e)}")
        return 0


def main():
    input_dir = "generated_images"  # Folder dengan grid images
    output_dir = "individual_images_Common_Rust"  # Folder untuk menyimpan file hasil potong
    grid_size = 8  # Grid size (e.g., 8x8)

    # Buat output directory jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    total_images = 0

    # Iterasi melalui semua file di folder input
    for grid_file in os.listdir(input_dir):
        if grid_file.endswith(".png"):  # Hanya memproses file PNG
            grid_path = os.path.join(input_dir, grid_file)
            total_images += split_grid_to_individual(grid_path, output_dir, grid_size)

    # Cetak total file hasil
    print("==================================================")
    print(f"Total images generated: {total_images}")
    print("\nGrid splitting process completed!")


if __name__ == "__main__":
    main()
