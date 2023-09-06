import os
import numpy as np
from PIL import Image
import shutil
import argparse

def add_gaussian_noise(image_path, percentage):
    """
    Add Gaussian noise to an image.

    Args:
        image_path (str): The path to the input image.
        percentage (float): The percentage of noise to add to the image.

    Returns:
        PIL.Image.Image: The noisy image.
    """
    # Load the image using Pillow
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Calculate the noise range based on the percentage
    max_intensity = 255
    std = max_intensity * (percentage / 100)

    # Generate Gaussian noise with the same shape as the image
    noise = np.random.normal(loc=0, scale=std, size=image_array.shape)

    # Add the noise to the image
    noisy_image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)

    # Convert the NumPy array back to an image
    noisy_image = Image.fromarray(noisy_image_array)

    return noisy_image

def add_noise_to_images_in_folder(input_folder, output_base_folder, percentages):
    """
    Add Gaussian noise to all images in a folder and save the noisy images.

    Args:
        input_folder (str): The path to the input folder containing the images.
        output_base_folder (str): The base folder where noisy images will be saved.
        percentages (list of float): List of noise percentages to apply to each image.

    Returns:
        None
    """
    # Create folders for each percentage
    for percentage in percentages:
        output_folder = os.path.join(output_base_folder, f"bcdr_gaussian_noise_{percentage}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Check if the file is an image (not a mask)
        if filename.endswith(".png") and "_mask" not in filename:
            for percentage in percentages:
                # Create folder for the current percentage if it doesn't exist
                output_folder = os.path.join(output_base_folder, f"bcdr_gaussian_noise_{percentage}")
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Add Gaussian noise to the image with the specified percentage
                noisy_image = add_gaussian_noise(file_path, percentage)

                # Save the noisy image to the output folder with the same filename
                noisy_image_path = os.path.join(output_folder, filename)
                noisy_image.save(noisy_image_path)

        else:
            # Copy the original images to each output folder if no noise added
            for percentage in percentages:
                output_folder = os.path.join(output_base_folder, f"bcdr_gaussian_noise_{percentage}")
                output_path = os.path.join(output_folder, filename)
                shutil.copyfile(file_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add Gaussian noise to images in a folder.")
    parser.add_argument("input_folder", type=str, help="Input folder path")
    parser.add_argument("--output_base_folder", type=str, default=os.getcwd(), help="Output base folder path (default: current directory)")
    parser.add_argument("--percentages", type=float, nargs="+", default=[0.1, 1, 5, 10, 20, 50], help="List of noise percentages (default: [0.1, 1, 5, 10, 20, 50])")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_base_folder = args.output_base_folder
    percentages = args.percentages

    add_noise_to_images_in_folder(input_folder, output_base_folder, percentages)








