import os
import cv2
import shutil
import numpy as np
import argparse

def calculate_variance_from_percentage(percentage_noise, max_pixel_value):
    """
    Calculate the variance for speckle noise based on a percentage of the maximum pixel value.

    Args:
        percentage_noise (float): The desired percentage of speckle noise.
        max_pixel_value (int): The maximum pixel value in the image.

    Returns:
        float: The calculated variance for speckle noise.
    """
    variance = (percentage_noise / 100) * max_pixel_value
    return variance

def add_speckle_noise_to_images(input_folder, output_base_folder, percentage_noises):
    """
    Add speckle noise to images in the input folder and save them in corresponding output folders.

    Args:
        input_folder (str): Path to the input folder containing images.
        output_base_folder (str): Path to the output base folder where noisy images will be saved.
        percentage_noises (list of float): List of desired percentage of speckle noise values for each image.

    Returns:
        None
    """
    for percentage_noise in percentage_noises:
        noise_folder = os.path.join(output_base_folder, f"bcdr_speckle_noise_{percentage_noise}_percent")
        if not os.path.exists(noise_folder):
            os.makedirs(noise_folder)

        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.png') and '_mask' not in filename:
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                max_pixel_value = np.iinfo(image.dtype).max
                variance = calculate_variance_from_percentage(percentage_noise, max_pixel_value)

                speckle_noise = np.random.normal(0, variance, image.shape)
                noisy_image = np.clip(image + speckle_noise, 0, max_pixel_value).astype(image.dtype)

                output_path = os.path.join(noise_folder, filename)
                cv2.imwrite(output_path, noisy_image)
            
            else:
                original_image_path = os.path.join(input_folder, filename)
                output_path = os.path.join(noise_folder, filename)
                shutil.copy(original_image_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add speckle noise to images.")
    parser.add_argument("input_folder", help="Path to the input folder containing images")
    parser.add_argument("--output_base_folder", help="Path to the output base folder (optional)")
    parser.add_argument("--percentages", type=float, nargs="+", default=[0.1, 1, 5, 10, 20, 50], help="List of noise percentages (default: [0.1, 1, 5, 10, 20, 50])")

    args = parser.parse_args()
    input_folder = args.input_folder
    output_base_folder = args.output_base_folder
    percentages = args.percentages

    if output_base_folder is None:
        output_base_folder = os.path.dirname(input_folder)

    add_speckle_noise_to_images(input_folder, output_base_folder, percentages)
