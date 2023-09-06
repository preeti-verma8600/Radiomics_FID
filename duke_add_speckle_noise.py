import os
import shutil
import argparse
import numpy as np
import SimpleITK as sitk

def calculate_variance_from_percentage(percentage_noise, max_pixel_value):
    """
    Calculate the variance for speckle noise based on the percentage_noise.

    Args:
        percentage_noise (float): The desired percentage of noise to be added.
        max_pixel_value (int): The maximum pixel value in the input image.

    Returns:
        float: The calculated variance for speckle noise.
    """
    variance = (percentage_noise / 100) * max_pixel_value
    return variance

def add_speckle_noise_to_images(input_folder, output_base_folder, percentage_noise_list):
    """
    Add speckle noise to images in the input folder and save the noisy images to the output folder.

    Args:
        input_folder (str): Path to the input folder containing the images.
        output_base_folder (str): Path to the output base folder (default: created in the current script directory).
        percentage_noise_list (list of int): List of percentages to apply noise (default: [1, 10]).
    """
    # Determine the output base folder if not provided
    base_folder = output_base_folder or os.path.join(os.path.dirname(__file__), "output_base_folder")
    base_folder_name = os.path.basename(input_folder)  # Get the base folder name of the input folder

    for percentage_noise in percentage_noise_list:
        # Create a subfolder for the noisy images based on the percentage of noise
        noise_folder = os.path.join(base_folder, f"{base_folder_name}_{percentage_noise}_percent_speckle_noise")
        if not os.path.exists(noise_folder):
            os.makedirs(noise_folder)

        for filename in os.listdir(input_folder):
            if (filename.endswith(".png") and "_mask" not in filename) or (filename.endswith(".nii.gz") and "_0001" in filename):
                # Read and process the input image
                image_path = os.path.join(input_folder, filename)
                image = sitk.ReadImage(image_path)
                image_np = sitk.GetArrayFromImage(image)
                max_pixel_value = np.iinfo(image_np.dtype).max
                variance = calculate_variance_from_percentage(percentage_noise, max_pixel_value)

                # Add speckle noise to the image
                speckle_noise = np.random.normal(0, variance, image_np.shape)
                noisy_image_np = np.clip(image_np + speckle_noise, 0, max_pixel_value).astype(image_np.dtype)
                noisy_image = sitk.GetImageFromArray(noisy_image_np)
                noisy_image.CopyInformation(image)

                # Save the noisy image to the output folder
                output_path = os.path.join(noise_folder, filename)
                sitk.WriteImage(noisy_image, output_path)
            else:
                # Copy non-image files directly to the output folder
                original_image_path = os.path.join(input_folder, filename)
                output_path = os.path.join(noise_folder, filename)
                shutil.copy(original_image_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Add speckle noise to images in a folder.")
    parser.add_argument("input_folder", help="Path to the input folder containing the images.")
    parser.add_argument("-o", "--output_base_folder", help="Path to the output base folder (default: created in the current folder where the script is saved)")
    parser.add_argument("-p", "--percentages", nargs="+", type=int, default=[1, 10], help="List of percentages to apply noise (default: [1, 10]).")

    args = parser.parse_args()
    input_folder = args.input_folder
    percentage_noise_list = args.percentages
    output_base_folder = args.output_base_folder

    add_speckle_noise_to_images(input_folder, output_base_folder, percentage_noise_list)

if __name__ == "__main__":
    main()
