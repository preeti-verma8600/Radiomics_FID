import argparse
import numpy as np
from PIL import Image
import os
import shutil
import SimpleITK as sitk

def add_gaussian_noise(image_path, percentage):
    """
    Add Gaussian noise to an image and return the noisy image.

    Parameters:
    - image_path (str): The path to the input image file.
    - percentage (float): The percentage of noise to add.

    Returns:
    - noisy_image (SimpleITK.Image): The noisy image with Gaussian noise added.
    """
    # Load the image using SimpleITK
    image = sitk.ReadImage(image_path)

    # Convert the SimpleITK image to a NumPy array
    image_array = sitk.GetArrayFromImage(image)

    # Calculate the noise range based on the percentage
    max_intensity = image_array.max()
    std = max_intensity * (percentage / 100)

    # Generate Gaussian noise with the same shape as the image
    noise = np.random.normal(loc=0, scale=std, size=image_array.shape)

    # Add the noise to the image and clip values to the valid intensity range
    noisy_image_array = np.clip(image_array + noise, 0, max_intensity).astype(image_array.dtype)

    # Convert the NumPy array back to a SimpleITK image
    noisy_image = sitk.GetImageFromArray(noisy_image_array)
    noisy_image.CopyInformation(image)

    return noisy_image

def add_noise_to_images_in_folder(input_folder, output_base_folder, percentages):
    """
    Add Gaussian noise to images in the input folder and save noisy images to output folders.

    Parameters:
    - input_folder (str): The path to the folder containing input images.
    - output_base_folder (str): The base path for output folders (optional).
    - percentages (list of int): List of percentages to apply noise (default: [1, 10]).
    """
    # Get the base folder name without the path
    base_folder_name = os.path.basename(input_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Check if the file is an image (not a mask) based on filename extensions
        if (filename.endswith(".png") and "_mask" not in filename) or (filename.endswith(".nii.gz") and "_0001" in filename):
            for percentage in percentages:
                # Create folder for the current percentage if it doesn't exist
                output_folder = os.path.join(output_base_folder, f"{base_folder_name}_gaussian_noise_{percentage}%")
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Add Gaussian noise to the image with the specified percentage
                noisy_image = add_gaussian_noise(file_path, percentage)

                # Save the noisy image to the output folder with the same filename
                noisy_image_path = os.path.join(output_folder, filename)
                sitk.WriteImage(noisy_image, noisy_image_path)

        else:
            for percentage in percentages:
                # Create subfolder for the current percentage if it doesn't exist
                subfolder_name = f"{base_folder_name}_gaussian_noise_{percentage}%"
                output_folder = os.path.join(output_base_folder, subfolder_name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Construct the correct output path with the subdirectory
                output_path = os.path.join(output_folder, filename)

                # Copy the original images to each output folder if no noise added
                shutil.copyfile(file_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add Gaussian noise to images in a folder.")
    parser.add_argument("input_folder", help="Input folder path containing images.")
    parser.add_argument("-o", "--output_base_folder", help="Output base folder path (optional).")
    parser.add_argument("-p", "--percentages", nargs="+", type=int, default=[1, 10], help="List of percentages to apply noise (default: [1, 10]).")
    
    args = parser.parse_args()

    input_folder = args.input_folder
    output_base_folder = args.output_base_folder or os.path.dirname(os.path.abspath(__file__))
    percentages = args.percentages

    add_noise_to_images_in_folder(input_folder, output_base_folder, percentages)
