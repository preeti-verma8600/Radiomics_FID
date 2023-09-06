import os
import cv2
import shutil
import numpy as np
import argparse

def calculate_kernel_size(original_size, percentage_blur):
    """
    Calculate the kernel size for Gaussian blur based on a given percentage blur.

    Args:
        original_size (tuple): The original image dimensions (height, width).
        percentage_blur (float): The desired percentage of blur.

    Returns:
        int: The calculated kernel size for Gaussian blur.
    """
    max_dimension = max(original_size)
    kernel_size = int((percentage_blur / 100) * max_dimension)
    kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    return kernel_size

def add_blur_to_images(input_folder, output_base_folder, percentage_blurs):
    """
    Apply Gaussian blur to images in the input folder and save them in output folders.

    Args:
        input_folder (str): Path to the input folder containing images to be blurred.
        percentage_blurs (list): List of desired percentage of blur values to apply.
        output_base_folder (str, optional): Path to the output base folder (default: current script directory).

    Note:
        If the output_base_folder is not provided, output folders will be created in the same
        directory as the script.

    Each output folder will be named "bcdr_blur_{percentage_blur}_percent" and contain the
    blurred images.
    """
    if output_base_folder is None:
        output_base_folder = os.path.dirname(os.path.abspath(__file__))

    for percentage_blur in percentage_blurs:
        blur_folder = os.path.join(output_base_folder, f"bcdr_blur_{percentage_blur}_percent")
        if not os.path.exists(blur_folder):
            os.makedirs(blur_folder)

        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.png') and '_mask' not in filename:
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)

                original_shape = image.shape
                original_dtype = image.dtype

                kernel_size = calculate_kernel_size(original_shape[:2], percentage_blur)

                blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
                blurred_image = np.clip(blurred_image, 0, 255)
                blurred_image = blurred_image.astype(original_dtype)
                blurred_image = cv2.resize(blurred_image, (original_shape[1], original_shape[0]))
                blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

                output_path = os.path.join(blur_folder, filename)
                cv2.imwrite(output_path, blurred_image)
            
            else:
                original_image_path = os.path.join(input_folder, filename)
                output_path = os.path.join(blur_folder, filename)
                shutil.copy(original_image_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add blur to images in a folder.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder")
    parser.add_argument(
        "--output_base_folder",
        type=str,
        default=None,
        help="Path to the output base folder (default: current script directory)",
    )
    parser.add_argument(
        "--percentages",
        nargs="+",
        type=float,
        default=[0.1, 1, 5, 10, 20, 50],
        help="List of desired percentage of blur values (default: [0.1, 1, 5, 10, 20, 50])",
    )

    args = parser.parse_args()
    add_blur_to_images(args.input_folder, args.output_base_folder, args.percentages)
