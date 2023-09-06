import os
import nibabel as nib
from PIL import Image
import numpy as np
import argparse
import glob

def extract_axial_slice(input_path, output_dir):
    """
    Extracts the axial slice from a NIfTI image and saves it as a PNG image.

    Parameters:
    input_path (str): Path to the input NIfTI image file.
    output_dir (str): Path to the output directory where the PNG image will be saved.

    Returns:
    None
    """
    # Load the NIfTI image
    nifti_img = nib.load(input_path)
    data = nifti_img.get_fdata()

    # Choose the axial slice index (0-based) you want to extract
    axial_slice_index = data.shape[2] // 2  # Selects the middle axial slice

    # Extract the axial slice
    axial_slice = data[:, :, axial_slice_index]

    # Normalize the slice to the [0, 255] range
    normalized_slice = ((axial_slice - axial_slice.min()) / (axial_slice.max() - axial_slice.min()) * 255).astype(np.uint8)

    # Convert to a PIL image
    pil_image = Image.fromarray(normalized_slice)

    # Rotate the image by 90 degrees towards the left
    rotated_image = pil_image.rotate(90, expand=True)

    # Get the original filename without the extension
    original_filename = os.path.basename(input_path)
    filename_without_extension = os.path.splitext(original_filename)[0]

    # Generate the output PNG filename
    output_filename = os.path.join(output_dir, f"{filename_without_extension}.png")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the axial slice as a PNG image
    rotated_image.save(output_filename)

    print(f"Axial slice saved as {output_filename}")

def main():
    """
    Main function to extract and save axial slices from NIfTI image files in a folder.

    Command Line Arguments:
    input_folder (str): Path to the input folder containing NIfTI files.
    output_dir (str, optional): Path to the output directory (default is None).

    Returns:
    None
    """
    parser = argparse.ArgumentParser(description='Extract and save axial slices from NIfTI image files in a folder.')
    parser.add_argument('input_folder', help='Path to the input folder containing NIfTI files')
    parser.add_argument('--output_dir', help='Path to the output directory (optional)', default=None)
    
    args = parser.parse_args()
    input_folder = args.input_folder
    output_dir = args.output_dir
    
    # Determine the output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        # If output_dir is not provided, use the default "input_folder_axial" directory in the script's location
        output_dir = os.path.join(script_dir, f"{os.path.basename(input_folder)}_axial")

    # Use glob to filter files based on filenames containing '_0001' or '_0001S_grayscale'
    file_patterns = ['*_0001.nii.gz', '*_0001S_grayscale.nii.gz']
    files_to_process = []
    for pattern in file_patterns:
        files_to_process.extend(glob.glob(os.path.join(input_folder, pattern)))

    for input_path in files_to_process:
        extract_axial_slice(input_path, output_dir)

if __name__ == "__main__":
    main()
