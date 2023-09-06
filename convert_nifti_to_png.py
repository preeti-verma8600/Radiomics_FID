import os
import nibabel as nib
import cv2
from tqdm import tqdm
import argparse

# Set other parameters
VIEWS = ['axial']
RESIZE_TO = 512
FOR_PIX2PIX = True
SLIDE_MIN = -1
SLIDE_MAX = 999999

def nifti_to_png(filepath, target_folder, filename):
    # Load NIfTI scan and extract data using nibabel
    nifti_file = nib.load(filepath)
    nifti_file_array = nifti_file.get_fdata()

    # Iterate over nifti_file_array in axial view to extract slices
    for view in VIEWS:
        for i in range(nifti_file_array.shape[2]):
            if FOR_PIX2PIX and (i < SLIDE_MIN or i > SLIDE_MAX):
                continue
            img = nifti_file_array[:, :, i]
            img = cv2.resize(img, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_AREA)

            # Create the target folder if it does not exist
            target_folder_w_view = os.path.join(target_folder, view)
            os.makedirs(target_folder_w_view, exist_ok=True)

            # Save the image as a PNG file
            cv2.imwrite(os.path.join(target_folder_w_view, f'{filename}_slice{i}.png'), img)

def main():
    parser = argparse.ArgumentParser(description="Convert NIfTI files to PNG images")
    parser.add_argument("input_folder", help="Path to the input folder containing NIfTI files")
    parser.add_argument("-o", "--output_folder", help="Path to the output folder for saving PNG files")
    args = parser.parse_args()

    # Set default output folder if not provided
    output_folder = args.output_folder or os.path.dirname(os.path.abspath(__file__))

    # Loop through NIfTI files in the input folder
    for filename in os.listdir(args.input_folder):
        if filename.endswith('.nii.gz'):
            nifti_to_png(filepath=os.path.join(args.input_folder, filename),
                         target_folder=output_folder,
                         filename=os.path.splitext(filename)[0])

if __name__ == "__main__":
    main()
