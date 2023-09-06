import os
import shutil

def keep_images_without_masks(input_folder):
    # Create a new folder with 'without_mask' in its name
    new_folder_name = input_folder + '_without_mask'
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)

    # Iterate through the files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image (not a z)
        if os.path.isfile(file_path) and filename.lower().endswith('.png'):
            # if not '_mask'  in filename:      # uncomment for original bcdr
            if '_0001S_grayscale'  in filename:   # for synthetic bcdr

                # Copy images without '_mask' to the new 'without_mask' folder
                new_file_path = os.path.join(new_folder_name, filename)
                shutil.copy(file_path, new_file_path)

if __name__ == "__main__":
    input_folder = "/home/preeti/uab_internship/bcdr_w_masks_png"     # path to bcdr dataset
    keep_images_without_masks(input_folder)

