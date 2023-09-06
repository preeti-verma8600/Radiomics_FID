import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Replace this with the path to the folder containing your images
folder_path = "/home/preeti/uab_internship/duke_dataset/Phase 0001 images/Only-Phase-1"
# folder_path = "/home/preeti/uab_internship/duke_dataset/correct_masks"
# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Loop through the files in the folder
for filename in file_list:
    # Check if the file is in the desired format
    if filename.endswith(".nii.gz"):
        # Load the NIfTI image using SimpleITK
        image_path = os.path.join(folder_path, filename)
        img = sitk.ReadImage(image_path)
        
        # Get the image data as a NumPy array
        image_data = sitk.GetArrayFromImage(img)

        # Display the image using matplotlib
        plt.imshow(image_data[:, :, image_data.shape[2] // 2], cmap='gray')
        plt.title(filename)
        plt.show()
