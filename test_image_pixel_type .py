import SimpleITK as sitk

# Read the image with SimpleITK
image_path1 = "/home/preeti/uab_internship/bcdr_w_masks_png/patientbcdr__3_image4_mass4457_is_benign_0.png"
image_path2 = "/home/preeti/uab_internship/random_motion_2/patientbcdr__3_image4_mass4457_is_benign_0.png"
image1 = sitk.ReadImage(image_path1)
image2 = sitk.ReadImage(image_path2)

# Get the pixel type as a string
pixel_type_string1 = image1.GetPixelIDTypeAsString()
pixel_type_string2 = image2.GetPixelIDTypeAsString()

print("Pixel type of image1:", pixel_type_string1)
print("Pixel type of image2:", pixel_type_string2)
