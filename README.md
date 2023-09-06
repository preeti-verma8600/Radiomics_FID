# Radiomics FID
This project put forths the novel metric, the Radiomic based Frechet Inception Distance (FRD) score for evaluating Generative Adversarial Networks (GANs) using radiomic features. The FID metric measures the distance between the distributions of radiomic features extracted from real-world images and generated images. It provides a quantitative measure of the quality of the generated images by comparing their feature distributions with those of real images.

The script is adapted from the original TTUR repository, and it uses PyTorch and PyRadiomics to perform radiomic feature extraction and FRD computation. The FRD score takes into account the radiomic feature statistics and can provide insights into the quality and authenticity of generated images.

## Features
Calculates the Radiomic FID score between the radiomic features of real and generated images.
Supports radiomic feature extraction using the PyRadiomics library.
Provides the option to save radiomic feature statistics for future use.
Offers flexible customization for batch size, device, and number of workers.
Outputs the calculated FID score for quality assessment.

## Prerequisite
Make sure you have the following libraries and packages installed before running the scripts:
- **Python version**: Python 3.7 or later
- **Python libraries/packages**: 
    ```
    1. os - Used for interacting with the operating system, such as file and directory operations.
    2. pathlib - Provides object-oriented filesystem path manipulation.
    3. argparse - Used for parsing command-line arguments and options.
    4. time - Provides time-related functions, which may be used for timing or scheduling tasks.
    5. numpy - Fundamental library for numerical operations and handling arrays.
    6. torch - PyTorch, a deep learning framework, used for neural network operations.
    7. torchvision - Part of PyTorch, it provides datasets, transforms, and models for computer vision tasks.
    8. PIL (Python Imaging Library) - Used for image processing and manipulation.
    9. scipy - Scientific library for various scientific and technical computing tasks. In your project, it might be used for certain scientific calculations or operations.
    10. logging - Used for generating log messages, which can be helpful for debugging and tracking the execution of your code.
    11. csv - Used for reading and writing CSV (Comma Separated Values) files, which are often used for data input and output.
    12. SimpleITK - SimpleITK is a simplified and user-friendly interface to the Insight Segmentation and Registration Toolkit (ITK), often used in medical image processing.
    13. medigan - It provides user-friendly medical image synthesis and allows users to choose from a range of pretrained generative models to generate synthetic datasets.
    14. nibabel - Used for reading and writing neuroimaging data formats, such as NIfTI and ANALYZE.
    15. cv2 (OpenCV) - OpenCV is a computer vision library used for various image and video processing tasks.
    16. tqdm - Provides a progress bar for iterating over sequences, which can be helpful for monitoring the progress of time-consuming operations.
    ```
- **Operating System**: Ubuntu 20.04 


## Installation
1. Install/clone the required dependencies:

```
To be able to work with this file, you will need to have downloaded the dataset. 

```

2. Clone the project repository into your workspace.

3. Extract the zip file in this exact location.

4. Make sure the script file has execution permissions. If not, run the following command:
    ```
    chmod +x <path_to_script_file>

    ```

## Usage/Execution
This repository comes with following python scripts according to the specific work: 

1. **To introduce different kind noise into the bcdr dataset:** The following scripts: `add_blur.py`, `add_gaussian_noise.py`, `add_speckle_noise.py`, introduce different kind of noise in the medical images depending on the percentage of noise. 

    To execute the 'add_blur.py', 'add_gaussian_noise.py',  'add_speckle_noise.py', run the following command:

        ```
        python <script_name> <path_to_input_image_folder> --output_base_folder <output_folder_path> --percentages 0.1 1 5 10 20 50

        ```
        Replace <script_name> with the name of script that you want to run.
        Replace <path_to_input_image_folder> with the path to your input folder containing the images. 
        --percentage_blurs: You can specify the desired percentage of noise values as a list. The provided values [0.1, 1, 5, 10, 20, 50] are the defaults.
        --output_base_folder (Optional): You can specify the path to the output base folder where the modified images will be saved. If not provided, the modified images will be saved in the same directory as the script.

    __Output:__
    Once the script run is complete, you can check the output folder(s) to find the noisy images with the applied noise type.

2. **To introduce different kind of noise in "Duke" dataset:** The followinf scripts can be utilized to create various variation of noise in 3D dataset: `duke_add_blur.py`, `duke_add_gaussian_noise.py`, `duke_add_speckle_noise.py`.
    To execute the any of the above script, run the following command:

        ```
        python <script_name> <path_to_input_image_folder> -o <output_folder_path> -p 1 10

        ```
    Replace <path_to_input_image_folder> with the path to your input folder containing the images, <output_base_folder_path> with the path where you want to save the modified images, and adjust the -p argument to specify the list of percentages noise varaiations you want to apply.

    __Output:__
    The script will process the images in the input folder, apply noises, depending on which script is run with the specified percentages, and save the modified images in the specified output folders. 

3. **To compute the FID score based on ImageNetV3 and RadImageNetV3 feature extractor models**
    # Frechet Inception Distance (FID) Calculator

    `fid_radimagenetv3.py` script calculates the Frechet Inception Distance (FID) between two distributions using a chosen feature extractor model. 

    ## Feature Extractor Models

    - **RadImageNet Model:** You can use the RadImageNet model for feature extraction. You can find the model source [here](https://github.com/BMEII-AI/RadImageNet).
    - **ImageNet Model:** Alternatively, you can use the ImageNet feature extractor model.

    ## Usage

        To calculate the FID between two sets of images, run the script as follows:

        ```bash
        python fid.py dir1 dir2 [--options]
        ```
        dir1: Path to the directory containing images from the first dataset.
        dir2: Path to the directory containing images from the second dataset.
        Options
        --model: Choose the feature extractor model (default: "imagenet" for ImageNet model).
        --description: Describe the run, e.g., specify the checkpoint name and important configuration information.
        --lower_bound: Calculate the lower bound of FID using a 50/50 split of images from dir1.
        --normalize_images: Normalize images from both data sources using min and max values of each sample.
        --is_split_per_patient: If the dataset is split for FID calculation, split it per patient.
        --reverse_split_ds1: Reverse the splitting order for the dataset from dir1 if it's split deterministically.
        --reverse_split_ds2: Reverse the splitting order for the dataset from dir2 if it's split deterministically.
        --is_only_splitted_loaded: If the dataset is split into two, load only the first split and ignore the second.
        --limit: Set the maximum number of images to load from each data source (default: 3000).

    ___Output:__
    The script will calculate the FID between the specified datasets and display the result. It will also store the results in a CSV file named fid.csv for future reference.

4. **To compute the FRD score**
    # Radiomics based Frechet Inception Distance (FID) Calculator

    ## How it works:

    i. **Radiomics features extraction**: `compute_frd.py` script leverages PyRadiomics to extract radiomics features from both real and generated images. Radiomics features capture quantitative information about the texture and shape of regions of interest (ROIs) within medical images.

    ii. **FID computation**: The Frechet Inception Distance (FID) is computed based on the extracted radiomics features. FID measures the similarity between two distributions of images, in this case, real medical images and images generated by a GAN. It quantifies how well the generated images match the real ones in terms of radiomics features.

    iii. **Quality evaluation**: The calculated FID value can be used to evaluate the quality of GAN-generated images within the context of radiomics analysis. Lower FID values indicate that the generated images are closer to the real images in terms of radiomics features, implying better image quality.

    ### Usage:

    - You can run the `compute_frd.py`script by providing the paths to the real and generated images as command-line arguments. For example:
    ```
    python compute_frd.py --device cuda:0  <path_to_generated_images> <path_to_real_images>

    ```

    ## Additional Options:

        - **Customization**: You have the flexibility to customize various options within the script to tailor it to your specific needs. These options include:

        - **Batch Size**: You can adjust the batch size used for processing images by modifying the `--batch-size` argument. A larger batch size can accelerate processing on systems with sufficient memory.

        - **Device**: To choose whether the script runs on CPU or GPU (if available), you can specify the `--device` argument. For example, you can use `--device cuda:0` to run on the first available GPU.

        - **Dimensionality of Inception Features**: You can set the dimensionality of the Inception features used for FID calculation by modifying the `--dims` argument. The default value is 102, but you can choose different values depending on your requirements.

        - **Saving Precomputed Statistics**: To save precomputed statistics for future use and avoid reprocessing images, you can use the `--save-stats` option. 
        


    __Note__: Depending on which dataset you're working on you need to tune the following parametes:

    ## Radiomics Feature Computation

        In the script, you will find a section that specifies the radiomics features to be computed for evaluation. These features are defined in the `features_to_compute` list, and they correspond to various aspects of the image data, such as texture, shape, and more. Depending on your dataset and whether it's in 2D or 3D, you may need to customize this list:

        - **For 2D Datasets (e.g., BCDR Dataset)**: The default feature set is configured to include both 2D and 3D features. In this case, we use `"shape2D"` to compute 2D shape features.

        - **For 3D Datasets (e.g., Duke Dataset)**: If you are working with a 3D dataset, you should modify the `features_to_compute` list to include `"shape"` instead of `"shape2D"`. This will enable the computation of 3D shape features.


    ## Customizing Parameters for Specific Datasets

        If you intend to use the script with the BCDR (Breast Cancer Digital Repository) dataset or similar datasets, you may need to adjust specific parameters to match your data. Here are the parameters that you might need to change:

        - **label**: In the script, there's an option to specify the label for the region of interest (ROI) in your images. By default, it's set to `255`, which works fine for the BCDR dataset. However, for the Duke dataset, you need to change it to '1'.

        - **dims**: The dimensionality of features used for FRD calculation is set to 102 by default, which works well for 2D datasets, depending on how many radiomics features are extracted. However, for **3D** datasets, it needs to be '107'. 
        __Note__: "The 'dims' parameter value depends on the types and quantity of radiomics features extracted, which may vary." 
         Customize the list of radiomics features to compute based on your dataset in 'calculate_frd_given_paths' function definition in `compute_frd.py`script:
            ### For 2D datasets like BCDR Dataset, use "shape2D"
            ### For 3D datasets like Duke Dataset, use "shape"

        These adjustments ensure that the script processes your data correctly and calculates the Frechet Inception Distance (FID) accurately. Make these changes as needed to suit the specific characteristics of your dataset.


## Troubleshooting

If you encounter issues while using the files in this package, you can try the following troubleshooting steps:

- **Check Dependencies**: Make sure all required dependencies are installed and up-to-date. Refer to the documentation.

- **Provide Correct Arguments**: When executing the script, ensure that you provide the correct number of arguments as specified in the documentation. Incorrect arguments can lead to errors or unexpected behavior.

- **Verify File Paths**: Double-check that you have provided the correct path for the folder. Incorrect file paths can prevent the scripts from accessing the required resources.

- **Clear Previous Data**: If you are rerunning the scripts, make sure to clear any previous data generated by the previous script execution. This ensures a clean execution environment and prevents conflicts or unexpected results.

- **Consult Documentation and Forums**: If the above steps do not resolve the issue, consult the package documentation for specific usage instructions and troubleshooting tips. You can also search online forums or communities related to the package for assistance. Often, others may have encountered similar issues and found solutions.

If the issue persists, consider submitting a bug report or reaching out to the package maintainers for further assistance.

---

<sup>
Preeti Verma, Richard Osuala - 
September 2023.
</sup>
