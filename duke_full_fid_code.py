
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import time
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import logging
import csv
from pathlib import Path
from typing import List
import SimpleITK as sitk
# from dacite import from_dict
from radiomics.featureextractor import generalinfo, getFeatureClasses, getImageTypes, getParameterValidationFiles, imageoperations
from radiomics import featureextractor

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

# from pytorch_fid.inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=1,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
# parser.add_argument('--dims', type=int, default=2048,
#                     choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
#                     help=('Dimensionality of Inception features to use. '
#                           'By default, uses pool3 features'))
parser.add_argument('--dims', type=int, default=107,
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--save-stats', action='store_true',
                    help=('Generate an npz archive from a directory of samples. '
                          'The first path is used as input and the second as output.'))
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp', 'nii.gz'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=1, dims=107, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    # model.eval()

    # if batch_size > len(files):
    #     print(('Warning: batch size is bigger than the data size. '
    #            'Setting batch size to data size'))
    #     batch_size = len(files)

    # dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                          batch_size=batch_size,
    #                                          shuffle=False,
    #                                          drop_last=False,
    #                                          num_workers=num_workers)

    # pred_arr = np.empty((len(files), dims))

    # start_idx = 0

    # for batch in tqdm(dataloader):
    #     batch = batch.to(device)

    #     with torch.no_grad():
    #         pred = model(batch)[0]

    #     # If model output is not scalar, apply global spatial average pooling.
    #     # This happens if you choose a dimensionality not equal 2048.
    #     if pred.size(2) != 1 or pred.size(3) != 1:
    #         pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    #     pred = pred.squeeze(3).squeeze(2).cpu().numpy()

    #     pred_arr[start_idx:start_idx + pred.shape[0]] = pred

    #     start_idx = start_idx + pred.shape[0]

    image_paths = []
    mask_paths = []
    radiomics_results = []
    # print('files in get_activation', files)
    # image_paths = sorted(list(Path(files).glob("*_img.jpg")) + list(Path(files).glob("*.png")))
    # for file in files:
    #     image_paths.extend(sorted(list(Path(file).glob("*_img.jpg")) + list(Path(file).glob("*.png"))))
    # image_paths = sorted(list(Path(files).glob("*_img.jpg")) + list(Path(files).glob("*.png")))
    # for file_path in files:
    #     if str(file_path).endswith("_mask.png") or str(file_path).endswith("_mask.jpg"):
    #         mask_paths.append(file_path)
    #     else:
    #         image_paths.append(file_path)

    # below five lines are correct
    # for file_path in files:
    #     if str(file_path).endswith("_mask.png") or str(file_path).endswith("_mask_synth.jpg"):
    #         mask_paths.append(file_path)
    #     else:
    #         image_paths.append(file_path)
    for file_path in files:
        if str(file_path).endswith("_mask.png") or str(file_path).endswith("_mask_synth.jpg"):
            mask_paths.append(file_path)
        elif not str(file_path).endswith("_0001.nii.gz"):
            mask_paths.append(file_path)
        else:
            image_paths.append(file_path)
    # print('image_paths', image_paths)
    # print('mask_paths', mask_paths)

    pred_arr = np.empty((int(len(image_paths)), dims))
    # print('dimension of pred_arr', pred_arr.shape)

    count = 0
    # print('count of the loop-------1', count)
    # for image_path in tqdm(image_paths):
    for i, image_path in tqdm(enumerate(image_paths)):
        count += 1
        image_name = image_path.stem  # Get the name of the image file without the extension
        image_extension = image_path.suffix  # Get the extension of the image file

        # print('image_path and i++++++++++', image_path, i)
        
        # print('count of the loop-------2', count)
        if "_img_synth" in image_name:
            mask_name = image_name.replace("_img_synth", "_mask_synth")
        elif "_0001" in image_name:
            mask_name = image_name.replace("_0001", "")
        else:
            mask_name = image_name + "_mask"
        mask_file_name = mask_name + image_extension  # Append the image extension to the mask file name

        mask_path = image_path.with_name(mask_file_name)
        # print('mask_path-------------------', mask_path)
        # print('count of the loop', count)
        if mask_path.exists():
            sitk_image = sitk.ReadImage(str(image_path))
            sitk_mask = sitk.ReadImage(str(mask_path))
            # output = model.execute(sitk_image, sitk_mask, label=255)
            output = model.execute(sitk_image, sitk_mask) # voxelBased=True)#label=1)

            # print('output', output)
            radiomics_features = {}
            for feature_name in output.keys():
                if "diagnostics" not in feature_name:
                    radiomics_features[feature_name.replace("original_", "")] = float(output[feature_name])

            radiomics_results.append(radiomics_features)
            # radiomics = radiomics_features.
            pred_arr[i] = list(radiomics_features.values())
            # print('radiomic features-----computed', pred_arr)
            print(f"Total number of features extracted: {len(pred_arr[i])}")
    return pred_arr, radiomics_results, image_paths, mask_paths

# def save_features_to_csv(csv_file_path, image_paths, mask_paths, feature_data):
#     """Save the feature data to a CSV file.

#     Params:
#     -- csv_file_path   : Path to the CSV file where the results will be saved
#     -- image_paths     : List of image file paths
#     -- mask_paths      : List of mask file paths
#     -- feature_data    : Feature data to be saved in the CSV file
#     """
#     with open(csv_file_path, "w", newline="") as csv_file:
#         writer = csv.writer(csv_file)

#         # Write the header row
#         header = ["image_path", "mask_path"]
#         for feature_name in feature_data[0].keys():
#             header.append(feature_name)
#         writer.writerow(header)

#         # Write the rows for each image
#         for image_path, mask_path, features in zip(image_paths, mask_paths, feature_data):
#             mask_path = mask_path.with_name(mask_path.name.replace("_img_synth.jpg", "_mask_synth.jpg"))
#             row = [str(image_path), str(mask_path)]
#             row.extend(features.values())
#             writer.writerow(row)

#     print("Feature data saved to", csv_file_path)

# def save_features_to_csv(csv_file_path, image_paths, mask_paths, feature_data):
#     """Save the feature data to a CSV file.

#     Params:
#     -- csv_file_path   : Path to the CSV file where the results will be saved
#     -- image_paths     : List of image file paths
#     -- mask_paths      : List of mask file paths
#     -- feature_data    : Feature data to be saved in the CSV file
#     """
#     with open(csv_file_path, "w", newline="") as csv_file:
#         writer = csv.writer(csv_file)

#         # Write the header row
#         header = ["image_path", "mask_path"]
#         for feature_name in feature_data[0].keys():
#             header.append(feature_name)
#         writer.writerow(header)

#         # Write the rows for each image
#         for image_path, mask_path, features in zip(image_paths, mask_paths, feature_data):
#             mask_path = mask_path.with_name(mask_path.name.replace("_img_synth.jpg", "_mask_synth.jpg"))
#             row = [str(image_path), str(mask_path)]
#             row.extend(features.values())
#             writer.writerow(row)

#         # Compute and save the min and max values for each column
#         num_features = len(feature_data[0])
#         min_values = [np.min([data[feature_name] for data in feature_data]) for feature_name in feature_data[0].keys()]
#         max_values = [np.max([data[feature_name] for data in feature_data]) for feature_name in feature_data[0].keys()]
#         empty_row = [''] * 2  # Create an empty row to separate the data

#         # Write the rows for min values
#         writer.writerow(empty_row)
#         writer.writerow(['Min Values', ''] + min_values)

#         # Write the rows for max values
#         writer.writerow(empty_row)
#         writer.writerow(['Max Values', ''] + max_values)

#     print("Feature data saved to", csv_file_path)

def save_features_to_csv(csv_file_path, image_paths, mask_paths, feature_data):
    """Save the feature data to a CSV file.

    Params:
    -- csv_file_path   : Path to the CSV file where the results will be saved
    -- image_paths     : List of image file paths
    -- mask_paths      : List of mask file paths
    -- feature_data    : Feature data to be saved in the CSV file
    """
    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        header = ["image_path", "mask_path"]
        for feature_name in feature_data[0].keys():
            header.append(feature_name)
        writer.writerow(header)

        # Write the rows for each image
        for image_path, mask_path, features in zip(image_paths, mask_paths, feature_data):
            mask_path = mask_path.with_name(mask_path.name.replace("_img_synth.jpg", "_mask_synth.jpg"))
            row = [str(image_path), str(mask_path)]
            row.extend(features.values())
            writer.writerow(row)

        # Compute and save the min and max values for each column
        num_features = len(feature_data[0])
        min_values = [np.min([data[feature_name] for data in feature_data]) for feature_name in feature_data[0].keys()]
        max_values = [np.max([data[feature_name] for data in feature_data]) for feature_name in feature_data[0].keys()]
        empty_row = [''] * (num_features + 2)  # Create an empty row to separate the data

        # Write the rows for min values
        writer.writerow(empty_row)
        writer.writerow(['Min', ''] + min_values)

        # Write the rows for max values
        writer.writerow(empty_row)
        writer.writerow(['Max', ''] + max_values)

    print("Feature data saved to", csv_file_path)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

# def min_max_normalize(features, new_min, new_max):
#     # Calculate the minimum and maximum values of each feature across all images
#     min_values = np.min(features, axis=0)
#     max_values = np.max(features, axis=0)

#     # small_range_features = np.where((max_values - min_values) < 1e-6)[0]
#     # if len(small_range_features) > 0:
#     #     print(f"Warning: Features with small range found at indices: {small_range_features}")
#     #     for feature_idx in small_range_features:
#     #         print(f"Feature {feature_idx} - Min: {min_values[feature_idx]}, Max: {max_values[feature_idx]}")
#     #     # Handle small range features as desired (e.g., remove them or apply a different strategy).
#     # print('min, max, new_min, new_max',min_values, max_values, new_min, new_max)
#     small_range_features = np.where((max_values - min_values) < 1e-6)[0]
#     print('the hindering cases:',small_range_features)
#     # if len(small_range_features) > 0:
#     #     print(f"Warning: Features with small range found at indices: {small_range_features}")
#     #     for feature_idx in small_range_features:
#     #         print(f"Feature {feature_idx} - Min: {min_values[feature_idx]}, Max: {max_values[feature_idx]}")

#     #         # Get the row indices where this feature appears in the features array
#     #         row_indices_with_feature = np.where(features[:, feature_idx] == features[:, feature_idx].min())[0]

#     #         print(f"Row indices with Feature {feature_idx}: {row_indices_with_feature}")

#     #  # Find the indices where (max_values - min_values) < 1e-6
#     # small_range_features_indices = np.where((max_values - min_values) < 1e-6)[0]

#     # # Get the problematic features and their row indices
#     # problematic_features = features[:, small_range_features_indices]
#     # problematic_row_indices = np.where((max_values - min_values) < 1e-6)[0]

#     # # Print the problematic features along with their row and column numbers
#     # for idx, (col_idx, row_idx) in enumerate(zip(small_range_features_indices, problematic_row_indices)):
#     #     print(f"Problematic Feature {col_idx} at Row {row_idx} - Value: {problematic_features[idx]}")

#     # Perform Min-Max normalization on each feature
#     normalized_features = ((features - min_values) / (max_values - min_values)) * (new_max - new_min) + new_min

#     # Round the result to the specified number of decimal places
#     normalized_features = np.around(normalized_features, decimals=decimals)


#     return normalized_features


def min_max_normalize(features, new_min, new_max):
    # Calculate the minimum and maximum values of each feature across all images
    min_values = np.min(features, axis=0)
    max_values = np.max(features, axis=0)

    # Find feature columns with min_values == max_values
    equal_min_max_mask = min_values == max_values

    # Create a new copy of features to perform normalization
    normalized_features = np.copy(features)

    # Perform Min-Max normalization only for columns with different min and max values
    for idx, (min_val, max_val) in enumerate(zip(min_values, max_values)):
        if not equal_min_max_mask[idx]:
            normalized_features[:, idx] = ((features[:, idx] - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min

    # Replace columns with the same min and max values with the mean of new_min and new_max
    mean_value = (new_min + new_max) / 2
    normalized_features[:, equal_min_max_mask] = mean_value

    # # Round the result to the specified number of decimal places
    # normalized_features = np.around(normalized_features, decimals=decimals)

    return normalized_features

def calculate_activation_statistics(files, model, batch_size=1, dims=107,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act, radiomics_results, image_paths, mask_paths = get_activations(files, model, batch_size, dims, device, num_workers)
    print('features of radiomics------', act)
    # print('features of radiomics shape------', type(act))

    # to check NaN values in features
    features = act
    # if np.isnan(features).any():
    #     nan_indices = np.where(np.isnan(features))
    #     unique_nan_indices = np.unique(nan_indices[1])
    #     print("Warning: NaN values detected in the features array.")
    #     print("Number of NaN values for each feature:")
    #     for feature_idx in unique_nan_indices:
    #         nan_count = np.sum(np.isnan(features[:, feature_idx]))
    #         print(f"Feature {feature_idx}: {nan_count} NaN values")

    if np.isnan(features).any():
        nan_indices = np.where(np.isnan(features))
        unique_nan_indices = np.unique(nan_indices[1])
        print("Warning: NaN values detected in the features array.")
        print("Number of NaN values for each feature:")
        for feature_idx in unique_nan_indices:
            nan_count = np.sum(np.isnan(features[:, feature_idx]))
            print(f"Feature {feature_idx}: {nan_count} NaN values")

            # Get the row indices with NaN values for this feature
            row_indices_with_nan = nan_indices[0][nan_indices[1] == feature_idx]

            print(f"Row indices with NaN values for Feature {feature_idx}: {row_indices_with_nan}")


    normalized_act = min_max_normalize(act, 0, 7.45670747756958)
    print('features found are as follows', act)
    print('normalized_features----------',normalized_act)
    # print('files--------++++++++++', files)
    # # Generate a unique identifier using the current timestamp
    # unique_identifier = int(time.time())

    # # # Define the CSV file path with a unique identifier in the name
    # csv_file_path = f"radiomics_results_{unique_identifier}.csv"

    # # Define the CSV file path where the results will be saved
    # csv_file_path = "radiomics_results.csv"
    # Save the feature data to CSV

    # Extract the folder name from the first image file path
    folder_name = Path(files[0]).parent.stem

    # Generate a unique identifier using the current timestamp
    unique_identifier = int(time.time())

    # Define the CSV file path with a unique identifier and the folder name in the name
    csv_file_path = f"radiomics_results_{folder_name}_{unique_identifier}.csv"
    norm_csv_file_path = f"radiomics_results_normalized_{folder_name}_{unique_identifier}.csv"
    # save_features_to_csv(csv_file_path, image_paths, mask_paths, radiomics_results)
    save_features_to_csv(csv_file_path, image_paths, mask_paths, radiomics_results)
    # for normalized features:
    # Write the data to the CSV file
    with open(norm_csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Iterate through the rows and write each element to a new row in the CSV file
        for row in normalized_act:
            writer.writerow(row)
    # mu = np.mean(act, axis=0)
    # sigma = np.cov(act, rowvar=False)
    mu = np.mean(normalized_act, axis=0)
    sigma = np.cov(normalized_act, rowvar=False)
    # print('mu and sigma-----------------------------++++++++++++++++', mu, sigma)
    # print('mu and sigma length_____________________', len(mu), len(sigma))
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        # print('files in compute_statistics_of_path', files)
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)
    # print('m and s********************', m, s)
    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    # for p in paths:
    #     if not os.path.exists(p):
    #         raise RuntimeError('Invalid path: %s' % p)

    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    # model = InceptionV3([block_idx]).to(device)

    # m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
    #                                     dims, device, num_workers)
    # m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
    #                                     dims, device, num_workers)
    # fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    features_to_compute: List[str] = [
        "firstorder",
        "shape",
        "glcm",
        "glrlm",
        "gldm",
        "glszm",
        "ngtdm",
    ]

    settings = {}
    radiomics_results = []
    # Resize mask if there is a size mismatch between image and mask
    settings["setting"] = {"correctMask": True}
    # Set the minimum number of dimensions for a ROI mask. Needed to avoid error, as in our MMG datasets we have some masses with dim=1.
    # https://pyradiomics.readthedocs.io/en/latest/radiomics.html#radiomics.imageoperations.checkMask
    settings["setting"] = {"minimumROIDimensions": 1}

    # Set feature classes to compute
    settings["featureClass"] = {feature: [] for feature in features_to_compute}
    model = featureextractor.RadiomicsFeatureExtractor(settings)

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers)
    # print('m1, s1------------------------------', m1, s1)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, num_workers)
    # print('m2, s2------------------------------', m2, s2)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def save_fid_stats(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    # if not os.path.exists(paths[0]):
    #     raise RuntimeError('Invalid path: %s' % paths[0])

    # if os.path.exists(paths[1]):
    #     raise RuntimeError('Existing output file: %s' % paths[1])

    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    # model = InceptionV3([block_idx]).to(device)

    # print(f"Saving statistics for {paths[0]}")

    # m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
    #                                     dims, device, num_workers)

    # np.savez_compressed(paths[1], mu=m1, sigma=s1)
    if not os.path.exists(paths[0]):
        raise RuntimeError('Invalid path: %s' % paths[0])

    if os.path.exists(paths[1]):
        raise RuntimeError('Existing output file: %s' % paths[1])

    model = featureextractor.RadiomicsFeatureExtractor(settings)

    print(f"Saving statistics for {paths[0]}")

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers)

    np.savez_compressed(paths[1], mu=m1, sigma=s1)


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    if args.save_stats:
        save_fid_stats(args.path, args.batch_size, device, args.dims, num_workers)
        return

    fid_value = calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          args.dims,
                                          num_workers)
    print('FID: ', fid_value)


if __name__ == '__main__':
    main()


