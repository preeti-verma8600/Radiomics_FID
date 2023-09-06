#!/usr/bin/python3
# import medigan
# import sys
# sys.path.append('home/preeti/libs/')
# import medigan
from medigan import Generators
generators = Generators()

# Generate 10 samples using one of the medigan models

# generators.generate(model_id="00003_CYCLEGAN_MMG_DENSITY_FULL", num_samples=10, install_dependencies=True)
# generators.generate(model_id="00003_CYCLEGAN_MMG_DENSITY_FULL", num_samples=10, install_dependencies=True)
# generators.generate(model_id="00003_CYCLEGAN_MMG_DENSITY_FULL", input_path="models/00003_CYCLEGAN_MMG_DENSITY_FULL/images",
#     image_size=[1332, 800], num_samples=10, install_dependencies=True)  # we can use number for model id, model_id =4, then you can remove 
generators.generate(model_id="4", num_samples=143, install_dependencies=True) 
# Get the model's generate method and run it to generate 3 samples

# gen_function = generators.get_generate_function(model_id="00001_DCGAN_MMG_CALC_ROI", 
#                                                 num_samples=3)
# gen_function = generators.get_generate_function(model_id="00003_CYCLEGAN_MMG_DENSITY_FULL", 
#                                                 num_samples=3)
# gen_function()
# # Create a list of search terms and find the models that have these terms in their config.

# values_list = ['dcgan', 'Mammography', 'inbreast']
# models = generators.find_matching_models_by_values(values=values_list, 
#                                                     target_values_operator='AND', 
#                                                     are_keys_also_matched=True, 
#                                                     is_case_sensitive=False)
# print(f'Found models: {models}')
# # Create a list of search terms, find a model and generate

# values_list = ['dcgan', 'mMg', 'ClF', 'modalities', 'inbreast']
# generators.find_model_and_generate(values=values_list, 
#                                     target_values_operator='AND', 
#                                     are_keys_also_matched=True, 
#                                     is_case_sensitive=False, 
#                                     num_samples=5)
# # Rank the models by a performance metric and return ranked list of models

# ranked_models = generators.rank_models_by_performance(metric="SSIM", 
#                                                         order="asc")
# print(ranked_models)
# # Find the models, then rank them by a performance metric and return ranked list of models

# ranked_models = generators.find_models_and_rank(values=values_list, 
#                                                 target_values_operator='AND',
#                                                 are_keys_also_matched=True,
#                                                 is_case_sensitive=False, 
#                                                 metric="SSIM", 
#                                                 order="asc")
# print(ranked_models)
# # Find the models, then rank them, and then generate samples with the best ranked model.

# generators.find_models_rank_and_generate(values=values_list, 
#                                         target_values_operator='AND',
#                                         are_keys_also_matched=True,
#                                         is_case_sensitive=False, 
#                                         metric="SSIM", 
#                                         order="asc", 
#                                         num_samples=5)
# # Find all models that contain a specific key-value pair in their model config.

# key = "modality"
# value = "Full-Field Mammography"
# found_models = generators.get_models_by_key_value_pair(key1=key, 
#                                                         value1=value, 
#                                                         is_case_sensitive=False)
# print(found_models)

# model specific parameters
# inputs= [
#     "input_path: default=models/00003_CYCLEGAN_MMG_DENSITY_FULL/images, help=the path to .png mammogram images that are translated from low to high breast density or vice versa",
#     "image_size: default=[1332, 800], help=list with image height and width. Images are rescaled to these pixel dimensions.",
#     "gpu_id: default=0, help=the gpu to run the model on.",
#     "translate_all_images: default=False, help=flag to override num_samples in case the user wishes to translate all images in the specified input_path folder."
# ]