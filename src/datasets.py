from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples, IntersectionDataset
from torchgeo.samplers import GridGeoSampler
from torch.utils.data import DataLoader

""" 
    This file contains TorchGeo RasterDataset classes for Train/Eval/Inference rasters, and helper functions for loading/preprocessing data
    PlanetScope -> Input Raster, Lidar -> Target Raster (Binary Mask)
    Refer: https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html
    Note: Configure the appropriate file names in the below classes
"""

class PlanetScopeTrain(RasterDataset):
    filename_glob = "clipped_right*.tif"
    filename_regex = "clipped_right_(?P<date>\w{10})"
    date_format = "%Y_%m_%d"
    is_image = True
    all_bands = ["Band 1", "Band 2", "Band 3", "Band 4"]
    rgb_bands = ["Band 3", "Band 2", "Band 1"]

class LidarMaskTrain(RasterDataset):
    filename_glob = "clipped_label_right*.tif"
    filename_regex = "clipped_label_right_(?P<date>\w{10})"
    date_format = "%Y_%m_%d"
    is_image = False

class PlanetScopeEval(RasterDataset):
    filename_glob = "clipped_verde_2019*.tif"
    filename_regex = "clipped_verde_(?P<date>\w{10})"
    date_format = "%Y_%m_%d"
    is_image = True
    all_bands = ["Band 1", "Band 2", "Band 3", "Band 4"]
    rgb_bands = ["Band 3", "Band 2", "Band 1"]

class LidarMaskEval(RasterDataset):
    filename_glob = "clipped_verde_2019*.tif"
    filename_regex = "clipped_verde_(?P<date>\w{10})"
    date_format = "%Y_%m_%d"
    is_image = False

class PlanetScopeInference(RasterDataset):
    filename_glob = "clipped_verde_2019*.tif"
    filename_regex = "clipped_verde_(?P<date>\w{10})"
    date_format = "%Y_%m_%d"
    is_image = True
    all_bands = ["Band 1", "Band 2", "Band 3", "Band 4"]
    rgb_bands = ["Band 3", "Band 2", "Band 1"]

def load_train_data(input_rasters_path, label_path):
    """
    This function is used to load train data rasters and target masks, combine them and returns a Dataloader
    """
    input_data = PlanetScopeTrain(input_rasters_path)
    print('Loaded Training Input Raster Data:')
    print(input_data)
    print(f'Res: {input_data.res}')

    label_data = LidarMaskTrain(label_path)
    print('Loaded Training Label Data:')
    print(label_data)
    print(f'Res: {label_data.res}')

    dataset = input_data & label_data
    print('Combined Train Input/Label Data:')
    print(dataset.res)
    print(f'Res of combined input/label data: {dataset.res}')

    sampler = GridGeoSampler(dataset, size=512, stride=480)
    num_samples = sampler.__len__()
    print(f'Number of samples in train dataset: {num_samples}')

    dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

    return dataloader, num_samples

def load_eval_data(input_rasters_path, label_path):
    """
    This function serves a similar purpose to above, but for eval data.
    """
    input_data = PlanetScopeEval(input_rasters_path)
    print('Loaded Eval Input Raster Data:')
    print(input_data)
    print(f'Res: {input_data.res}')

    label_data = LidarMaskEval(label_path)
    print('Loaded Eval Label Data:')
    print(label_data)
    print(f'Res: {label_data.res}')

    dataset = input_data & label_data
    print('Combined Eval Input/Label Data:')
    print(dataset.res)
    print(f'Res of combined input/label data: {dataset.res}')

    sampler = GridGeoSampler(dataset, size=512, stride=512)
    num_samples = sampler.__len__()
    print(f'Number of samples in eval dataset: {num_samples}')

    dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

    return dataloader, num_samples

def load_inference_data(input_rasters_path):
    """
    This function serves a similar purpose to above, but for inference data.
    """
    inference_data = PlanetScopeInference(input_rasters_path)
    print('Loaded Inference Raster Data:')
    print(inference_data)
    print(f'Res: {inference_data.res}')
    sampler = GridGeoSampler(inference_data, size=512, stride=512)
    num_samples = sampler.__len__()
    print(f'Number of samples in inference dataset: {num_samples}')
    dataloader = DataLoader(inference_data, sampler=sampler, collate_fn=stack_samples)

    return dataloader, num_samples

    
def preprocess_train_data(sample):
    """
    This function is used to do necessary preprocessing to the input image and label mask before
    passing them through the model; specifically ordering the image bands in R,G,B order and dealing with NoData pixels
    """
    rgb_indices = []
    for band in PlanetScopeTrain.rgb_bands:
        rgb_indices.append(PlanetScopeTrain.all_bands.index(band))
    print(f'RGB Band Indices in Train data: {rgb_indices}')
    image, mask = sample['image'], sample['mask']
    image = image[:, rgb_indices, :, :] # Orders RGB bands in input appropriately
    mask[mask<0] = 0
    image_empty_pos = (image <= 0).any(dim=1, keepdim=True) 
    mask[image_empty_pos] = 0 # Hack to zero out parts of label mask where image has empty data
    sample['image'] = image
    sample['mask'] = mask
    return sample

def preprocess_eval_data(sample):
    """
    This function serves a similar purpose to above, but for eval data. For eval, the mask pixels to be ignored
    are set to -1 so that it can be ignored during metric computations
    """
    rgb_indices = []
    for band in PlanetScopeEval.rgb_bands:
        rgb_indices.append(PlanetScopeEval.all_bands.index(band))
    print(f'RGB Band Indices in Eval data: {rgb_indices}')
    image, mask = sample['image'], sample['mask']
    image = image[:, rgb_indices, :, :] # Orders RGB bands in input appropriately
    mask[mask<0] = -1
    image_empty_pos = (image <= 0).any(dim=1, keepdim=True) 
    mask[image_empty_pos] = -1 # Hack to ignore pixels from label mask where image has empty data (during metric computation)
    sample['image'] = image
    sample['mask'] = mask
    return sample

def preprocess_inference_data(sample):
    """
    This function serves a similar purpose for inference-time image data. During inference, there is no target mask.
    A mask for NoData pixels is also returned so that the corresponding pixels in the stitched and rasterized predicted mask
    can be marked as NoData before saving to a .tif file.
    """
    rgb_indices = []
    for band in PlanetScopeInference.rgb_bands:
        rgb_indices.append(PlanetScopeInference.all_bands.index(band))
    print(f'RGB Band Indices in Inference data: {rgb_indices}')
    image = sample['image']
    image = image[:, rgb_indices, :, :] # Orders RGB bands in input appropriately
    sample['image'] = image
    image_empty_pos = (image <= 0).any(dim=1, keepdim=True) # Returns a mask for pixels with NoData
    return sample, image_empty_pos