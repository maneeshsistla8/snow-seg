# SnowSeg
A pipeline for training and evaluating a UNet model for snow cover segmentation using satellite imagery.

This was developed as part of the machine learning research efforts to estimate the dynamic snow cover in the Salt-Verde region in Arizona, undertaken by the Center of Hydrologic Innovations (ASU) in collaboration with Salt River Project (SRP). High resolution (3m) satellite imagery from Planet Labs
is used as input to the machine learning model, with rasterized lidar surveys (conducted by SRP) used as ground truth labels.

## Requirements
The code in this repository is organized in the form of Python scripts, that are based on the PyTorch framework and run in a Python3 CUDA environment. Modules from the following python packages are utilized, and these can be installed via pip or conda/mamba:
* [torch](https://pytorch.org/get-started/locally/)
* [torchvision](https://pytorch.org/vision/stable/index.html)
* [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/)
* [torchgeo](https://torchgeo.readthedocs.io/en/stable/)
* [rasterio](https://rasterio.readthedocs.io/en/stable/installation.html)
* [numpy](https://numpy.org/install/)
* [matplotlib](https://matplotlib.org/stable/install/index.html)

## Components
### Code
| File/Folder        | Description           | Notes  |
| :-------------: |:-------------| :-----|
| [src/datasets.py](src/datasets.py) | Dataset classes and utility functions for loading/preprocessing data | The dataset classes are based on [RasterDataset](https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html#) from torchgeo, please refer the relevant documentation to appropriately configure the file regex for loading rasters. Also contains utility functions called during training for loading and preprocessing data. |
| [src/train.py](src/train.py) | Python script for model training | This script handles the training flow, from data loading to model training/saving. Paths and training parameters are configured within the script and can be modified as necessary. <br> **Usage:** *python3 train.py* |
| [src/eval.py](src/eval.py) | Python script for model evaluation | This script handles the evaluation flow, from data loading to evaluation of the trained model using various metrics. Paths are configured within the script and can be modified as necessary. <br> **Usage:** *python3 eval.py*  |
| [src/inference.py](src/inference.py) | Python script for generating snow mask for a given inference raster | This script handles the inference flow for a single sample image. Output binary masks are generated on a patch level and are stitched together into a raster. Paths are configured within the script and can be modified as necessary. <br> **Usage:** *python3 inference.py*  |
| [src/backboned_unet](src/backboned_unet) | PyTorch implementation of the base UNet model architecture | This module is imported when loading the UNet model in the above Python scripts. |
| [utilities/wrap_search_create_<br>request_download_bywatershed.sh](utilities/wrap_search_create_request_download_bywatershed.sh) | Sample bash script to search, filter and order PS imagery via CLI | This script can be used to search, filter and order PS imagery for a given AOI defined in a geojson file. A sample geojson file has also been uploaded in this folder. [This guide](https://planet-sdk-for-python-v2.readthedocs.io/en/latest/get-started/quick-start-guide/) can be followed to get started with the Planet SDK for Python, necessary for using this "No Code" CLI approach. |

### Model Weights
The model weights trained as part of our research are available in the [latest release](https://github.com/maneeshsistla8/snow-seg/releases/latest).

## Acknowledgements
This research was supported by a Collaborative Research Seed Grant Program from Arizona State University (ASU) and the Arizona Water Innovation Initiative (AWII). The Arizona Water Innovation Initiative is a multi-year partnership with the state led by Arizona State University’s Julie Ann Wrigley Global Futures Laboratory in collaboration with the Ira A. Fulton Schools of Engineering.
