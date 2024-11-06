import torch
from matplotlib.figure import Figure
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from datasets import preprocess_inference_data, load_inference_data

from backboned_unet import Unet

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
inference_rasters_path = '../data/eval/Rasters/Harmonized/'
model_load_path = '../weights/model_weights_harmonized.pt'
raster_filename = 'clipped_verde_2019_03_05.tif'
output_mask_path = '../prediction_masks/' + raster_filename.split('.')[0] + '_outputmask' + raster_filename.split('.')[-1]
# TODO: This script currently supports generating the stitched snow binary mask for one file at a time in the directory,
# can add handling for multiple files dynamically as per requirement

def main():
    print(f'Device: {device}')
    model = Unet(backbone_name='resnet50', pretrained=True, encoder_freeze=True, classes=2)
    model = model.to(device)
    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    dataloader, num_samples = load_inference_data(inference_rasters_path)
    # Initialize variables for global bounds 
    raster_minx, raster_maxx, raster_miny, raster_maxy = None, None, None, None
    crs = None

    # Initialize variables to assemble the mask
    current_row_minx = None
    row_masks = []
    full_mask_rows = []

    with torch.no_grad():
        for sample in dataloader:
            print(sample)
            sample, nodata_mask = preprocess_inference_data(sample)
            nodata_mask = torch.squeeze(nodata_mask, 0)
            input_patch = sample['image']  
            bbox = sample['bbox']
            input_patch = input_patch.to(device)

            output_mask = model(input_patch)
            
            output_mask = torch.argmax(output_mask, 1)
            output_mask[nodata_mask] = -1 # Setting NoData pixels from input to -1 in output mask for distinguishability
            output_mask = output_mask.cpu().numpy().squeeze()

            bbox = bbox[0]
            # Assign global bounds
            minx, maxx, miny, maxy = bbox.minx, bbox.maxx, bbox.miny, bbox.maxy
            
            if minx_new is None or minx < minx_new:
                minx_new = minx
            if maxx_new is None or maxx > maxx_new:
                maxx_new = maxx
            if miny_new is None or miny < miny_new:
                miny_new = miny
            if maxy_new is None or maxy > maxy_new:
                maxy_new = maxy
            if crs is None:
                crs = sample['crs'][0]
                print('CRS of raster: {crs}')

            if current_row_minx is None:
                # First patch
                current_row_minx = minx
            # Check if we need to start a new row for the sampled grid
            elif minx == current_row_minx:
                # We've looped back to the start of a new row
                # Stack the current row masks horizontally and add to full_mask_rows
                row = np.hstack(row_masks)
                full_mask_rows.insert(0, row)
                row_masks = []

            # Append the output mask to the current row
            row_masks.append(output_mask)

        # Adding the last row after the loop exists
        if row_masks:
            row = np.hstack(row_masks)
            full_mask_rows.insert(0, row)
    
    # Stack all the rows vertically to form the full mask
    full_mask = np.vstack(full_mask_rows)
    print(f'Shape of generated output mask: {full_mask.shape}')
    # Generate transform for the output mask
    new_transform = from_bounds(
        west=raster_minx,
        south=raster_miny,
        east=raster_maxx,
        north=raster_maxy,
        width=full_mask.shape[1],
        height=full_mask.shape[0]
    )

    with rasterio.open(
        output_mask_path,
        'w',
        driver='GTiff',
        height=full_mask.shape[0],
        width=full_mask.shape[1],
        count=1,
        dtype=np.int8,
        crs=crs,
        transform=new_transform,
        nodata=-1 # Set -1 values to NoData
    ) as dst:
        dst.write(full_mask, 1)

if __name__ == '__main__':
    main()

