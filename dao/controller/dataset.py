"""
Load images and masks helper functions
Input:
-Satelite Images (CH,H,W)
Output:
- Images after transformations and convert to float tensor (CH,H,W)
"""

import numpy as np
import rasterio
import torch


def to_float_tensor(img):
    img = torch.from_numpy(np.moveaxis(img, -1, 0)).float()
    return img


def load_image(path):  # Input:CH,H,W  Output:H,W,CH
    img = rasterio.open(path)
    img = img.read()
    img = img.transpose((1, 2, 0))
    return img


def load_mask(path):  # Input:CH,H,W  Output:H,W,CH

    x = path.split(".")

    mask_name = x[0] + "_mask." + x[1]
    mask_name_parts = mask_name.split("rgbnir")

    mask = rasterio.open(mask_name_parts[0] + "_" + mask_name_parts[1])
    mask = mask.read()
    mask = np.squeeze(mask, axis=(0,))
    mask = np.float32(mask)
    return mask
