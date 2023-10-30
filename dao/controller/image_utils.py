import os
from io import BytesIO
from itertools import product

import numpy as np
import rasterio as rio
import torch
from dao.controller.dataset import to_float_tensor
from dao.controller.transform import DualCompose, ImageOnly, Normalize
from PIL import Image
from rasterio import windows
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess_image(img, dataset: str):
    """
    Image is converted in a proper tensor, so it can be procesed by the neural network
    Input shape: (X,512,512); Output shape: (1,X,512,512)
    """
    img = img.transpose((1, 2, 0))
    image_transform = transform_function(dataset)
    img_for_model = image_transform(img)[0]
    img_for_model = Variable(to_float_tensor(img_for_model), requires_grad=False)
    img_for_model = img_for_model.unsqueeze(0).to(device)

    return img_for_model


def transform_function(satelite):
    """
    Normalize the values of the tensor
    """
    if satelite == 2:
        image_transform = DualCompose(
            [
                ImageOnly(
                    Normalize(
                        mean=[
                            6.54707898e-02,
                            6.68120815e-02,
                            7.80696303e-02,
                            1.37027497e-01,
                            1.84557309e03,
                        ],
                        std=[
                            1.81216032e-02,
                            2.11627078e-02,
                            3.01965653e-02,
                            3.68936245e-02,
                            6.52083742e02,
                        ],
                    )
                )
            ]
        )
    else:
        image_transform = DualCompose(
            [
                ImageOnly(
                    Normalize(
                        mean=[
                            0.14308006,
                            0.12414238,
                            0.13847679,
                            0.14984046,
                            0.61647371,
                        ],
                        std=[0.0537779, 0.04049726, 0.03915002, 0.0497247, 0.48624467],
                    )
                )
            ]
        )
    return image_transform


def create_patches(dataset):
    """
    Generates blocks of (x, 512, 512) pixels from a satellite image.
    """
    patches = []

    def get_tiles(ds, width=512, height=512):
        nols, nrows = ds.meta["width"], ds.meta["height"]
        offsets = product(range(0, nols, width), range(0, nrows, height))
        big_window = windows.Window(
            col_off=0, row_off=0, width=nols, height=nrows  # type:ignore
        )

        for col_off, row_off in offsets:
            tile_window = windows.Window(
                col_off=col_off,  # type:ignore
                row_off=row_off,  # type:ignore
                width=width,  # type:ignore
                height=height,  # type:ignore
            ).intersection(big_window)
            tile_transform = windows.transform(tile_window, ds.transform)  # split

            yield tile_window, tile_transform

    with dataset as inds:
        tile_width, tile_height = 512, 512
        meta = inds.meta.copy()

        for window, transform in get_tiles(inds):
            if (int(window.width) == tile_width) and (
                int(window.height) == tile_height
            ):
                array = inds.read(window=window)
                patches.append(array)
    return patches, meta


def reconstruct_image(masks, metadata, img_shape, filename, level):
    """
    Combines a set of (1, 4, 512, 512) pixel blocks to generate a segmentation mask
    and saves the resulting mask locally.
    """
    pos = 0
    # C, H, W
    mask = np.zeros(shape=(1, img_shape[1], img_shape[2]))
    # rows = floor(H / 512), cols = floor(W / 512)

    for j in range(img_shape[2] // 512):
        for i in range(img_shape[1] // 512):
            cur_mask = masks[pos, 0, :, :, :]
            for k in range(512):
                for l in range(512):  # noqa
                    mask[0, i * 512 + k, j * 512 + l] = cur_mask[0, k, l]
            pos += 1

    # h, w = mask.shape[1], mask.shape[2]
    binary_mask = np.zeros(shape=mask.shape, dtype=np.uint8)
    for y in range(mask.shape[1]):
        for x in range(mask.shape[2]):
            binary_mask[0, y, x] = mask[0, y, x] > 0.5
    mask = binary_mask

    metadata["count"] = 1
    metadata["height"] = mask.shape[1]
    metadata["width"] = mask.shape[2]
    metadata["dtype"] = mask.dtype
    mask_filename = filename + "_MASK_{}.TIF".format(level)
    UPLOAD_DIRECTORY = "static/"
    with rio.open(
        os.path.join(UPLOAD_DIRECTORY, mask_filename), "w", **metadata
    ) as outds:
        outds.write(mask)

    return mask_filename, mask, metadata


def define_mask(mask):
    """
    Gets the first value of the segmentation mask
    """
    mask = mask[0]

    for x in range(512):
        for y in range(512):
            for z in range(2):
                mask[z, y, x] = int(mask[z, y, x] > 0.5)

    return mask.astype(int)


def convert_mask_to_png(filename, raster, metadata, colours, level):
    """
    Transforms a mask into a PNG image for visualization on web platforms.
    """
    new_metadata = metadata
    new_metadata["count"] = 3
    new_metadata["driver"] = "PNG"
    new_metadata["dtype"] = "uint8"

    png_filename = filename + "_{}.png".format(level)

    new_raster = np.zeros(shape=[3, new_metadata["height"], new_metadata["width"]])
    new_raster[0] = raster * colours[0]
    new_raster[1] = raster * colours[1]
    new_raster[2] = raster * colours[2]
    new_raster = new_raster.astype("uint8")
    UPLOAD_DIRECTORY = "static/"
    with rio.open(UPLOAD_DIRECTORY + png_filename, "w", **new_metadata) as dst:
        dst.write(new_raster)

    return png_filename


def convert_raster_to_png(filename, raster, metadata):
    """
    Converts a satellite image into a PNG image for visualization on web platforms.
    """
    new_metadata = metadata
    new_metadata["count"] = 3
    new_metadata["driver"] = "PNG"
    new_metadata["dtype"] = "uint8"

    png_filename = filename + "_imagen.png"
    raster = raster[:3]
    new_raster = (raster / 3512 * 255).astype("uint8")
    UPLOAD_DIRECTORY = "static/"
    with rio.open(UPLOAD_DIRECTORY + png_filename, "w", **new_metadata) as dst:
        dst.write(new_raster)

    return png_filename


def get_png_raster(filepath, sftp, metadata):
    """
    Converts an image file to a PNG format and saves it to a specified directory
    """
    last_slash = filepath.rfind("/") + 1  # Occurrence of the last slash + 1
    last_dot = filepath.rfind(".")  # Occurrence of the last dot
    filename = filepath[last_slash:last_dot]  # Image name

    last_slash = filepath.rfind("/") + 1  # Occurrence of the last slash + 1
    last_dot = filepath.rfind(".")  # Occurrence of the last dot
    filepath = (
        filepath[:last_slash]
        + "PREVIEW"
        + filepath[last_slash + 3 : last_dot]  # noqa
        + ".JPG"
    )

    file = BytesIO()
    sftp.getfo(filepath, file)
    file.seek(0)
    pic = np.array(Image.open(file))

    pic = pic.transpose((2, 0, 1))

    new_metadata = metadata
    new_metadata["count"] = 3
    new_metadata["driver"] = "PNG"
    new_metadata["dtype"] = "uint8"

    png_filename = filename + ".JPG"
    UPLOAD_DIRECTORY = "static/"
    with rio.open(UPLOAD_DIRECTORY + png_filename, "w", **new_metadata) as dst:
        dst.write(pic)

    return png_filename
