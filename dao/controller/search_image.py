import glob

import numpy as np
from api.schema.search_image import BoundingBox, ImageOut, ImagesOut
from pyproj import Transformer
from rasterio.io import MemoryFile



def get_bounding_box(dataset):
    """
    Obtains the coordinates of a satellite image in EPSG:4326 format.

    :param dataset: a raster dataset reader
    :type dataset: rio.DatasetReader
    """
    # Retrieves the original bounding box

    origin_bb = dataset.bounds

    transformer = Transformer.from_crs(dataset.profile["crs"], "epsg:4326")
    bottom, left = transformer.transform(origin_bb.left, origin_bb.bottom)
    top, right = transformer.transform(origin_bb.right, origin_bb.top)

    bounding_box = {
        "left": round(left, 3),
        "bottom": round(bottom, 3),
        "right": round(right, 3),
        "top": round(top, 3),
    }

    return bounding_box


def get_bounding_box_from_file(file):
    """
    Extracts the bounding box coordinates from a raster file.

    Reads the contents of the specified file, retrieves the bounding box information
    from the raster dataset, and returns the coordinates in EPSG:4326 format.

    Parameters:
    file (str): The path to the raster file.

    Returns:
    dict: A dictionary containing the coordinates of the bounding box
          in EPSG:4326 format, with keys 'left', 'bottom', 'right', and 'top'.
    """
    file = open(file, "rb")
    data = file.read()
    memfile = MemoryFile(data)
    dataset = memfile.open()
    return get_bounding_box(dataset)


def SearchImage(top: float, bottom: float, left: float, right: float):
    """
    Searches for images within the specified geographical coordinates.

    Retrieves images from a directory, obtains their bounding box information,
    and generates a list of ImageOut objects containing image metadata.

    Parameters:
    top (float): The top latitude of the search area.
    bottom (float): The bottom latitude of the search area.
    left (float): The left longitude of the search area.
    right (float): The right longitude of the search area.

    Returns:
    ImagesOut: A collection of ImageOut objects containing image paths, names,
    and bounding box details.
    """
    images = []

    data_path = "./mock_maps"
    input_files = np.array(sorted(glob.glob(data_path + "/*.tif")))

    for file in input_files:
        bounding_box = get_bounding_box_from_file(file)
        last_slash = file.rfind("/") + 1  # Occurrence of the last slash + 1
        last_dot = file.rfind(".")  # Occurrence of the last dot
        filename = file[last_slash:last_dot]  # Image name
        bounding_box_out = BoundingBox(
            left=bounding_box["left"],
            bottom=bounding_box["bottom"],
            right=bounding_box["right"],
            top=bounding_box["top"],
        )
        images.append(ImageOut(path=file, name=filename, bounding_box=bounding_box_out))

    return ImagesOut(images=images)
