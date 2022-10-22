import glob

import numpy as np
from api.schema.search_image import BoundingBox, ImageOut, ImagesOut
from pyproj import Transformer
from rasterio.io import MemoryFile


def rect_overlap(l1, r1, l2, r2):
    if l1[0] <= r2[0] or l2[0] <= r1[0]:
        return False

    if l1[1] >= r2[1] or l2[1] >= r1[1]:
        return False

    return True


def get_bounding_box(dataset):
    """Obtiene las coordenadas de una imagen satelital en formato EPSG:4326

    :param dataset: lector de conjunto de datos ráster
    :type dataset: rio.DatasetReader
    """
    # Obtiene el bounding box original

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
    file = open(file, "rb")
    data = file.read()
    memfile = MemoryFile(data)
    dataset = memfile.open()
    return get_bounding_box(dataset)


def SearchImage(top: float, bottom: float, left: float, right: float):
    # area_of_interest = {
    #     "left": left,
    #     "bottom": bottom,
    #     "right": right,
    #     "top": top,
    # }

    images = []

    data_path = "./mock_maps"
    input_files = np.array(sorted(glob.glob(data_path + "/*.tif")))

    for file in input_files:
        bounding_box = get_bounding_box_from_file(file)

        # if rect_overlap(
        #     (bounding_box["top"], bounding_box["right"]),
        #     (bounding_box["bottom"], bounding_box["left"]),
        #     (area_of_interest["top"], area_of_interest["right"]),
        #     (area_of_interest["bottom"], area_of_interest["left"]),
        # ):
        last_slash = file.rfind("/") + 1  # Ocurrencia de la última diagonal + 1
        last_dot = file.rfind(".")  # Ocurrencia del último punto
        filename = file[last_slash:last_dot]  # Nombre de la imagen
        bounding_box_out = BoundingBox(
            left=bounding_box["left"],
            bottom=bounding_box["bottom"],
            right=bounding_box["right"],
            top=bounding_box["top"],
        )
        images.append(ImageOut(path=file, name=filename, bounding_box=bounding_box_out))

    return ImagesOut(images=images)
