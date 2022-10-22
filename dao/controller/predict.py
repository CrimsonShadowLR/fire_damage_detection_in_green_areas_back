import logging
import os
import time
from copy import deepcopy

import numpy as np
import rasterio
import torch
from dao.controller.image_utils import define_mask
from dao.controller.json_utils import build_response
from dao.controller.model_utils import load_model, predict
from dao.controller.search_image import get_bounding_box
from rasterio.io import MemoryFile

logger = logging.getLogger("root")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
UPLOAD_DIRECTORY = "static/"
MODEL_PATH = "trained_models/model_20_percent_burn_Unet11_400epochs_sentinel"


def convert_mask_to_png(filename, raster, metadata, colours=[255, 0, 255]):
    """Transforma una máscara en una imagen png para su visualización en plataformas web.

    :param filename: ruta de la máscara originalmente generada como TIF
    :type filename: str


    :param raster: matriz bidimensional con los valores de la máscara
    :type raster: np.ndarray


    :param metadata: diccionario con los metadatos de la máscara
    :type metadata: dict

    :param colours: colores para colorear la máscara
    :type colours: tuple[int, int, int]

    :param level: índice del nivel
    :type level: str

    :rtype: str

    """
    new_metadata = metadata
    new_metadata["count"] = 3
    new_metadata["driver"] = "PNG"
    new_metadata["dtype"] = "uint8"

    png_filename = filename + ".png"

    new_raster = np.zeros(shape=[3, new_metadata["height"], new_metadata["width"]])
    new_raster[0] = raster * colours[0]
    new_raster[1] = raster * colours[1]
    new_raster[2] = raster * colours[2]
    new_raster = new_raster.astype("uint8")

    with rasterio.open(UPLOAD_DIRECTORY + png_filename, "w", **new_metadata) as dst:
        dst.write(new_raster)

    return png_filename


def convert_raster_to_png(filename, raster, metadata):
    """Transforma una imagen satelital en una imagen png para su visualización en
        plataformas web.

    :param filename: ruta del raster original
    :type filename: str


    :param raster: matriz bidimensional con los valores del raster
    :type raster: np.ndarray


    :param metadata: diccionario con los metadatos del raster
    :type metadata: dict

    """
    new_metadata = metadata
    new_metadata["count"] = 3
    new_metadata["driver"] = "PNG"
    new_metadata["dtype"] = "uint8"

    png_filename = filename + "_imagen.png"
    raster = raster[:3]
    new_raster = (raster / 3512 * 255).astype("uint8")

    with rasterio.open(UPLOAD_DIRECTORY + png_filename, "w", **new_metadata) as dst:
        dst.write(new_raster)

    return png_filename


def PredictResource(filepath: str):
    start = time.time()
    logger.debug("Receiving image...")

    last_slash = filepath.rfind("/") + 1  # Ocurrencia de la última diagonal + 1
    last_dot = filepath.rfind(".")  # Ocurrencia del último punto
    filename = filepath[last_slash:last_dot]  # Nombre de la imagen

    logger.debug("Filename: {}".format(filename))

    # Leyendo imagen
    file = open(filepath, "rb")
    data = file.read()
    reading = time.time()
    logger.debug(
        "Image Received. Elapsed time: {}s".format(str(round(reading - start, 2)))
    )
    logger.debug("Opening image...")

    memfile = MemoryFile(data)
    dataset = memfile.open()
    meta = dataset.profile
    img_npy = dataset.read()

    opening = time.time()
    logger.debug(
        "Image opened. Elapsed time: {}s".format(str(round(opening - reading, 2)))
    )

    logger.debug("Image shape: {}".format(str(img_npy.shape)))

    logger.debug("Generating mask...")

    model = load_model(model_path=MODEL_PATH, input_channels=5)

    mask = predict(model, img_npy)
    mask = define_mask(mask)
    layers_paths = []

    layers_paths.append(convert_raster_to_png(filename, img_npy, meta))

    for idx, layer in enumerate(mask):
        layer_path = convert_mask_to_png(filename, layer, meta)
        layers_paths.append(layer_path)

    predicting = time.time()
    logger.debug(
        "Mask generated! Elapsed time: {}s".format(str(round(predicting - opening, 2)))
    )

    bounding_box = get_bounding_box(memfile.open())

    mask = mask.astype("uint8")
    metadata = deepcopy(meta)
    metadata["driver"] = "GTiff"
    metadata["count"] = mask.shape[0]
    metadata["height"] = mask.shape[1]
    metadata["width"] = mask.shape[2]
    metadata["dtype"] = mask.dtype

    with rasterio.open(
        os.path.join(UPLOAD_DIRECTORY, filename + ".tif"), "w", **metadata
    ) as outds:
        outds.write(mask)

    end = time.time()
    response = build_response(filename, bounding_box, layers_paths)
    logger.debug("Total Elapsed Time: {}s".format(str(round(end - start, 2))))

    return response
