from typing import List

from api.schema.adapters import FSBaseModel
from pydantic import BaseModel


class BoundingBox(BaseModel):
    left: float
    bottom: float
    right: float
    top: float


class ImageOut(FSBaseModel):
    path: str
    name: str
    bounding_box: BoundingBox


class ImagesOut(FSBaseModel):
    images: List[ImageOut]
