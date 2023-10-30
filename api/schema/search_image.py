from typing import List

from api.schema.adapters import CustomBaseModel
from pydantic import BaseModel


class BoundingBox(BaseModel):
    left: float
    bottom: float
    right: float
    top: float


class ImageOut(CustomBaseModel):
    path: str
    name: str
    bounding_box: BoundingBox


class ImagesOut(CustomBaseModel):
    images: List[ImageOut]
