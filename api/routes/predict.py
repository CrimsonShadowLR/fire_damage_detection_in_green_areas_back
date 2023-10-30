from dao.controller.predict import PredictResource
from fastapi import APIRouter, Body, status

router = APIRouter()


@router.post("/", status_code=status.HTTP_200_OK)
def search_images(
    filepath: str = Body(...),
):
    """
    Endpoint searchs and returns the image for the given filepath
    """
    return PredictResource(filepath=filepath)
