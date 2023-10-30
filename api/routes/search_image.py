from api.schema.search_image import ImagesOut
from dao.controller.search_image import SearchImage
from fastapi import APIRouter, Body, status

router = APIRouter()


@router.post("/", status_code=status.HTTP_200_OK, response_model=ImagesOut)
def search_images(
    top: float = Body(...),
    bottom: float = Body(...),
    left: float = Body(...),
    right: float = Body(...),
):
    """
    Endpoint searches for image filenames whose metadata contains coordinates
    within the specified range.
    """
    return SearchImage(top=top, bottom=bottom, left=left, right=right)
