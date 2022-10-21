from api.routes.predict import router as predict_router
from api.routes.check_file import router as check_file_router
from api.routes.search_image import router as search_image_router
from fastapi import APIRouter

router = APIRouter()

router.include_router(
    predict_router,
    prefix="/predict",
)
router.include_router(
    check_file_router,
    prefix="/checkFiles",
)
router.include_router(
    search_image_router,
    prefix="/searchImages",
)