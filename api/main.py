import os
from typing import Any

import django
from dao.exceptions import FSException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
from starlette.requests import Request

from api.config import settings

# Django settings must be loaded before the FastAPI app
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dao.core.settings")

django.setup(set_prefix=False)

from api.routes.main_router import router


app: Any = FastAPI(
    title="fire API",
    version=settings.API_VERSION,
    root_path=settings.ROOT_PATH,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(FSException)
def fs_exception_handler(request: Request, cmd_exception: FSException):
    return JSONResponse(
        status_code=cmd_exception.status_code,
        content=cmd_exception.detail,
    )


@app.get("/", tags=["Health Check"])
def health_check():
    return {"message": f"Hi, it works fine! (v{settings.API_VERSION})"}


app.include_router(router, prefix=settings.API_URL_PREFIX)

# For deployment via AWS Lambda

handler = Mangum(app)