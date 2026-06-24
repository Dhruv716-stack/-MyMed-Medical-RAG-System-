from fastapi import APIRouter

from api.chat import router as chat_router

from api.upload import router as upload_router
from api.history import router as history_router
from api.summary import router as summary_router
from api.health import router as health_router

router = APIRouter()

router.include_router(

    chat_router
)
router.include_router(
    upload_router
)

router.include_router(
    history_router
)

router.include_router(
    summary_router
)

router.include_router(
    health_router
)