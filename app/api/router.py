from fastapi import APIRouter

from app.api.auth import router as auth_router

from app.api.session import router as session_router

from app.api.chat import router as chat_router

from app.api.upload import router as upload_router

from app.api.history import router as history_router

from app.api.summary import router as summary_router

from app.api.health import router as health_router

from app.api.documents import router as documents_router

from app.api.search import router as search_router
router = APIRouter()

router.include_router(

    auth_router
)

router.include_router(

    session_router
)

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

router.include_router(
    documents_router
)

router.include_router(
    search_router
)