from fastapi import APIRouter

from app.schemas.common_schema import APIResponse

from app.services.health_service import (
    health_check
)

router = APIRouter(

    prefix="/health",

    tags=["Health"]
)


@router.get(
    ""
)
def health():

    return APIResponse(

        success=True,

        message="Server healthy.",

        data=health_check()
    )