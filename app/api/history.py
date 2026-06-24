from fastapi import APIRouter

from schemas.common_schema import APIResponse

from services.history_service import (
    get_history
)

router = APIRouter(

    prefix="/history",

    tags=["History"]
)


@router.get(
    ""
)
def history():

    return APIResponse(

        success=True,

        message="History fetched.",

        data=get_history()
    )