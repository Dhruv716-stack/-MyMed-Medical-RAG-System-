from fastapi import APIRouter

from app.schemas.common_schema import APIResponse

from app.services.summary_service import (
    get_chat_summary
)

router = APIRouter(

    prefix="/summary",

    tags=["Summary"]
)


@router.get(
    ""
)
def summary():

    return APIResponse(

        success=True,

        message="Summary fetched.",

        data=get_chat_summary()
    )