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
def summary(
    user_id: str = "default_user",
    session_id: str = "default_session",
):

    return APIResponse(

        success=True,

        message="Summary fetched.",

        data=get_chat_summary(
            user_id=user_id,
            session_id=session_id,
        )
    )