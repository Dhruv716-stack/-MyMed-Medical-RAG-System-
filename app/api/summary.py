from fastapi import APIRouter, Depends

from app.schemas.common_schema import APIResponse

from app.services.summary_service import (
    get_chat_summary
)

from app.core.deps import get_current_user

router = APIRouter(

    prefix="/summary",

    tags=["Summary"]
)


@router.get(
    ""
)
def summary(
    session_id: str = "default_session",
    user_id: str = Depends(get_current_user),
):

    return APIResponse(

        success=True,

        message="Summary fetched.",

        data=get_chat_summary(
            user_id=user_id,
            session_id=session_id,
        )
    )