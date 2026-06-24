from fastapi import APIRouter

from app.schemas.common_schema import APIResponse

from app.services.history_service import (
    get_history
)

router = APIRouter(

    prefix="/history",

    tags=["History"]
)


@router.get(
    ""
)
def history(
    user_id: str = "default_user",
    session_id: str = "default_session",
    limit: int = 10,
):

    return APIResponse(

        success=True,

        message="History fetched.",

        data=get_history(
            user_id=user_id,
            session_id=session_id,
            limit=limit,
        )
    )