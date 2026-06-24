from fastapi import APIRouter, Depends

from app.schemas.common_schema import APIResponse

from app.services.history_service import (
    get_history
)

from app.core.deps import get_current_user

router = APIRouter(

    prefix="/history",

    tags=["History"]
)


@router.get(
    ""
)
def history(
    session_id: str = "default_session",
    limit: int = 10,
    user_id: str = Depends(get_current_user),
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