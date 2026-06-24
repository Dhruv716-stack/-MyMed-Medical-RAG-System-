from fastapi import APIRouter, Depends

from app.schemas.common_schema import APIResponse

from app.schemas.session_schema import CreateSessionRequest

from app.services.session_service import (
    create_session,
    list_sessions,
)

from app.core.deps import get_current_user


router = APIRouter(
    prefix="/sessions",
    tags=["Sessions"],
)


@router.post("")
def new_session(
    request: CreateSessionRequest,
    user_id: str = Depends(get_current_user),
):

    result = create_session(
        user_id=user_id,
        title=request.title,
    )

    return APIResponse(
        success=True,
        message="New chat session created.",
        data=result,
    )


@router.get("")
def get_sessions(
    user_id: str = Depends(get_current_user),
):

    return APIResponse(
        success=True,
        message="Sessions fetched.",
        data={"sessions": list_sessions(user_id)},
    )
