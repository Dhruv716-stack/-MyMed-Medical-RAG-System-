from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.common_schema import APIResponse

from app.schemas.session_schema import CreateSessionRequest

from app.services.session_service import (
    create_session,
    list_sessions,
    delete_session,
)

from app.core.deps import get_current_user

from fastapi import status


router = APIRouter(
    prefix="/sessions",
    tags=["Sessions"],
)


@router.post("", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
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


@router.get("", response_model=APIResponse)
def get_sessions(
    user_id: str = Depends(get_current_user),
):

    return APIResponse(
        success=True,
        message="Sessions fetched.",
        data={"sessions": list_sessions(user_id)},
    )


@router.delete("/{session_id}")
def remove_session(
    session_id: str,
    user_id: str = Depends(get_current_user),
):
    """
    Delete one of the current user's chat sessions (and its messages +
    summary). 404 if it does not exist or is not owned by this user, so
    a user can never delete someone else's chat.
    """

    deleted = delete_session(session_id=session_id, user_id=user_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )

    return APIResponse(
        success=True,
        message="Session deleted.",
        data={"session_id": session_id},
    )
