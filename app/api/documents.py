from fastapi import APIRouter, Depends

from app.schemas.common_schema import APIResponse

from app.services.upload_service import list_documents

from app.core.deps import get_current_user, ensure_session_owner


router = APIRouter(

    prefix="/documents",

    tags=["Documents"]
)


@router.get("")
def documents(
    session_id: str = "default_session",
    user_id: str = Depends(get_current_user),
):
    """
    List the documents the current user has uploaded in this session.
    Read-only; scoped to the authenticated user + session so a user can
    only ever see their own files.
    """

    ensure_session_owner(session_id, user_id)

    return APIResponse(
        success=True,
        message="Documents fetched.",
        data=list_documents(
            user_id=user_id,
            session_id=session_id,
        ),
    )
