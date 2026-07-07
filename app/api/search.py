from fastapi import APIRouter, Depends

from app.schemas.common_schema import APIResponse

from app.services.search_service import search_history

from app.core.deps import get_current_user


router = APIRouter(

    prefix="/search",

    tags=["Search"]
)


@router.get("")
def search(
    q: str = "",
    limit: int = 50,
    user_id: str = Depends(get_current_user),
):
    """
    Global search across the current user's chat history (all sessions).
    Read-only; scoped to the authenticated user so results are always
    their own messages.
    """

    return APIResponse(
        success=True,
        message="Search completed.",
        data=search_history(
            query=q,
            user_id=user_id,
            limit=limit,
        ),
    )
