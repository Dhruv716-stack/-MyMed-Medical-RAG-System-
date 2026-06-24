from fastapi import APIRouter, Depends

from app.schemas.chat_schema import ChatRequest

from app.schemas.common_schema import APIResponse

from app.services.chat_services import chat

from app.core.deps import get_current_user


router = APIRouter(

    prefix="/chat",

    tags=["Chat"]
)


@router.post(
    ""
)

def chat_endpoint(

        request: ChatRequest,

        user_id: str = Depends(get_current_user),

):

    result = chat(

        query=request.query,

        user_id=user_id,

        session_id=request.session_id,
    )

    return APIResponse(

        success=True,

        message="Response generated successfully.",

        data=result
    )