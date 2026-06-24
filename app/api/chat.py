from fastapi import APIRouter

from app.schemas.chat_schema import ChatRequest

from app.schemas.common_schema import APIResponse

from app.services.chat_services import chat


router = APIRouter(

    prefix="/chat",

    tags=["Chat"]
)


@router.post(
    ""
)

def chat_endpoint(

        request: ChatRequest

):

    result = chat(

        query=request.query,

        user_id=request.user_id,

        session_id=request.session_id,
    )

    return APIResponse(

        success=True,

        message="Response generated successfully.",

        data=result
    )