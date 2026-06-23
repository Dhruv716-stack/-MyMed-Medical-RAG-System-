from fastapi import APIRouter

from schemas.chat_schema import (
    ChatRequest
)

from schemas.common_schema import (
    APIResponse
)

from services.chat_services import (
    chat
)


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

        file_path=request.file_path
    )

    return APIResponse(

        success=True,

        message="Response generated successfully.",

        data=result
    )