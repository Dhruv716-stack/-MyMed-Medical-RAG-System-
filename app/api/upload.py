from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form
)

from app.schemas.common_schema import APIResponse

from app.services.upload_service import (
    save_file
)

router = APIRouter(

    prefix="/upload",

    tags=["Upload"]
)


@router.post("")
def upload_file(

    file: UploadFile = File(...),

    user_id: str = Form("default_user"),

    session_id: str = Form("default_session"),

):

    result = save_file(
        file,
        user_id=user_id,
        session_id=session_id,
    )

    return APIResponse(

        success=True,

        message="File uploaded and indexed successfully.",

        data=result
    )