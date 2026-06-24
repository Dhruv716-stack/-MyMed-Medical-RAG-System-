from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    Depends
)

from app.schemas.common_schema import APIResponse

from app.services.upload_service import (
    save_file
)

from app.core.deps import get_current_user

router = APIRouter(

    prefix="/upload",

    tags=["Upload"]
)


@router.post("")
def upload_file(

    file: UploadFile = File(...),

    session_id: str = Form("default_session"),

    user_id: str = Depends(get_current_user),

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