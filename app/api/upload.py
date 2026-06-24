from fastapi import (
    APIRouter,
    UploadFile,
    File
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

    file: UploadFile = File(...)

):

    result = save_file(
        file
    )

    return APIResponse(

        success=True,

        message="File uploaded and indexed successfully.",

        data=result
    )