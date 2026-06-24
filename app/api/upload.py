from fastapi import (
    APIRouter,
    UploadFile,
    File
)

from schemas.common_schema import APIResponse

from services.upload_service import (
    save_file
)

router = APIRouter(

    prefix="/upload",

    tags=["Upload"]
)


@router.post(
    ""
)
def upload_file(

        file: UploadFile = File(...)
):

    path = save_file(
        file
    )

    return APIResponse(

        success=True,

        message="File uploaded successfully.",

        data={

            "filename": file.filename,

            "path": path
        }
    )