import shutil

from pathlib import Path

from app.core.singleton import get_pipeline

from memory.upload_manager import save_uploaded_file


UPLOAD_DIR = "uploads"

Path(
    UPLOAD_DIR
).mkdir(
    exist_ok=True
)


def save_file(
    file,
    user_id: str = "default_user",
    session_id: str = "default_session",
):

    save_path = Path(
        UPLOAD_DIR
    ) / file.filename

    with open(
        save_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    saved_path = str(
        save_path
    )

    # --------------------------
    # Index document as this
    # user's private upload, so
    # retrieval can be restricted
    # to their own files only.
    # --------------------------

    pipeline = get_pipeline()

    pipeline.ingest(

        file_path=saved_path,

        force_reindex=False,

        user_id=user_id,

        session_id=session_id,

        source_type="user_upload",
    )

    # --------------------------
    # Record the upload so the
    # user sees it again when the
    # session is reopened (same as
    # the pipeline's own run()).
    # --------------------------

    save_uploaded_file(
        source_path=saved_path,
        original_filename=file.filename,
        user_id=user_id,
        session_id=session_id,
    )

    # --------------------------
    # Make uploaded document
    # active for future chats
    # --------------------------

    pipeline.active_file = saved_path

    return {

        "filename": file.filename,

        "path": saved_path,

        "indexed": True
    }
