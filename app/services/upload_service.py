import shutil

from pathlib import Path

from app.core.singleton import get_pipeline

from memory.upload_manager import save_uploaded_file, get_uploaded_files


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


def list_documents(
    user_id: str = "default_user",
    session_id: str = "default_session",
):
    """
    Return the documents this user has uploaded in this session, using the
    records the upload flow already saves (memory.upload_manager). Read-only:
    it does not touch disk, the pipeline, or the vector store.

    Shape: {"documents": [{"filename", "uploaded_at"}, ...]} — internal disk
    paths are intentionally not exposed to the client.
    """

    rows = get_uploaded_files(user_id=user_id, session_id=session_id)

    documents = [
        {
            "filename": r.get("original_filename"),
            "uploaded_at": (
                r["uploaded_at"].isoformat()
                if r.get("uploaded_at") is not None
                and hasattr(r["uploaded_at"], "isoformat")
                else r.get("uploaded_at")
            ),
        }
        for r in rows
    ]

    return {"documents": documents}
