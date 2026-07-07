import shutil
import uuid

from pathlib import Path

from app.core.singleton import get_pipeline

from memory.upload_manager import save_uploaded_file, get_uploaded_files


UPLOAD_DIR = "uploads"

Path(
    UPLOAD_DIR
).mkdir(
    parents=True,
    exist_ok=True
)


def save_file(
    file,
    user_id: str,
    session_id: str,
):
    """
    Save uploaded file.

    Flow:
    Save file
    -> Index into Qdrant
    -> Register upload in DB
    -> Set active document
    -> Return upload metadata
    """

    # ----------------------------------
    # Validation
    # ----------------------------------

    if not user_id:

        raise ValueError(
            "user_id is required"
        )

    if not session_id:

        raise ValueError(
            "session_id is required"
        )

    # ----------------------------------
    # User-specific upload directory
    # ----------------------------------

    user_upload_dir = (

        Path(UPLOAD_DIR)

        / user_id
    )

    user_upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------
    # Unique filename
    # ----------------------------------

    unique_name = (

        f"{uuid.uuid4()}_"

        f"{file.filename}"
    )

    save_path = (

        user_upload_dir

        / unique_name
    )

    # ----------------------------------
    # Save file
    # ----------------------------------

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

    # ----------------------------------
    # Index document
    # ----------------------------------

    pipeline = get_pipeline()

    pipeline.ingest(

        file_path=saved_path,

        force_reindex=False,

        user_id=user_id,

        session_id=session_id,

        source_type="user_upload"
    )

    # ----------------------------------
    # Save upload metadata
    # ----------------------------------

    save_uploaded_file(

        source_path=saved_path,

        original_filename=file.filename,

        user_id=user_id,

        session_id=session_id
    )

    # ----------------------------------
    # Temporary until Session table
    # manages active documents
    # ----------------------------------

    pipeline.active_file = saved_path

    return {

        "filename": file.filename,

        "document_id": unique_name,

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
