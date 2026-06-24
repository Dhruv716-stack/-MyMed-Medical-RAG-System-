import shutil

from pathlib import Path

from app.core.singleton import get_pipeline


UPLOAD_DIR = "uploads"

Path(
    UPLOAD_DIR
).mkdir(
    exist_ok=True
)


def save_file(file):

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
    # Index document
    # --------------------------

    pipeline = get_pipeline()

    pipeline.ingest(

        file_path=saved_path,

        force_reindex=False
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