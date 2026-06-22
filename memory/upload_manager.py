import shutil
from pathlib import Path

from memory.database import SessionLocal
from memory.memory_models import UploadedFile


# ==========================================
# STORAGE PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_BOOKS_DIR = DATA_DIR / "default_books"
USER_UPLOADS_DIR = DATA_DIR / "user_uploads"

DEFAULT_BOOKS_DIR.mkdir(parents=True, exist_ok=True)
USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# SAVE UPLOADED FILE TO DISK + DB
# ==========================================

def save_uploaded_file(
    source_path: str,
    original_filename: str,
    user_id: str,
    session_id: str
) -> str:

    session_dir = USER_UPLOADS_DIR / f"{user_id}_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    dest_path = session_dir / original_filename

    shutil.copyfile(source_path, dest_path)

    db = SessionLocal()

    db.add(
        UploadedFile(
            user_id=user_id,
            session_id=session_id,
            original_filename=original_filename,
            file_path=str(dest_path)
        )
    )

    db.commit()
    db.close()

    return str(dest_path)


# ==========================================
# LIST A USER'S UPLOADED FILES FOR A SESSION
# ==========================================

def get_uploaded_files(
    user_id: str,
    session_id: str
):

    db = SessionLocal()

    rows = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.user_id == user_id,
            UploadedFile.session_id == session_id
        )
        .order_by(UploadedFile.uploaded_at.desc())
        .all()
    )

    db.close()

    return [
        {
            "original_filename": r.original_filename,
            "file_path": r.file_path,
            "uploaded_at": r.uploaded_at
        }
        for r in rows
    ]
