import shutil
from pathlib import Path

UPLOAD_DIR = "uploads"

Path(UPLOAD_DIR).mkdir(
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

    return str(save_path)