import hashlib
import json
from pathlib import Path


CACHE_FILE = "vectorstore/index_cache.json"


def get_file_hash(
    file_path: str
):

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as f:

        while chunk := f.read(8192):

            sha256.update(chunk)

    return sha256.hexdigest()


def load_cache():

    path = Path(CACHE_FILE)

    if not path.exists():

        return {}

    with open(path, "r") as f:

        return json.load(f)


def save_cache(data):

    with open(
        CACHE_FILE,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )