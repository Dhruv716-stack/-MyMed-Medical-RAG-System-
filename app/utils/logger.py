import logging

from pathlib import Path


LOG_DIR = "logs"

Path(
    LOG_DIR
).mkdir(
    exist_ok=True
)


logging.basicConfig(

    level=logging.INFO,

    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),

    handlers=[

        logging.FileHandler(
            f"{LOG_DIR}/app.log"
        ),

        logging.StreamHandler()
    ]
)

logger = logging.getLogger(
    "mymed"
)