from dotenv import load_dotenv

# Load .env first so JWT_SECRET_KEY (and other keys) are available
# everywhere, regardless of import order.
load_dotenv()

from fastapi import FastAPI

from app.api.router import router

from app.core.config import (
    APP_NAME,
    API_VERSION
)

from app.middleware.cors import (
    add_cors
)

from app.middleware.request_logger import (
    log_requests
)

from app.middleware.exception_handler import (
    global_exception_handler
)


app = FastAPI(

    title=APP_NAME,

    version=API_VERSION,

    docs_url="/docs",

    redoc_url="/redoc"
)


# ==========================
# CORS
# ==========================

add_cors(
    app
)


# ==========================
# REQUEST LOGGER
# ==========================

app.middleware(
    "http"
)(
    log_requests
)


# ==========================
# GLOBAL EXCEPTION HANDLER
# ==========================

app.add_exception_handler(

    Exception,

    global_exception_handler
)


# ==========================
# ROUTES
# ==========================

app.include_router(
    router
)


# ==========================
# API STATUS (JSON)
# ==========================

@app.get(
    "/api"
)
def api_status():

    return {

        "success": True,

        "message": "MyMED API Running",

        "data": {

            "version": API_VERSION
        },

        "error": None
    }


# ==========================
# WEB UI  (served at "/")  -- self-contained frontend in app/web/
# To remove the UI entirely: delete the statements below and app/web/.
# ==========================

from fastapi.responses import FileResponse
from pathlib import Path

@app.get("/")
def serve_ui():
    # no-store: always serve the latest index.html so browsers never show a
    # stale cached copy of the UI after it is updated.
    return FileResponse(
        Path(__file__).parent / "web" / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )