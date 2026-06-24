from fastapi import Request

from fastapi.responses import (
    JSONResponse
)

from utils.logger import logger


async def global_exception_handler(

        request: Request,

        exc: Exception

):

    logger.exception(

        str(exc)
    )

    return JSONResponse(

        status_code=500,

        content={

            "success": False,

            "message": "Internal Server Error",

            "data": None,

            "error": {

                "details": str(exc)
            }
        }
    )