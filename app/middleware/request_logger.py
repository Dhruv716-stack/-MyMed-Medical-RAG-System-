import time

from utils.logger import logger


async def log_requests(
    request,
    call_next
):

    start_time = time.time()

    response = await call_next(
        request
    )

    duration = round(

        time.time() - start_time,

        3
    )

    logger.info(

        f"{request.method} "
        f"{request.url.path} "
        f"{response.status_code} "
        f"{duration}s"
    )

    return response