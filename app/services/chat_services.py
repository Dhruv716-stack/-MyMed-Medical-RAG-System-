import time

from app.core.singleton import get_pipeline

pipeline=get_pipeline()
def chat(
    query: str,
):
    """
    Main chat service.

    - Handles Swagger's default 'string' value.
    - Falls back to normal chat/RAG routing when no file is supplied.
    - Measures latency.
    - Prevents API crashes from bubbling up.
    """

    # ----------------------------------
    # Normalize file path
    # ----------------------------------

    if file_path in (
        None,
        "",
        "string",
        "null",
        "None"
    ):
        file_path = None

    start = time.time()

    try:

        answer = pipeline.run(

            query=query,
        )

        latency = round(
            time.time() - start,
            2
        )

        return {

            "answer": answer,

            "citations": [],

            "latency": latency
        }

    except Exception as e:

        latency = round(
            time.time() - start,
            2
        )

        return {

            "answer": None,

            "citations": [],

            "latency": latency,

            "error": str(e)
        }