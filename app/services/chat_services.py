import time

from app.core.singleton import get_pipeline


def chat(
    query: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
):
    """
    Main chat service.

    - Routes the query through the existing MedicalRAGPipeline, which
      internally decides general-chat vs medical-RAG and saves memory.
    - Passes user_id / session_id so each user's memory and uploads
      stay isolated (multi-tenant storage).
    - Measures latency.
    - Prevents API crashes from bubbling up to the client.
    """

    pipeline = get_pipeline()

    start = time.time()

    try:

        answer = pipeline.run(
            query=query,
            user_id=user_id,
            session_id=session_id,
        )

        latency = round(
            time.time() - start,
            2
        )

        return {
            "answer": answer,
            "citations": [],
            "latency": latency,
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
            "error": str(e),
        }
