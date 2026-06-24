import time

from app.core.singleton import get_pipeline

from fastapi import HTTPException


def chat(
    query: str,
    user_id: str,
    session_id: str
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

      raise HTTPException(
        status_code=500,
        detail=str(e)
      )
