import time

from core.singleton import pipeline


def chat(

        query: str,

        file_path: str = None

):

    start = time.time()

    answer = pipeline.run(

        query=query,

        file_path=file_path

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