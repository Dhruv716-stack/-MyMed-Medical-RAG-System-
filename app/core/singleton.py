from rag_pipeline.complete_pipeline import MedicalRAGPipeline

_pipeline = None


def get_pipeline():

    global _pipeline

    if _pipeline is None:

        print(
            "Loading MedicalRAGPipeline..."
        )

        _pipeline = MedicalRAGPipeline()

    return _pipeline