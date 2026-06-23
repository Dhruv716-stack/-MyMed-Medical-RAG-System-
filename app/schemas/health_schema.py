from pydantic import BaseModel


class HealthResponse(BaseModel):

    status: str

    qdrant: bool

    memory_db: bool

    ollama: bool