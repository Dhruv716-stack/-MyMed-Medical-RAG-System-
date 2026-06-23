from pydantic import BaseModel, Field


class ChatRequest(BaseModel):

    query: str

    file_path: str | None = None


class Citation(BaseModel):

    source: str

    page: int


class ChatData(BaseModel):

    answer: str

    citations: list[Citation] = Field(
        default_factory=list
    )

    latency: float