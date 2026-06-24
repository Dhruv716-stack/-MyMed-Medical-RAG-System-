from pydantic import BaseModel, Field


class ChatRequest(BaseModel):

    query: str = Field(
        description="User query"
    )

class Citation(BaseModel):

    source: str = Field(
        description="Source of the citation"
    )

    page: int


class ChatData(BaseModel):

    answer: str

    citations: list[Citation] = Field(
        default_factory=list
    )

    latency: float