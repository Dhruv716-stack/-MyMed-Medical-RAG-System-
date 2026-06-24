from pydantic import BaseModel, Field


class ChatRequest(BaseModel):

    query: str = Field(
        description="User query"
    )

    session_id: str = Field(
        default="default_session",
        description="Identifies the chat session; one user can have many sessions"
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