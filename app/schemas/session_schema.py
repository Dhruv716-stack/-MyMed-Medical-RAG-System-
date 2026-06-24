from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):

    title: str = Field(
        default="New Chat",
        min_length=1,
        max_length=100,
        description="Optional label for the chat (shown in the sidebar)"
    )