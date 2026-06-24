from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):

    title: str = Field(
        default="New Chat",
        description="Optional label for the chat (shown in the sidebar)"
    )
