from memory.memory_manager import get_recent_history
from memory.summary_manager import get_summary

from rag_pipeline.config import (
    ENABLE_SUMMARY_MEMORY,
    RECENT_MEMORY_LIMIT
)

DEFAULT_USER = "local_user"
DEFAULT_SESSION = "chat_1"


def build_memory_context(
    user_id: str=DEFAULT_USER,
    session_id: str=DEFAULT_SESSION,
):

    recent_history = get_recent_history(

        limit=RECENT_MEMORY_LIMIT,

        user_id=user_id,

        session_id=session_id
    )

    if not ENABLE_SUMMARY_MEMORY:

        return recent_history

    summary = get_summary(

        user_id=user_id,

        session_id=session_id
    )

    return f"""
Conversation Summary:

{summary}


Recent Conversation:

{recent_history}
"""