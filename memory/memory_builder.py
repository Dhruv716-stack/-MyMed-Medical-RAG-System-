from memory.memory_manager import (
    get_recent_history
)

from memory.summary_manager import (
    get_summary
)

from rag_pipeline.config import (
    ENABLE_SUMMARY_MEMORY,
    RECENT_MEMORY_LIMIT
)


def build_memory_context(
    user_id=None,
    session_id=None
):

    kwargs = {}
    if user_id is not None:
        kwargs["user_id"] = user_id
    if session_id is not None:
        kwargs["session_id"] = session_id

    recent_history = get_recent_history(
        limit=RECENT_MEMORY_LIMIT,
        **kwargs
    )

    if not ENABLE_SUMMARY_MEMORY:

        return recent_history

    summary = get_summary(**kwargs)

    return f"""

Conversation Summary:

{summary}


Recent Conversation:

{recent_history}

"""