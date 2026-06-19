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


def build_memory_context():

    recent_history = get_recent_history(
        limit=RECENT_MEMORY_LIMIT
    )

    if not ENABLE_SUMMARY_MEMORY:

        return recent_history

    summary = get_summary()

    return f"""

Conversation Summary:

{summary}


Recent Conversation:

{recent_history}

"""