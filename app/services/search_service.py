from memory.memory_manager import search_messages


def search_history(
    query: str,
    user_id: str = "default_user",
    limit: int = 50,
):
    """
    Search across all of a user's chat messages. Thin wrapper over the
    read-only memory_manager.search_messages(); scoped to user_id by the
    layer below, so results are always the caller's own messages.
    """

    results = search_messages(
        query=query,
        user_id=user_id,
        limit=limit,
    )

    return {
        "query": query,
        "results": results,
    }
