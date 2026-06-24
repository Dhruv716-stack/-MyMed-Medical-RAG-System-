from memory.memory_manager import (
    get_recent_history
)


def get_history(
    user_id: str = "default_user",
    session_id: str = "default_session",
    limit: int = 10,
):

    history = get_recent_history(
        limit=limit,
        user_id=user_id,
        session_id=session_id,
    )

    return {

        "history": history
    }