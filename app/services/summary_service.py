from memory.summary_manager import (
    get_summary
)


def get_chat_summary(
    user_id: str = "default_user",
    session_id: str = "default_session",
):

    summary = get_summary(
        user_id=user_id,
        session_id=session_id,
    )

    return {

        "summary": summary
    }