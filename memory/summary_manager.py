from memory.database import SessionLocal
from memory.memory_models import ConversationSummary


DEFAULT_USER = "local_user"
DEFAULT_SESSION = "chat_1"


def get_summary(
    user_id=DEFAULT_USER,
    session_id=DEFAULT_SESSION
):

    db = SessionLocal()

    row = (

        db.query(
            ConversationSummary
        )

        .filter(
            ConversationSummary.user_id == user_id
        )

        .filter(
            ConversationSummary.session_id == session_id
        )

        .first()
    )

    db.close()

    if row:

        return row.summary

    return ""


def save_summary(
    summary: str,
    user_id=DEFAULT_USER,
    session_id=DEFAULT_SESSION
):

    db = SessionLocal()

    row = (

        db.query(
            ConversationSummary
        )

        .filter(
            ConversationSummary.user_id == user_id
        )

        .filter(
            ConversationSummary.session_id == session_id
        )

        .first()
    )

    if row:

        row.summary = summary

    else:

        row = ConversationSummary(

            user_id=user_id,

            session_id=session_id,

            summary=summary
        )

        db.add(row)

    db.commit()

    db.close()