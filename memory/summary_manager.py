from memory.database import SessionLocal
from memory.memory_models import ConversationSummary


DEFAULT_USER = "local_user"
DEFAULT_SESSION = "chat_1"


def get_summary():

    db = SessionLocal()

    row = (

        db.query(
            ConversationSummary
        )

        .filter(
            ConversationSummary.user_id == DEFAULT_USER
        )

        .filter(
            ConversationSummary.session_id == DEFAULT_SESSION
        )

        .first()
    )

    db.close()

    if row:

        return row.summary

    return ""


def save_summary(
    summary: str
):

    db = SessionLocal()

    row = (

        db.query(
            ConversationSummary
        )

        .filter(
            ConversationSummary.user_id == DEFAULT_USER
        )

        .filter(
            ConversationSummary.session_id == DEFAULT_SESSION
        )

        .first()
    )

    if row:

        row.summary = summary

    else:

        row = ConversationSummary(

            user_id=DEFAULT_USER,

            session_id=DEFAULT_SESSION,

            summary=summary
        )

        db.add(row)

    db.commit()

    db.close()