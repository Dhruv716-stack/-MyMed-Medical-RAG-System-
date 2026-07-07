from memory.database import SessionLocal
from memory.memory_models import ConversationSummary
DEFAULT_USER = "local_user"
DEFAULT_SESSION = "chat_1"

def get_summary(
    user_id: str=DEFAULT_USER,
    session_id: str=DEFAULT_SESSION
) -> str:

    db = SessionLocal()

    try:

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

        return (

            row.summary

            if row

            else ""
        )

    finally:

        db.close()


def save_summary(
    summary: str,
    user_id: str=DEFAULT_USER,
    session_id: str=DEFAULT_SESSION
) -> None:

    db = SessionLocal()

    try:

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

            db.add(
                row
            )

        db.commit()

    finally:

        db.close()