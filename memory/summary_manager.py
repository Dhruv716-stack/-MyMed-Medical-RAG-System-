from memory.database import SessionLocal
from memory.memory_models import ConversationSummary


def get_summary(
    user_id: str,
    session_id: str
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
    user_id: str,
    session_id: str
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