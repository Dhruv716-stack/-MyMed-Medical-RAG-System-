from memory.database import SessionLocal
from memory.memory_models import ChatMessage


# ==========================================
# TEMPORARY LOCAL MEMORY CONFIG
# ==========================================

DEFAULT_USER = "local_user"

DEFAULT_SESSION = "chat_1"


# ==========================================
# SAVE MESSAGE
# ==========================================

def save_message(
    role,
    message
):

    db = SessionLocal()

    db.add(

        ChatMessage(

            user_id=DEFAULT_USER,

            session_id=DEFAULT_SESSION,

            role=role,

            message=message
        )
    )

    db.commit()

    db.close()


# ==========================================
# GET RECENT HISTORY
# ==========================================

def get_recent_history(
    limit=10
):

    db = SessionLocal()

    rows = (

        db.query(ChatMessage)

        .filter(

            ChatMessage.user_id == DEFAULT_USER,

            ChatMessage.session_id == DEFAULT_SESSION
        )

        .order_by(
            ChatMessage.id.desc()
        )

        .limit(limit)

        .all()
    )

    db.close()

    rows.reverse()

    return "\n".join(

        f"{r.role}: {r.message}"

        for r in rows
    )