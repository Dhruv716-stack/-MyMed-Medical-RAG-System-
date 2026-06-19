from memory.database import SessionLocal
from memory.memory_models import ChatMessage

from memory.summary_manager import (
    get_summary,
    save_summary
)

from memory.summarizer import (
    update_summary
)

from rag_pipeline.config import (
    SUMMARY_UPDATE_INTERVAL,
    ENABLE_SUMMARY_MEMORY
)
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
    
def get_message_count():

    db = SessionLocal()

    count = (

        db.query(
            ChatMessage
        )

        .count()
    )

    db.close()

    return count

# ====================================
# AUTO UPDATE SUMMARY
# ====================================
if ENABLE_SUMMARY_MEMORY:

    count = get_message_count()

    if count > 0 and count % SUMMARY_UPDATE_INTERVAL == 0:

        try:

            old_summary = get_summary()

            recent_history = get_recent_history(
                limit=SUMMARY_UPDATE_INTERVAL
            )

            new_summary = update_summary(

                old_summary,

                recent_history
            )

            save_summary(
                new_summary
            )

            print(
                "Conversation summary updated."
            )

        except Exception as e:

            print(
                f"Summary update failed: {e}"
            )