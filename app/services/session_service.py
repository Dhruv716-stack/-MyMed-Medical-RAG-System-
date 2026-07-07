"""
Session logic: create a new chat session, list a user's sessions,
and confirm a session belongs to a given user.

A "session" is one chat conversation. One user can have many.
"""

import uuid

from memory.database import SessionLocal

from memory.memory_models import (
    ChatSession,
    ChatMessage,
    ConversationSummary,
)


def _generate_session_id() -> str:
    return "s_" + uuid.uuid4().hex


def create_session(user_id: str, title: str = "New Chat") -> dict:
    """Make a fresh chat session owned by this user."""

    db = SessionLocal()

    try:
        session = ChatSession(
            session_id=_generate_session_id(),
            user_id=user_id,
            title=title,
        )

        db.add(session)
        db.commit()

        return {
            "session_id": session.session_id,
            "title": session.title,
        }

    finally:
        db.close()


# Titles considered "not yet named" — safe to auto-rename from the first
# message. Anything else means the user (or a previous auto-name) already set
# a real title, so we leave it alone.
_DEFAULT_TITLES = {"new chat", ""}


def _make_title(text: str, max_len: int = 48) -> str:
    """Turn the first user message into a short, clean sidebar title."""
    t = " ".join((text or "").split())          # collapse whitespace/newlines
    if len(t) > max_len:
        t = t[:max_len].rstrip() + "…"
    return t or "New Chat"


def rename_session_if_default(session_id: str, user_id: str, first_message: str) -> None:
    """
    Auto-name a chat after its first user message — but ONLY if it still has a
    default title ("New Chat"). Scoped to this user's own session, and never
    overwrites a title that is already meaningful. Best-effort: any failure is
    swallowed so it can never break the chat flow.
    """
    try:
        db = SessionLocal()
        try:
            session = (
                db.query(ChatSession)
                .filter(
                    ChatSession.session_id == session_id,
                    ChatSession.user_id == user_id,
                )
                .first()
            )

            if session is None:
                return

            current = (session.title or "").strip().lower()
            if current not in _DEFAULT_TITLES:
                return  # already named — leave it

            session.title = _make_title(first_message)
            db.commit()
        finally:
            db.close()
    except Exception:
        # Naming is cosmetic; never let it affect the answer.
        pass


def list_sessions(user_id: str) -> list[dict]:
    """All of this user's chat sessions, newest first."""

    db = SessionLocal()

    try:
        rows = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.created_at.desc())
            .all()
        )

        return [
            {
                "session_id": r.session_id,
                "title": r.title,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    finally:
        db.close()


def session_belongs_to_user(session_id: str, user_id: str) -> bool:
    """Check a session is owned by this user (used to block cross-user access)."""

    db = SessionLocal()

    try:
        row = (
            db.query(ChatSession)
            .filter(
                ChatSession.session_id == session_id,
                ChatSession.user_id == user_id,
            )
            .first()
        )
        return row is not None

    finally:
        db.close()


def delete_session(session_id: str, user_id: str) -> bool:
    """
    Delete one chat session and everything tied to it, but ONLY if it belongs
    to this user. Returns True if a session was deleted, False if it did not
    exist / was not owned by the user.

    What gets removed (all keyed by session_id + user_id, so we never touch
    another chat or another user's data):
      - the session row              (sessions table)
      - all its messages             (chat_messages table)
      - its conversation summary      (conversation_summaries table)

    Note: this does NOT delete the uploaded-file rows or the physical files on
    disk, and it does NOT touch the Qdrant vector store — that keeps the RAG
    pipeline completely unaffected. Only the chat record is removed.
    """

    db = SessionLocal()

    try:
        session = (
            db.query(ChatSession)
            .filter(
                ChatSession.session_id == session_id,
                ChatSession.user_id == user_id,
            )
            .first()
        )

        # Not found, or not owned by this user -> nothing to delete.
        if session is None:
            return False

        # Delete child rows first (scoped to this user + session only).
        db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == user_id,
        ).delete(synchronize_session=False)

        db.query(ConversationSummary).filter(
            ConversationSummary.session_id == session_id,
            ConversationSummary.user_id == user_id,
        ).delete(synchronize_session=False)

        db.delete(session)
        db.commit()
        return True

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()
