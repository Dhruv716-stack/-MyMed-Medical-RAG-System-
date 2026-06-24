"""
Session logic: create a new chat session, list a user's sessions,
and confirm a session belongs to a given user.

A "session" is one chat conversation. One user can have many.
"""

import uuid

from memory.database import SessionLocal

from memory.memory_models import ChatSession


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
        db.refresh(session)

        return {
            "session_id": session.session_id,
            "title": session.title,
        }

    finally:
        db.close()


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
