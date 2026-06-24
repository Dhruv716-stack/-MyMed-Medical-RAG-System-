from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
)

from sqlalchemy.orm import declarative_base

from datetime import datetime,timezone


Base = declarative_base()


# ==========================================================
# USER (authentication)
# ==========================================================

class User(Base):

    __tablename__ = "users"

    # Our own generated id (e.g. "u_<uuid>") used everywhere in the
    # pipeline as user_id. This is the trusted identity.
    user_id = Column(
        String,
        primary_key=True
    )

    email = Column(
        String,
        unique=True,
        nullable=False,
        index=True
    )

    # Optional display name the user chooses. May be empty.
    username = Column(
        String,
        nullable=True
    )

    # bcrypt hash of the password — never the raw password.
    hashed_password = Column(
        String,
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )


class ChatSession(Base):

    __tablename__ = "sessions"

    # Our generated id (e.g. "s_<uuid>") used as session_id everywhere.
    session_id = Column(
        String,
        primary_key=True
    )

    # Which user owns this chat session.
    user_id = Column(
        String,
        nullable=False,
        index=True
    )

    # A short label for the chat (UI shows this in the sidebar).
    title = Column(
        String,
        nullable=False,
        default="New Chat"
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )


class ChatMessage(Base):

    __tablename__ = "chat_messages"

    id = Column(
        Integer,
        primary_key=True
    )

    user_id = Column(
        String,
        nullable=False
    )

    session_id = Column(
        String,
        nullable=False
    )

    role = Column(
        String,
        nullable=False
    )

    message = Column(
        Text,
        nullable=False
    )

    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )
    
class ConversationSummary(Base):

    __tablename__ = "conversation_summaries"

    id = Column(
        Integer,
        primary_key=True
    )

    user_id = Column(
        String,
        nullable=False
    )

    session_id = Column(
        String,
        nullable=False
    )

    summary = Column(
        Text,
        nullable=False,
        default=""
    )

    updated_at = Column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc)
    )


class UploadedFile(Base):

    __tablename__ = "uploaded_files"

    id = Column(
        Integer,
        primary_key=True
    )

    user_id = Column(
        String,
        nullable=False
    )

    session_id = Column(
        String,
        nullable=False
    )

    original_filename = Column(
        String,
        nullable=False
    )

    file_path = Column(
        String,
        nullable=False
    )

    uploaded_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )