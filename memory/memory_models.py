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