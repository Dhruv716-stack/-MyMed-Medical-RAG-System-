from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memory.memory_models import Base


# ==========================================================
# DATABASE CONFIGURATION
# ==========================================================

DATABASE_URL = "sqlite:///chat_memory.db"


# ==========================================================
# ENGINE
# ==========================================================

engine = create_engine(

    DATABASE_URL,

    echo=False
)


# ==========================================================
# SESSION FACTORY
# ==========================================================

SessionLocal = sessionmaker(

    autocommit=False,

    autoflush=False,

    bind=engine
)


# ==========================================================
# CREATE TABLES
# ==========================================================

Base.metadata.create_all(
    bind=engine
)