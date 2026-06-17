from sqlalchemy import create_engine

from sqlalchemy.orm import sessionmaker

from memory.memory_models import Base

engine = create_engine(

    "sqlite:///chat_memory.db",

    echo=False
)

SessionLocal = sessionmaker(
    bind=engine
)

Base.metadata.create_all(
    bind=engine
)