"""
Auth business logic: register a new user, and log an existing one in.

Uses the SAME SQLite database as the rest of the app (memory/database.py),
and the User table added in memory/memory_models.py.
"""

from memory.database import SessionLocal

from memory.memory_models import User

from app.core.security import (
    hash_password,
    verify_password,
    generate_user_id,
    create_access_token,
)


# ----------------------------------------------------------
# REGISTER
# ----------------------------------------------------------

def register(email: str, password: str, username: str = None) -> dict:
    """
    Create a new account.

    Returns {"ok": True, "user_id": ...} on success,
    or {"ok": False, "error": ...} if the email is already taken.
    """

    db = SessionLocal()

    try:

        existing = (
            db.query(User)
            .filter(User.email == email)
            .first()
        )

        if existing:
            return {
                "ok": False,
                "error": "Email already registered.",
            }

        user = User(
            user_id=generate_user_id(),
            email=email,
            username=(username or None),
            hashed_password=hash_password(password),
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return {
            "ok": True,
            "user_id": user.user_id,
            "username": user.username,
        }

    finally:
        db.close()


# ----------------------------------------------------------
# LOGIN
# ----------------------------------------------------------

def login(email: str, password: str) -> dict:
    """
    Check credentials and, if correct, hand back a token + user_id.

    Returns {"ok": True, "access_token": ..., "user_id": ...},
    or {"ok": False, "error": ...} if email/password is wrong.
    """

    db = SessionLocal()

    try:

        user = (
            db.query(User)
            .filter(User.email == email)
            .first()
        )

        # Same generic message whether the email is missing or the
        # password is wrong (don't reveal which one to attackers).
        if not user or not verify_password(password, user.hashed_password):
            return {
                "ok": False,
                "error": "Invalid email or password.",
            }

        token = create_access_token(user.user_id)

        return {
            "ok": True,
            "access_token": token,
            "user_id": user.user_id,
            "username": user.username,
        }

    finally:
        db.close()
