"""
Security helpers for authentication.

Two jobs:
  1. Passwords  -> hash on register, verify on login (bcrypt).
  2. Tokens     -> issue a signed JWT on login, decode it on each request.

The raw password is NEVER stored. Only the bcrypt hash is.
"""

import os
import uuid

from datetime import datetime, timedelta, timezone

import bcrypt

from jose import jwt, JWTError


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

# Secret used to sign tokens. In production this MUST come from the
# environment. The fallback only exists so local dev works out of the box.
SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "dev-only-insecure-change-me"
)

ALGORITHM = "HS256"

# How long a login stays valid (minutes).
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7   # 7 days


# ----------------------------------------------------------
# PASSWORDS
# ----------------------------------------------------------

# bcrypt operates on bytes and has a 72-byte limit, so we encode and
# truncate defensively. The stored hash is kept as a UTF-8 string.

def hash_password(password: str) -> str:
    """Scramble a plain password into a one-way bcrypt hash."""
    pw_bytes = password.encode("utf-8")[:72]
    hashed = bcrypt.hashpw(pw_bytes, bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plain password against the stored hash."""
    pw_bytes = plain_password.encode("utf-8")[:72]
    return bcrypt.checkpw(pw_bytes, hashed_password.encode("utf-8"))


# ----------------------------------------------------------
# USER ID
# ----------------------------------------------------------

def generate_user_id() -> str:
    """Create a fresh, unique user_id like 'u_3f9a...'."""
    return "u_" + uuid.uuid4().hex


# ----------------------------------------------------------
# TOKENS (JWT)
# ----------------------------------------------------------

def create_access_token(user_id: str) -> str:
    """Make a signed token that proves 'this request is user_id'."""

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": user_id,      # 'sub' = subject = who this token belongs to
        "exp": expire, # 'exp' = when it stops being valid
        "iat": datetime.now(timezone.utc), # 'iat' = when it was issued
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """
    Read a token and return the user_id inside it.
    Returns None if the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None
