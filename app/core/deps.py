"""
Request dependencies.

get_current_user() reads the 'Authorization: Bearer <token>' header,
decodes the token, and returns the verified user_id. If the token is
missing or invalid, the request is rejected with 401.

Any endpoint that does `user_id: str = Depends(get_current_user)` is
automatically protected and gets a trusted user_id.
"""

from fastapi import Depends, HTTPException, status

from fastapi.security import OAuth2PasswordBearer

from app.core.security import decode_access_token


# Tells FastAPI where the login endpoint is (used by Swagger's
# "Authorize" button) and to read the Bearer token from the header.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:

    user_id = decode_access_token(token)

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_id
