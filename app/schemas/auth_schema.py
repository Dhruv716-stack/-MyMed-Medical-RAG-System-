from pydantic import BaseModel, EmailStr, Field

from typing import Optional


class RegisterRequest(BaseModel):

    email: EmailStr = Field(
        description="User's email address (must be unique)"
    )

    password: str = Field(
    min_length=8,
    max_length=128,
    description="Password"
    )

    username: Optional[str] = Field(
        default=None,
        description="Optional display name / preferred username"
    )


class LoginRequest(BaseModel):

    email: EmailStr = Field(
        description="Registered email address"
    )

    password: str = Field(
        description="Account password"
    )


class TokenData(BaseModel):

    access_token: str = Field(
        description="JWT token to send on future requests"
    )

    token_type: str = Field(
        default="bearer"
    )

    user_id: str = Field(
        description="The verified user_id for this account"
    )

    username: Optional[str] = Field(
        default=None,
        description="The user's display name, if they set one"
    )
    
    expires_in: int = Field(
    description="Token validity in seconds"
    )
