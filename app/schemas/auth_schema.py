from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):

    email: EmailStr = Field(
        description="User's email address (must be unique)"
    )

    password: str = Field(
        min_length=6,
        description="Password (at least 6 characters)"
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
