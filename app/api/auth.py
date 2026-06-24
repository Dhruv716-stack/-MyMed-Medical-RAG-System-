from fastapi import APIRouter, HTTPException, status

from app.schemas.common_schema import APIResponse

from app.schemas.auth_schema import (
    RegisterRequest,
    LoginRequest,
    TokenData,
)

from app.services.auth_service import (
    register as register_service,
    login as login_service,
)


router = APIRouter(
    prefix="/auth",
    tags=["Auth"],
)


@router.post("/register")
def register(request: RegisterRequest):

    result = register_service(
        email=request.email,
        password=request.password,
        username=request.username,
    )

    if not result["ok"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"],
        )

    return APIResponse(
        success=True,
        message="Registration successful. You can now log in.",
        data={
            "user_id": result["user_id"],
            "username": result.get("username"),
        },
    )


@router.post("/login")
def login(request: LoginRequest):

    result = login_service(
        email=request.email,
        password=request.password,
    )

    if not result["ok"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=result["error"],
        )

    return APIResponse(
        success=True,
        message="Login successful.",
        data=TokenData(
            access_token=result["access_token"],
            user_id=result["user_id"],
            username=result.get("username"),
        ).model_dump(),
    )
