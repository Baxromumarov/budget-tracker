from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import crud
from ..config import get_settings
from ..db import get_db
from ..schemas import AuthResponse, LoginRequest, UserRegister, UserOut
from ..security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])

settings = get_settings()


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register_user(data: UserRegister, db: Session = Depends(get_db)) -> AuthResponse:
    if crud.get_user_by_username(db, data.username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken.")
    if data.email and crud.get_user_by_email(db, data.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered.")

    password_hash = hash_password(data.password)
    user = crud.create_user(db, data, password_hash=password_hash)
    token = create_access_token({"sub": user.username})

    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user=UserOut.model_validate(user),
    )


@router.post("/login", response_model=AuthResponse)
def login_user(data: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    user = crud.get_user_by_username(db, data.username)
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

    expires_delta = timedelta(minutes=settings.access_token_expire_minutes)
    token = create_access_token({"sub": user.username}, expires_delta=expires_delta)

    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user=UserOut.model_validate(user),
    )
