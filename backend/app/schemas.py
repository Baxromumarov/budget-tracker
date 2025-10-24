from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field


TransactionType = Literal["income", "expense"]


class UserRegister(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    username: str = Field(min_length=3, max_length=120)
    email: Optional[EmailStr] = None
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    id: int
    name: str
    username: str
    email: Optional[EmailStr] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ProfileOut(UserOut):
    pass


class LoginRequest(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AuthResponse(Token):
    user: UserOut


class TransactionBase(BaseModel):
    amount: float = Field(gt=0)
    date: date
    category: str = Field(min_length=1, max_length=120)
    type: TransactionType
    description: Optional[str] = Field(default=None, max_length=255)


class TransactionCreate(TransactionBase):
    pass


class TransactionUpdate(BaseModel):
    amount: Optional[float] = Field(default=None, gt=0)
    date: Optional[date] = None
    category: Optional[str] = Field(default=None, min_length=1, max_length=120)
    type: Optional[TransactionType] = None
    description: Optional[str] = Field(default=None, max_length=255)


class TransactionOut(TransactionBase):
    id: int
    user_id: int

    class Config:
        from_attributes = True


class MonthlySummary(BaseModel):
    month: int
    year: int
    total_income: float
    total_expenses: float
    balance: float
    top_category: Optional[str]
