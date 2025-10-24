from datetime import date
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


TransactionType = Literal["income", "expense"]


class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr


class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr

    class Config:
        from_attributes = True


class TransactionBase(BaseModel):
    amount: float = Field(gt=0)
    date: date
    category: str = Field(min_length=1, max_length=120)
    type: TransactionType
    description: str | None = Field(default=None, max_length=255)


class TransactionCreate(TransactionBase):
    pass


class TransactionUpdate(BaseModel):
    amount: float | None = Field(default=None, gt=0)
    date: date | None = None
    category: str | None = Field(default=None, min_length=1, max_length=120)
    type: TransactionType | None = None
    description: str | None = Field(default=None, max_length=255)


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
    top_category: str | None


class ReportResponse(BaseModel):
    content: str
    filename: str
    media_type: str

