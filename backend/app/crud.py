from datetime import date

from sqlalchemy import extract, func, select, update
from sqlalchemy.orm import Session

from .domain.entities import (
    IncomeTransaction,
    ExpenseTransaction,
    Transaction,
    domain_transaction_from_model,
)
from .models import TransactionKind, TransactionModel, UserModel
from .schemas import TransactionCreate, TransactionType, TransactionUpdate, UserCreate


def create_user(db: Session, data: UserCreate) -> UserModel:
    user = UserModel(name=data.name, email=data.email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user(db: Session, user_id: int) -> UserModel | None:
    return db.get(UserModel, user_id)


def get_user_by_email(db: Session, email: str) -> UserModel | None:
    return db.scalar(select(UserModel).where(UserModel.email == email))


def list_users(db: Session) -> list[UserModel]:
    return list(db.scalars(select(UserModel).order_by(UserModel.name)))


def create_transaction(db: Session, user_id: int, data: TransactionCreate) -> TransactionModel:
    kind = TransactionKind(data.type)
    transaction = TransactionModel(
        user_id=user_id,
        amount=data.amount,
        date=data.date,
        category=data.category,
        kind=kind,
        description=data.description,
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)
    return transaction


def get_transaction(db: Session, transaction_id: int, user_id: int) -> TransactionModel | None:
    stmt = select(TransactionModel).where(
        TransactionModel.id == transaction_id, TransactionModel.user_id == user_id
    )
    return db.scalar(stmt)


def list_transactions(
    db: Session,
    user_id: int,
    category: str | None = None,
    type_: TransactionType | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[TransactionModel]:
    stmt = select(TransactionModel).where(TransactionModel.user_id == user_id)

    if category:
        stmt = stmt.where(TransactionModel.category.ilike(category))
    if type_:
        stmt = stmt.where(TransactionModel.kind == TransactionKind(type_))
    if start_date:
        stmt = stmt.where(TransactionModel.date >= start_date)
    if end_date:
        stmt = stmt.where(TransactionModel.date <= end_date)

    stmt = stmt.order_by(TransactionModel.date.desc(), TransactionModel.id.desc())
    return list(db.scalars(stmt))


def update_transaction(
    db: Session,
    transaction: TransactionModel,
    data: TransactionUpdate,
) -> TransactionModel:
    changes = data.dict(exclude_unset=True)
    if "type" in changes:
        changes["kind"] = TransactionKind(changes.pop("type"))

    if changes:
        stmt = (
            update(TransactionModel)
            .where(TransactionModel.id == transaction.id)
            .values(**changes)
            .execution_options(synchronize_session="fetch")
        )
        db.execute(stmt)
        db.commit()
        db.refresh(transaction)
    return transaction


def delete_transaction(db: Session, transaction: TransactionModel) -> None:
    db.delete(transaction)
    db.commit()


def build_domain_transaction(model: TransactionModel) -> Transaction:
    return domain_transaction_from_model(model)


def hydrate_user_with_transactions(db: Session, user: UserModel) -> tuple:
    transactions = list_transactions(db, user.id)
    domain_transactions = [build_domain_transaction(tx) for tx in transactions]
    return user, domain_transactions


def calculate_monthly_summary(db: Session, user_id: int, month: int, year: int) -> dict[str, float | None]:
    income_stmt = (
        select(func.coalesce(func.sum(TransactionModel.amount), 0.0))
        .where(
            TransactionModel.user_id == user_id,
            TransactionModel.kind == TransactionKind.INCOME,
            extract("month", TransactionModel.date) == month,
            extract("year", TransactionModel.date) == year,
        )
    )
    expense_stmt = (
        select(func.coalesce(func.sum(TransactionModel.amount), 0.0))
        .where(
            TransactionModel.user_id == user_id,
            TransactionModel.kind == TransactionKind.EXPENSE,
            extract("month", TransactionModel.date) == month,
            extract("year", TransactionModel.date) == year,
        )
    )
    top_category_stmt = (
        select(TransactionModel.category, func.sum(TransactionModel.amount).label("total"))
        .where(
            TransactionModel.user_id == user_id,
            extract("month", TransactionModel.date) == month,
            extract("year", TransactionModel.date) == year,
        )
        .group_by(TransactionModel.category)
        .order_by(func.sum(TransactionModel.amount).desc())
    )

    total_income = db.scalar(income_stmt) or 0.0
    total_expenses = db.scalar(expense_stmt) or 0.0
    balance = total_income - total_expenses
    top_category_row = db.execute(top_category_stmt).first()
    top_category = top_category_row[0] if top_category_row else None

    return {
        "total_income": float(total_income),
        "total_expenses": float(total_expenses),
        "balance": float(balance),
        "top_category": top_category,
    }

