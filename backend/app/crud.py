from collections import defaultdict
from datetime import date, timedelta

from sqlalchemy import extract, func, select, update
from sqlalchemy.orm import Session

from .domain.entities import (
    IncomeTransaction,
    ExpenseTransaction,
    Transaction,
    domain_transaction_from_model,
)
from .models import TelegramProfileModel, TransactionKind, TransactionModel, UserModel
from .config import get_settings

settings = get_settings()
from .schemas import TransactionCreate, TransactionType, TransactionUpdate, UserRegister


def create_user(db: Session, data: UserRegister, password_hash: str) -> UserModel:
    user = UserModel(
        name=data.name,
        username=data.username,
        email=data.email,
        password_hash=password_hash,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user(db: Session, user_id: int) -> UserModel | None:
    return db.get(UserModel, user_id)


def get_user_by_email(db: Session, email: str) -> UserModel | None:
    return db.scalar(select(UserModel).where(UserModel.email == email))


def get_user_by_username(db: Session, username: str) -> UserModel | None:
    return db.scalar(select(UserModel).where(UserModel.username == username))

def get_user_by_telegram_id(db: Session, telegram_id: str) -> UserModel | None:
    profile = db.scalar(
        select(TelegramProfileModel).where(TelegramProfileModel.telegram_id == telegram_id)
    )
    return profile.user if profile else None


def list_users(db: Session) -> list[UserModel]:
    return list(db.scalars(select(UserModel).order_by(UserModel.name)))


def create_transaction(db: Session, user_id: int, data: TransactionCreate) -> TransactionModel:
    kind = TransactionKind(data.type)
    currency = (data.currency or settings.default_currency).upper()
    transaction = TransactionModel(
        user_id=user_id,
        amount=data.amount,
        date=data.date,
        category=data.category,
        kind=kind,
        currency=currency,
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
    if "currency" in changes and changes["currency"]:
        changes["currency"] = str(changes["currency"]).upper()

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


def delete_transactions_for_user(db: Session, user_id: int) -> int:
    count = db.query(TransactionModel).filter(TransactionModel.user_id == user_id).delete()
    db.commit()
    return count


def delete_transaction_by_id(db: Session, transaction_id: int, user_id: int) -> bool:
    tx = (
        db.query(TransactionModel)
        .filter(TransactionModel.id == transaction_id, TransactionModel.user_id == user_id)
        .first()
    )
    if not tx:
        return False
    db.delete(tx)
    db.commit()
    return True


def count_transactions(db: Session, user_id: int) -> int:
    return db.query(func.count(TransactionModel.id)).filter(TransactionModel.user_id == user_id).scalar() or 0


def aggregate_transactions(transactions: list[TransactionModel]) -> dict[str, object]:
    income_by_currency: defaultdict[str, float] = defaultdict(float)
    expenses_by_currency: defaultdict[str, float] = defaultdict(float)
    category_totals: defaultdict[str, float] = defaultdict(float)
    monthly_totals: defaultdict[str, defaultdict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"income": 0.0, "expenses": 0.0})
    )

    for tx in transactions:
        currency = (getattr(tx, "currency", None) or settings.default_currency).upper()
        amount = float(tx.amount)
        month_key = tx.date.strftime("%Y-%m") if isinstance(tx.date, date) else str(tx.date)

        if tx.kind == TransactionKind.INCOME:
            income_by_currency[currency] += amount
            monthly_totals[month_key][currency]["income"] += amount
        else:
            expenses_by_currency[currency] += amount
            monthly_totals[month_key][currency]["expenses"] += amount

        category_totals[tx.category] += amount

    # Convert nested default dicts to plain dicts with floats
    monthly_serializable: dict[str, dict[str, dict[str, float]]] = {}
    for month_key, currency_map in monthly_totals.items():
        monthly_serializable[month_key] = {
            currency: {
                "income": float(values["income"]),
                "expenses": float(values["expenses"]),
            }
            for currency, values in currency_map.items()
        }

    return {
        "income_by_currency": dict(income_by_currency),
        "expenses_by_currency": dict(expenses_by_currency),
        "category_totals": dict(category_totals),
        "monthly_totals": monthly_serializable,
    }


def build_domain_transaction(model: TransactionModel) -> Transaction:
    return domain_transaction_from_model(model)


def hydrate_user_with_transactions(db: Session, user: UserModel) -> tuple:
    transactions = list_transactions(db, user.id)
    domain_transactions = [build_domain_transaction(tx) for tx in transactions]
    return user, domain_transactions


def get_telegram_profile(db: Session, telegram_id: str) -> TelegramProfileModel | None:
    return db.scalar(
        select(TelegramProfileModel).where(TelegramProfileModel.telegram_id == telegram_id)
    )


def create_telegram_profile(
    db: Session,
    user: UserModel,
    telegram_id: str,
    username: str | None,
    first_name: str | None,
    last_name: str | None,
    language_code: str | None,
) -> TelegramProfileModel:
    profile = TelegramProfileModel(
        user_id=user.id,
        telegram_id=telegram_id,
        username=username,
        first_name=first_name,
        last_name=last_name,
        language_code=language_code,
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def update_telegram_profile(
    db: Session,
    profile: TelegramProfileModel,
    **changes: object,
) -> TelegramProfileModel:
    for key, value in changes.items():
        if hasattr(profile, key) and value is not None:
            setattr(profile, key, value)
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def calculate_monthly_summary(db: Session, user_id: int, month: int, year: int) -> dict[str, float | None]:
    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)

    transactions = list_transactions(
        db,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )

    aggregated = aggregate_transactions(transactions)
    income_by_currency = aggregated["income_by_currency"]
    expenses_by_currency = aggregated["expenses_by_currency"]
    category_totals = aggregated["category_totals"]
    monthly_totals = aggregated["monthly_totals"]

    default_code = settings.default_currency.upper()
    total_income_default = float(income_by_currency.get(default_code, 0.0))
    total_expenses_default = float(expenses_by_currency.get(default_code, 0.0))

    all_currencies = sorted(set(income_by_currency) | set(expenses_by_currency))
    totals_by_currency = {
        code: {
            "income": float(income_by_currency.get(code, 0.0)),
            "expenses": float(expenses_by_currency.get(code, 0.0)),
            "balance": float(income_by_currency.get(code, 0.0) - expenses_by_currency.get(code, 0.0)),
        }
        for code in all_currencies
    }

    top_category = None
    if category_totals:
        top_category = max(category_totals.items(), key=lambda item: item[1])[0]

    return {
        "total_income": total_income_default,
        "total_expenses": total_expenses_default,
        "balance": total_income_default - total_expenses_default,
        "top_category": top_category,
        "totals_by_currency": totals_by_currency,
        "monthly_totals": monthly_totals,
    }
