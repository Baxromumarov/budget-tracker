from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from .. import crud
from ..db import get_db
from ..schemas import TransactionCreate, TransactionOut, TransactionType, TransactionUpdate

router = APIRouter(prefix="/users/{user_id}/transactions", tags=["transactions"])


@router.post("", response_model=TransactionOut, status_code=status.HTTP_201_CREATED)
def create_transaction(
    user_id: int,
    data: TransactionCreate,
    db: Session = Depends(get_db),
) -> TransactionOut:
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    transaction = crud.create_transaction(db, user_id, data)
    return TransactionOut.model_validate(transaction)


@router.get("", response_model=list[TransactionOut])
def list_transactions(
    user_id: int,
    db: Session = Depends(get_db),
    category: str | None = Query(default=None),
    type_: TransactionType | None = Query(default=None, alias="type"),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
) -> list[TransactionOut]:
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    transactions = crud.list_transactions(
        db,
        user_id=user_id,
        category=category,
        type_=type_,
        start_date=start_date,
        end_date=end_date,
    )
    return [TransactionOut.model_validate(tx) for tx in transactions]


@router.get("/{transaction_id}", response_model=TransactionOut)
def get_transaction(
    user_id: int,
    transaction_id: int,
    db: Session = Depends(get_db),
) -> TransactionOut:
    transaction = crud.get_transaction(db, transaction_id, user_id)
    if not transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found.")
    return TransactionOut.model_validate(transaction)


@router.put("/{transaction_id}", response_model=TransactionOut)
def update_transaction(
    user_id: int,
    transaction_id: int,
    data: TransactionUpdate,
    db: Session = Depends(get_db),
) -> TransactionOut:
    transaction = crud.get_transaction(db, transaction_id, user_id)
    if not transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found.")
    updated = crud.update_transaction(db, transaction, data)
    return TransactionOut.model_validate(updated)


@router.delete("/{transaction_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_transaction(
    user_id: int,
    transaction_id: int,
    db: Session = Depends(get_db),
) -> None:
    transaction = crud.get_transaction(db, transaction_id, user_id)
    if not transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found.")
    crud.delete_transaction(db, transaction)

