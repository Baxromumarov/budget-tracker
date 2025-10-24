from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from .. import crud
from ..db import get_db
from ..domain.entities import BudgetManager, CSVReportGenerator, JSONReportGenerator, User
from ..schemas import MonthlySummary

router = APIRouter(prefix="/users/{user_id}", tags=["reports"])


@router.get("/summary", response_model=MonthlySummary)
def get_monthly_summary(
    user_id: int,
    month: int = Query(..., ge=1, le=12),
    year: int = Query(..., ge=2000),
    db: Session = Depends(get_db),
) -> MonthlySummary:
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    summary = crud.calculate_monthly_summary(db, user_id=user.id, month=month, year=year)
    return MonthlySummary(
        month=month,
        year=year,
        total_income=summary["total_income"],
        total_expenses=summary["total_expenses"],
        balance=summary["balance"],
        top_category=summary["top_category"],
    )


@router.get("/reports/{format}", response_class=Response)
def export_report(
    user_id: int,
    format: str,
    month: int = Query(..., ge=1, le=12),
    year: int = Query(..., ge=2000),
    db: Session = Depends(get_db),
) -> Response:
    if format not in {"csv", "json"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported format.")

    user_model = crud.get_user(db, user_id)
    if not user_model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    transactions = crud.list_transactions(db, user_id=user_model.id)
    domain_transactions = [crud.build_domain_transaction(tx) for tx in transactions]
    domain_user = User(
        id=user_model.id,
        name=user_model.name,
        email=user_model.email,
        transactions=domain_transactions,
    )

    manager = BudgetManager(
        user=domain_user,
        report_generators={
            "csv": CSVReportGenerator(),
            "json": JSONReportGenerator(),
        },
    )

    content = manager.generate_monthly_report(month=month, year=year, format_=format)
    filename = f"report_{year:04d}_{month:02d}.{format}"
    media_type = "text/csv" if format == "csv" else "application/json"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

