from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from .. import crud
from ..db import get_db
from ..domain.entities import BudgetManager, CSVReportGenerator, JSONReportGenerator, User
from ..schemas import MonthlySummary
from ..models import UserModel
from ..security import get_current_user

router = APIRouter(prefix="/me", tags=["reports"])


@router.get("/summary", response_model=MonthlySummary)
def get_monthly_summary(
    month: int = Query(..., ge=1, le=12),
    year: int = Query(..., ge=2000),
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
) -> MonthlySummary:
    summary = crud.calculate_monthly_summary(db, user_id=current_user.id, month=month, year=year)
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
    format: str,
    month: int = Query(..., ge=1, le=12),
    year: int = Query(..., ge=2000),
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
) -> Response:
    if format not in {"csv", "json"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported format.")

    transactions = crud.list_transactions(db, user_id=current_user.id)
    domain_transactions = [crud.build_domain_transaction(tx) for tx in transactions]
    domain_user = User(
        id=current_user.id,
        name=current_user.name,
        username=current_user.username,
        email=current_user.email,
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
