from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Literal

from ..models import TransactionKind

TransactionType = Literal["income", "expense"]


@dataclass(slots=True)
class Transaction:
    """Base transaction data structure."""

    id: int | None
    amount: float
    date: date
    category: str
    type: TransactionType
    description: str | None = None

    def get_details(self) -> str:
        """Return a formatted, human-readable representation."""
        type_label = self.type.capitalize()
        description_part = f" ({self.description})" if self.description else ""
        return f"{type_label}: {self.category} on {self.date.isoformat()} — {self.amount:.2f}{description_part}"

    def to_dict(self) -> dict[str, object]:
        """Convert the transaction to a serialisable dictionary."""
        return {
            "id": self.id,
            "amount": self.amount,
            "date": self.date.isoformat(),
            "category": self.category,
            "type": self.type,
            "description": self.description,
        }


@dataclass(slots=True)
class IncomeTransaction(Transaction):
    """Income specialisation for Transaction."""

    type: TransactionType = "income"


@dataclass(slots=True)
class ExpenseTransaction(Transaction):
    """Expense specialisation for Transaction."""

    type: TransactionType = "expense"


@dataclass(slots=True)
class User:
    """Aggregate root for a user's budget data."""

    id: int | None
    name: str
    username: str
    email: str | None
    transactions: list[Transaction] = field(default_factory=list)

    def add_transaction(self, transaction: Transaction) -> None:
        self.transactions.append(transaction)

    def remove_transaction(self, transaction_id: int) -> None:
        self.transactions = [t for t in self.transactions if t.id != transaction_id]

    def get_all_transactions(self) -> list[Transaction]:
        return list(self.transactions)

    def get_transactions_by_category(self, category: str) -> list[Transaction]:
        return [t for t in self.transactions if t.category.lower() == category.lower()]

    def get_transactions_by_type(self, type_: TransactionType) -> list[Transaction]:
        return [t for t in self.transactions if t.type == type_]

    def calculate_totals(self) -> tuple[float, float, float]:
        income = sum(t.amount for t in self.transactions if t.type == "income")
        expenses = sum(t.amount for t in self.transactions if t.type == "expense")
        balance = income - expenses
        return income, expenses, balance


@dataclass(slots=True)
class MonthlyReport:
    month: int
    year: int
    total_income: float
    total_expenses: float
    net_savings: float
    top_category: str | None
    transactions: list[Transaction]

    def to_dict(self) -> dict[str, object]:
        return {
            "month": self.month,
            "year": self.year,
            "total_income": self.total_income,
            "total_expenses": self.total_expenses,
            "net_savings": self.net_savings,
            "top_category": self.top_category,
            "transactions": [t.to_dict() for t in self.transactions],
        }


class ReportGenerator(ABC):
    """Abstract report generator type."""

    @abstractmethod
    def generate(self, user: User, month: int, year: int) -> MonthlyReport:
        """Produce the logical report contents."""

    @abstractmethod
    def export(self, report: MonthlyReport) -> bytes:
        """Serialize a report to bytes."""


class CSVReportGenerator(ReportGenerator):
    """CSV implementation using the report structure."""

    def generate(self, user: User, month: int, year: int) -> MonthlyReport:
        transactions = filter_transactions_by_month(user.transactions, month, year)
        total_income = sum_tx_by_type(transactions, "income")
        total_expenses = sum_tx_by_type(transactions, "expense")
        net = total_income - total_expenses
        top_category = determine_top_category(transactions)
        return MonthlyReport(
            month=month,
            year=year,
            total_income=total_income,
            total_expenses=total_expenses,
            net_savings=net,
            top_category=top_category,
            transactions=list(transactions),
        )

    def export(self, report: MonthlyReport) -> bytes:
        from io import StringIO
        import csv

        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Month", "Year", "Total Income", "Total Expenses", "Net Savings", "Top Category"])
        writer.writerow(
            [
                report.month,
                report.year,
                f"{report.total_income:.2f}",
                f"{report.total_expenses:.2f}",
                f"{report.net_savings:.2f}",
                report.top_category or "",
            ]
        )
        writer.writerow([])
        writer.writerow(["ID", "Type", "Category", "Date", "Amount", "Description"])
        for txn in report.transactions:
            writer.writerow(
                [
                    txn.id or "",
                    txn.type,
                    txn.category,
                    txn.date.isoformat(),
                    f"{txn.amount:.2f}",
                    txn.description or "",
                ]
            )
        return buffer.getvalue().encode("utf-8")


class JSONReportGenerator(ReportGenerator):
    """JSON export implementation."""

    def generate(self, user: User, month: int, year: int) -> MonthlyReport:
        transactions = filter_transactions_by_month(user.transactions, month, year)
        total_income = sum_tx_by_type(transactions, "income")
        total_expenses = sum_tx_by_type(transactions, "expense")
        net = total_income - total_expenses
        top_category = determine_top_category(transactions)
        return MonthlyReport(
            month=month,
            year=year,
            total_income=total_income,
            total_expenses=total_expenses,
            net_savings=net,
            top_category=top_category,
            transactions=list(transactions),
        )

    def export(self, report: MonthlyReport) -> bytes:
        import json

        return json.dumps(report.to_dict(), indent=2).encode("utf-8")


@dataclass(slots=True)
class BudgetManager:
    """Coordinator for user transactions and reports."""

    user: User
    report_generators: dict[str, ReportGenerator]

    def add_transaction(self, transaction: Transaction) -> None:
        self.user.add_transaction(transaction)

    def remove_transaction(self, transaction_id: int) -> None:
        self.user.remove_transaction(transaction_id)

    def generate_monthly_report(self, month: int, year: int, format_: str) -> bytes:
        generator = self.report_generators[format_]
        report = generator.generate(self.user, month, year)
        return generator.export(report)


def filter_transactions_by_month(
    transactions: Iterable[Transaction], month: int, year: int
) -> list[Transaction]:
    return [t for t in transactions if t.date.month == month and t.date.year == year]


def sum_tx_by_type(transactions: Iterable[Transaction], type_: TransactionType) -> float:
    return sum(t.amount for t in transactions if t.type == type_)


def determine_top_category(transactions: Iterable[Transaction]) -> str | None:
    totals: dict[str, float] = {}
    for txn in transactions:
        if txn.category not in totals:
            totals[txn.category] = 0.0
        totals[txn.category] += txn.amount
    if not totals:
        return None
    return max(totals.items(), key=lambda item: item[1])[0]


def domain_transaction_from_model(model: object) -> Transaction:
    from ..models import TransactionModel

    if not isinstance(model, TransactionModel):
        raise TypeError("Expected TransactionModel instance.")

    kwargs = {
        "id": model.id,
        "amount": model.amount,
        "date": model.date,
        "category": model.category,
        "type": model.kind.value,
        "description": model.description,
    }

    if model.kind == TransactionKind.INCOME:
        return IncomeTransaction(**kwargs)
    return ExpenseTransaction(**kwargs)
