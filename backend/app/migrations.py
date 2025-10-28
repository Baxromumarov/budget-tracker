from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


def _ensure_currency_column(engine: Engine) -> None:
    inspector = inspect(engine)
    try:
        columns = {column["name"] for column in inspector.get_columns("transactions")}
    except SQLAlchemyError as exc:  # pragma: no cover - defensive
        logger.error("Failed to inspect transactions table: %s", exc)
        return

    if "currency" in columns:
        return

    logger.info("Adding currency column to transactions table (default USD).")

    dialect = engine.dialect.name.lower()
    add_column_sql = "ALTER TABLE transactions ADD COLUMN currency VARCHAR(3)"

    try:
        with engine.begin() as connection:
            if dialect == "sqlite":
                connection.execute(
                    text(f"{add_column_sql} NOT NULL DEFAULT 'USD'")
                )
            else:
                connection.execute(text(add_column_sql))
                connection.execute(
                    text("UPDATE transactions SET currency = 'USD' WHERE currency IS NULL")
                )
                connection.execute(
                    text("ALTER TABLE transactions ALTER COLUMN currency SET DEFAULT 'USD'")
                )
                connection.execute(
                    text("ALTER TABLE transactions ALTER COLUMN currency SET NOT NULL")
                )
    except SQLAlchemyError as exc:  # pragma: no cover - defensive
        logger.error("Failed to add currency column: %s", exc)


def run_migrations(engine: Engine) -> None:
    """Execute lightweight, idempotent migrations on application start."""
    _ensure_currency_column(engine)
