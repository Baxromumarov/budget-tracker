from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import Base, engine
from .routers import reports, transactions, users

settings = get_settings()

app = FastAPI(title="Expense Tracker API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router, prefix=settings.api_prefix)
app.include_router(transactions.router, prefix=settings.api_prefix)
app.include_router(reports.router, prefix=settings.api_prefix)


@app.on_event("startup")
def on_startup() -> None:
    """Ensure database tables exist."""
    Base.metadata.create_all(bind=engine)


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

