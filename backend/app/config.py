from functools import lru_cache
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator

_ENV_CANDIDATES = (
    Path(__file__).resolve().parent.parent.parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    Path.cwd() / ".env",
)

for env_path in _ENV_CANDIDATES:
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        break


class Settings(BaseSettings):
    """Application configuration sourced from environment variables."""

    database_url: str = "postgresql+psycopg://postgres:postgres@db:5432/budget"
    api_prefix: str = "/api"
    allow_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:8080",
            "http://localhost:8000",
            "https://budget-tracker-frontend.vercel.app",
            "https://budget-tracker-frontend.up.railway.app",
            "https://budget-tracker-front-production.up.railway.app",
        ]
    )
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    telegram_bot_token: str | None = Field(default=None, alias="TELEGRAM_BOT")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    default_currency: str = "USD"
    ai_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(env_file=None, populate_by_name=True)

    @field_validator("allow_origins", mode="before")
    @classmethod
    def parse_allow_origins(cls, value: object) -> list[str]:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in stripped.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        raise ValueError("Invalid allow_origins format.")

    @model_validator(mode="after")
    def populate_from_env(self) -> "Settings":
        if not self.telegram_bot_token:
            self.telegram_bot_token = os.getenv("TELEGRAM_BOT")
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached settings to avoid repeated environment parsing."""
    return Settings()
