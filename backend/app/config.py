from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    allow_origins: list[str] = ["http://localhost:5173"]
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    model_config = SettingsConfigDict(env_file=None)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings to avoid repeated environment parsing."""
    return Settings()
