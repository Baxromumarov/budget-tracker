from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration sourced from environment variables."""

    database_url: str = "postgresql+psycopg://postgres:postgres@db:5432/budget"
    api_prefix: str = "/api"
    allow_origins: list[str] = ["http://localhost:5173"]
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings to avoid repeated environment parsing."""
    return Settings()
