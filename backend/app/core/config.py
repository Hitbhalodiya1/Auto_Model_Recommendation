"""
Application configuration using Pydantic Settings.
Loads from environment variables with .env file fallback.
"""

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "AutoRec"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    # ── API ───────────────────────────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./autorec.db"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_ECHO: bool = False

    # ── Storage ───────────────────────────────────────────────────────────────
    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 500

    # ── ML Engine ─────────────────────────────────────────────────────────────
    MAX_TRAINING_WORKERS: int = 4
    CV_FOLDS: int = 5
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    OVERFITTING_THRESHOLD: float = 0.15

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "text"

    @field_validator("TEST_SIZE")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        if not 0.1 <= v <= 0.5:
            raise ValueError("TEST_SIZE must be between 0.1 and 0.5")
        return v

    @field_validator("CV_FOLDS")
    @classmethod
    def validate_cv_folds(cls, v: int) -> int:
        if v < 2:
            raise ValueError("CV_FOLDS must be at least 2")
        return v

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance. Use as FastAPI dependency."""
    return Settings()
