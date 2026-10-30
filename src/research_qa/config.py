"""Environment-based application configuration, independent for each provider."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load settings from the environment or a local, untracked ``.env`` file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    collection_name: str = Field(default="research", min_length=3, max_length=512)

    chat_provider: Literal["openai", "openai-compatible"] = "openai"
    chat_model: str = Field(default="gpt-4o-mini", min_length=1)
    chat_base_url: str = "https://api.openai.com/v1"
    chat_api_key: SecretStr | None = None

    embedding_provider: Literal["openai", "openai-compatible"] = "openai"
    embedding_model: str = Field(default="text-embedding-3-small", min_length=1)
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: SecretStr | None = None

    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_pdf_pages: int = Field(default=500, gt=0)
    max_extracted_chars: int = Field(default=2_000_000, gt=0)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    default_top_k: int = Field(default=5, ge=1, le=20)
    min_relevance_score: float = Field(default=0.35, ge=0, le=1)
    max_context_chars: int = Field(default=16000, gt=0)
    request_timeout_seconds: float = Field(default=60, gt=0)

    @field_validator("chat_base_url", "embedding_base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """Permit HTTP(S) endpoints but keep secrets out of URL configuration."""
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Provider base URLs must be absolute HTTP(S) URLs")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Provider base URLs must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("Provider base URLs must not contain a query or fragment")
        return value

    @field_validator("chat_model", "embedding_model", "collection_name")
    @classmethod
    def strip_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Model and collection names must not be blank")
        return value

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return one settings snapshot per process."""
    return Settings()
