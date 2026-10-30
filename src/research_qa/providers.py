"""Factories for independently configured OpenAI-compatible providers."""

import hashlib
import json

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from research_qa.config import Settings


def _api_key(provider: str, configured: SecretStr | None, setting: str) -> str:
    key = configured.get_secret_value() if configured else ""
    if key.strip():
        return key
    if provider == "openai-compatible":
        # Local OpenAI-compatible servers often do not require authentication,
        # but the OpenAI client still requires a nonempty key at construction.
        return "local-no-key"
    raise ValueError(f"{setting} is required for the openai provider")


def build_embeddings(settings: Settings) -> Embeddings:
    """Build an embedding client without inheriting another provider's key/URL."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=_api_key(
            settings.embedding_provider, settings.embedding_api_key, "EMBEDDING_API_KEY"
        ),
        base_url=settings.embedding_base_url,
        request_timeout=settings.request_timeout_seconds,
        max_retries=1,
        check_embedding_ctx_length=settings.embedding_provider == "openai",
    )


def build_chat(settings: Settings) -> BaseChatModel:
    """Build a chat-completions client using the chat provider's own settings."""
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=_api_key(settings.chat_provider, settings.chat_api_key, "CHAT_API_KEY"),
        base_url=settings.chat_base_url,
        temperature=0,
        timeout=settings.request_timeout_seconds,
        max_retries=1,
    )


def embedding_fingerprint(settings: Settings) -> str:
    """Identify embedding spaces so incompatible persistent indexes are rejected.

    API keys are deliberately excluded: rotating credentials does not change the
    embedding space, and secrets must never be stored in collection metadata.
    """
    identity = {
        "provider": settings.embedding_provider,
        "base_url": settings.embedding_base_url.rstrip("/"),
        "model": settings.embedding_model,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
