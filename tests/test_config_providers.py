"""Provider isolation and configuration invariants, with HTTP disabled."""

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from research_qa.config import Settings
from research_qa.providers import build_chat, build_embeddings, embedding_fingerprint


@pytest.fixture(autouse=True)
def no_http(monkeypatch):
    def unexpected_request(*args, **kwargs):
        raise AssertionError("Configuration tests must not make HTTP requests")

    monkeypatch.setattr(httpx.Client, "send", unexpected_request)
    monkeypatch.setattr(httpx.AsyncClient, "send", unexpected_request)


def test_local_provider_keys_and_endpoints_ignore_ambient_openai_settings(settings, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-ambient-secret")
    monkeypatch.setenv("OPENAI_API_BASE", "https://ambient-base.invalid/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient-url.invalid/v1")

    chat = build_chat(settings)
    embeddings = build_embeddings(settings)

    assert chat.openai_api_base == "http://chat.invalid/v1"
    assert embeddings.openai_api_base == "http://embeddings.invalid/v1"
    assert chat.openai_api_key.get_secret_value() == "local-no-key"
    assert embeddings.openai_api_key.get_secret_value() == "local-no-key"
    assert embeddings.check_embedding_ctx_length is False


@pytest.mark.parametrize(
    "factory,setting",
    [
        (build_chat, "CHAT_API_KEY"),
        (build_embeddings, "EMBEDDING_API_KEY"),
    ],
)
def test_missing_explicit_openai_key_does_not_use_ambient_key(
    settings, monkeypatch, factory, setting
):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-ambient-secret")
    configured = settings.model_copy(
        update={
            "chat_provider": "openai",
            "embedding_provider": "openai",
            "chat_api_key": None,
            "embedding_api_key": None,
        }
    )
    with pytest.raises(ValueError, match=setting):
        factory(configured)


def test_dotenv_keeps_chat_and_embedding_configuration_independent(tmp_path, monkeypatch):
    config = {
        "CHAT_PROVIDER": "openai-compatible",
        "CHAT_MODEL": "local-chat",
        "CHAT_BASE_URL": "http://localhost:1234/v1/",
        "CHAT_API_KEY": "chat-only-key",
        "EMBEDDING_PROVIDER": "openai-compatible",
        "EMBEDDING_MODEL": "local-embeddings",
        "EMBEDDING_BASE_URL": "http://localhost:5678/v1/",
        "EMBEDDING_API_KEY": "embedding-only-key",
    }
    for key in config:
        monkeypatch.delenv(key, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("\n".join(f"{key}={value}" for key, value in config.items()))
    loaded = Settings(_env_file=env_path)

    chat, embeddings = build_chat(loaded), build_embeddings(loaded)
    assert chat.model_name == "local-chat"
    assert embeddings.model == "local-embeddings"
    assert chat.openai_api_base == "http://localhost:1234/v1"
    assert embeddings.openai_api_base == "http://localhost:5678/v1"
    assert chat.openai_api_key.get_secret_value() == "chat-only-key"
    assert embeddings.openai_api_key.get_secret_value() == "embedding-only-key"


@pytest.mark.parametrize("field", ["chat_base_url", "embedding_base_url"])
@pytest.mark.parametrize(
    "value",
    [
        "ftp://example.com/v1",
        "relative/endpoint",
        "https://user:secret@example.com/v1",
        "http://localhost/v1?key=secret",
        "https://example.com/v1#fragment",
    ],
)
def test_both_provider_urls_reject_unsafe_or_ambiguous_endpoints(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize("size,overlap", [(100, 100), (100, 101), (100, -1), (0, 0)])
def test_chunk_configuration_rejects_invalid_boundaries(size, overlap):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=size, chunk_overlap=overlap)


@pytest.mark.parametrize(
    "change",
    [
        {"embedding_provider": "openai"},
        {"embedding_base_url": "http://different-embeddings.invalid/v1"},
        {"embedding_model": "different-embedding-space"},
    ],
)
def test_embedding_identity_changes_require_a_different_fingerprint(settings, change):
    baseline = embedding_fingerprint(settings)
    assert embedding_fingerprint(settings.model_copy(update=change)) != baseline


def test_fingerprint_is_secret_free_and_stable_when_only_credentials_or_chat_change(settings):
    rotated = settings.model_copy(
        update={
            "chat_model": "different-chat-model",
            "chat_base_url": "http://different-chat.invalid/v1",
            "chat_api_key": SecretStr("new-chat-secret"),
            "embedding_api_key": SecretStr("new-embedding-secret"),
        }
    )
    fingerprint = embedding_fingerprint(settings)
    assert len(fingerprint) == 64
    assert set(fingerprint) <= set("0123456789abcdef")
    assert embedding_fingerprint(rotated) == fingerprint
