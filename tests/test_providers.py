"""Exercise real LangChain/OpenAI adapters with mocked HTTP, without live APIs."""

import json

import httpx
import pytest
from conftest import REVENUE, TopicEmbeddings, pdf_bytes, supported_answer
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from research_qa import providers
from research_qa.config import Settings
from research_qa.main import create_app


def test_openai_compatible_http_contract_and_independent_endpoints(settings, monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        body = json.loads(request.content)
        if request.url.path == "/v1/embeddings":
            assert request.url.host == "embeddings.invalid"
            assert request.headers["authorization"] == "Bearer embedding-secret"
            assert body["model"] == "test-embeddings"
            # Compatible endpoints receive strings rather than OpenAI token IDs.
            assert all(isinstance(text, str) for text in body["input"])
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": body["model"],
                    "data": [
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": TopicEmbeddings.vector(text),
                        }
                        for index, text in enumerate(body["input"])
                    ],
                    "usage": {"prompt_tokens": 12, "total_tokens": 12},
                },
            )
        assert request.url.path == "/v1/chat/completions"
        assert request.url.host == "chat.invalid"
        assert request.headers["authorization"] == "Bearer chat-secret"
        assert body["model"] == "test-chat"
        assert [message["role"] for message in body["messages"]] == ["system", "user"]
        assert json.loads(body["messages"][1]["content"])["evidence"][0]["page"] == 2
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": supported_answer()},
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            },
        )

    original_embeddings = providers.OpenAIEmbeddings
    original_chat = providers.ChatOpenAI
    # A host OPENAI_BASE_URL/key must not silently replace either explicit provider.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong-provider.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-inherited-key")
    configured = settings.model_copy(
        update={
            "chat_api_key": SecretStr("chat-secret"),
            "embedding_api_key": SecretStr("embedding-secret"),
        }
    )
    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        monkeypatch.setattr(
            providers,
            "OpenAIEmbeddings",
            lambda **kwargs: original_embeddings(http_client=http_client, **kwargs),
        )
        monkeypatch.setattr(
            providers,
            "ChatOpenAI",
            lambda **kwargs: original_chat(http_client=http_client, **kwargs),
        )
        app = create_app(configured)
        with TestClient(app) as client:
            response = client.post(
                "/documents",
                files={
                    "file": ("research.pdf", pdf_bytes("", REVENUE), "application/pdf"),
                },
            )
            assert response.status_code == 201
            response = client.post("/ask", json={"question": "Revenue?"})
            assert response.status_code == 200
            assert response.json()["abstained"] is False
            assert response.json()["citations"][0]["page"] == 2
    assert len(requests) == 3  # document embedding, query embedding, then chat


def test_embedding_http_failure_cleans_upload_and_returns_safe_error(settings, chat, monkeypatch):
    def handle(_request):
        return httpx.Response(503, json={"error": {"message": "private-upstream-details"}})

    original_embeddings = providers.OpenAIEmbeddings
    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        monkeypatch.setattr(
            providers,
            "OpenAIEmbeddings",
            lambda **kwargs: original_embeddings(
                http_client=http_client,
                max_retries=0,
                **{k: v for k, v in kwargs.items() if k != "max_retries"},
            ),
        )
        with TestClient(create_app(settings, chat=chat)) as client:
            response = client.post(
                "/documents",
                files={
                    "file": ("report.pdf", pdf_bytes(REVENUE), "application/pdf"),
                },
            )
            assert response.status_code == 502
            assert "private-upstream-details" not in response.text
            assert client.get("/health").json()["indexed_chunks"] == 0
            assert not list((settings.data_dir / "uploads").glob("*.pdf"))
            chat.invoke.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("chat_base_url", "file:///etc/passwd"),
        ("embedding_base_url", "https://user:secret@example.com/v1"),
        ("chat_base_url", "https://example.com/v1?api_key=secret"),
        ("chunk_overlap", 1000),
        ("max_upload_bytes", 0),
        ("default_top_k", 21),
    ],
)
def test_invalid_configuration_is_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_credentials_required_for_openai_and_optional_for_local(settings):
    openai_settings = settings.model_copy(
        update={
            "chat_provider": "openai",
            "embedding_provider": "openai",
            "chat_api_key": None,
            "embedding_api_key": None,
        }
    )
    with pytest.raises(ValueError, match="CHAT_API_KEY"):
        providers.build_chat(openai_settings)
    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        providers.build_embeddings(openai_settings)
    # Constructing local clients makes no HTTP request and needs no real secret.
    assert providers.build_chat(settings).model_name == "test-chat"
    assert providers.build_embeddings(settings).model == "test-embeddings"


def test_embedding_fingerprint_changes_only_for_embedding_identity(settings):
    baseline = providers.embedding_fingerprint(settings)
    rotated = settings.model_copy(
        update={
            "embedding_api_key": SecretStr("rotated"),
            "chat_model": "another-chat",
        }
    )
    assert providers.embedding_fingerprint(rotated) == baseline
    assert (
        providers.embedding_fingerprint(
            settings.model_copy(
                update={
                    "embedding_model": "different-embeddings",
                }
            )
        )
        != baseline
    )
