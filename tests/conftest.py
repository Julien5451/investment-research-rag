"""Offline fixtures: generated PDFs, deterministic embeddings and mocked chat."""

import json
from unittest.mock import MagicMock

import pymupdf
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from research_qa.config import Settings

REVENUE = "Revenue increased by 12% in FY2025."
RISK = "The principal risk is foreign exchange volatility."


class TopicEmbeddings(Embeddings):
    """Tiny deterministic vectors make retrieval assertions independent of APIs."""

    @staticmethod
    def vector(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float("revenue" in lowered),
            float("risk" in lowered),
            float("emissions" in lowered),
            0.01,
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vector(text)


def pdf_bytes(*texts: str, encrypted: bool = False) -> bytes:
    with pymupdf.open() as pdf:
        for text in texts:
            page = pdf.new_page()
            if text:
                page.insert_textbox((50, 50, 550, 790), text, fontsize=11)
        if encrypted:
            return pdf.tobytes(
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner-password",
                user_pw="reader-password",
            )
        return pdf.tobytes()


def supported_answer(quote: str = REVENUE, *, source: str = "S1") -> str:
    return json.dumps(
        {
            "abstained": False,
            "claims": [{"text": quote, "evidence": [{"source_id": source, "quote": quote}]}],
        }
    )


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        embedding_provider="openai-compatible",
        embedding_base_url="http://embeddings.invalid/v1",
        embedding_model="test-embeddings",
        chat_provider="openai-compatible",
        chat_base_url="http://chat.invalid/v1",
        chat_model="test-chat",
        min_relevance_score=0.6,
    )


@pytest.fixture
def embeddings():
    return TopicEmbeddings()


@pytest.fixture
def chat():
    model = MagicMock()
    model.invoke.return_value = AIMessage(content=supported_answer())
    return model
