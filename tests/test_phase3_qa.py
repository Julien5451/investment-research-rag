"""Regression cases derived from captured Phase 3 table diagnostics."""

import json
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from research_qa.qa import QuestionAnswerer

TABLE = (
    "Gross margin percentage:\n\n"
    "                               2025      2024      2023\n"
    "Services                       75.4%     73.9%     70.8%\n"
)
CASH = (
    "Cash, Cash Equivalents and Marketable Securities\n"
    "2025 (in millions)\n"
    "Cash  $ 28,267\n\n"
    "Level 1:\n\n"
    "Money market funds  5,272\n"
)


class FixedIndex:
    def __init__(self, *texts):
        self.documents = [
            Document(
                page_content=text,
                metadata={
                    "document_id": "public-report",
                    "document_name": "annual.pdf",
                    "page": number,
                    "chunk_id": f"public-report:{number}:0",
                },
            )
            for number, text in enumerate(texts, 1)
        ]

    def search(self, _question, top_k):
        return [(document, 0.8) for document in self.documents[:top_k]]


def ask(settings, texts, claim, quote, source="S1"):
    chat = MagicMock()
    chat.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "abstained": False,
                "claims": [{"text": claim, "evidence": [{"source_id": source, "quote": quote}]}],
            }
        )
    )
    return QuestionAnswerer(settings, FixedIndex(*texts), chat).ask("What was the 2025 value?")


def test_paraphrased_table_label_is_not_a_whitespace_equivalent_quote(settings):
    response = ask(
        settings,
        [TABLE],
        "Services gross margin percentage in FY2025 was 75.4%.",
        "Services gross margin percentage: 75.4%",
    )
    assert response.abstained is True
    assert response.reason == "unverifiable_citation"
    assert response.citations == []


def test_omitting_intervening_table_block_does_not_create_a_valid_quote(settings):
    response = ask(
        settings,
        [CASH],
        "The combined balance was USD 33.539 billion.",
        "Cash  $ 28,267\n\nMoney market funds  5,272",
    )
    assert response.abstained is True
    assert response.reason == "unverifiable_citation"


def test_financial_row_is_valid_only_in_the_selected_source(settings):
    response = ask(
        settings,
        [TABLE, "2025 net sales were USD 100 million."],
        "Services gross margin percentage in FY2025 was 75.4%.",
        "Services 75.4% 73.9% 70.8%",
        source="S2",
    )
    assert response.abstained is True
    assert response.reason == "unverifiable_citation"


def test_exact_financial_row_preserves_original_source_text(settings):
    response = ask(
        settings,
        [TABLE],
        "Services gross margin percentage in FY2025 was 75.4%.",
        "Services 75.4% 73.9% 70.8%",
    )
    assert response.abstained is False
    assert response.citations[0].quote == (
        "Services                       75.4%     73.9%     70.8%"
    )
    assert response.citations[0].page == 1
