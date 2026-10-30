"""PDF whitespace matching must preserve the exact original cited passage."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from research_qa.qa import QuestionAnswerer, match_original_quote


@pytest.mark.parametrize(
    ("original", "generated"),
    [
        ("Revenue was USD 120 million.", "Revenue was USD 120 million."),
        (
            "Services net sales increased due to the App Store\nand cloud services.",
            "Services net sales increased due to the App Store and cloud services.",
        ),
        ("Revenue\t\twas USD 120 million.", "Revenue was USD 120 million."),
        ("Products        307,003    294,866", "Products 307,003 294,866"),
        ("Revenue\u00a0was\u2003USD\u202f120 million.", "Revenue was USD 120 million."),
        ("Revenue was USD 120 million.", "Revenue\nwas\tUSD 120 million."),
    ],
)
def test_only_whitespace_equivalence_returns_exact_original(original, generated):
    source = "Before this passage. " + original + " After this passage."
    matched = match_original_quote(source, generated)
    assert matched == original
    assert matched in source


@pytest.mark.parametrize(
    "generated",
    [
        "Revenue was USD 121 million.",
        "Revenue was USD 120 million!",
        "revenue was USD 120 million.",
        "Revenue was 120 million.",
        "Revenue ... USD 120 million.",
        "Revenue was USD 120-million.",
        "Revenue was USD 120million.",
        "Revenue was USD 120 million. Invented fact.",
    ],
)
def test_changed_numbers_punctuation_case_and_skipped_words_are_rejected(generated):
    assert match_original_quote("Revenue\nwas USD 120 million.", generated) is None


@pytest.mark.parametrize("generated", ["", "1234567", " " * 8, "a" * 1001])
def test_input_quote_bounds_are_enforced(generated):
    assert match_original_quote(generated, generated) is None


def test_whitespace_expansion_cannot_create_an_unbounded_raw_quote():
    source = "Revenue" + " " * 1200 + "was USD 120 million."
    assert match_original_quote(source, "Revenue was USD 120 million.") is None


def test_match_can_skip_an_over_budget_occurrence_for_a_bounded_original_span():
    source = "Revenue" + " " * 1200 + "was USD 120 million. " + "Revenue\nwas USD 120 million."
    assert match_original_quote(source, "Revenue was USD 120 million.") == (
        "Revenue\nwas USD 120 million."
    )


def document(text, page, chunk):
    return Document(
        page_content=text,
        metadata={
            "document_name": "annual.pdf",
            "document_id": "doc",
            "page": page,
            "chunk_id": f"doc:{page}:{chunk}",
        },
    )


def generated_answer(quote, *, source_id="S1"):
    return AIMessage(
        content=json.dumps(
            {
                "abstained": False,
                "claims": [
                    {
                        "text": "Revenue was USD 120 million.",
                        "evidence": [{"source_id": source_id, "quote": quote}],
                    }
                ],
            }
        )
    )


def test_qa_citation_preserves_raw_pdf_whitespace_and_metadata(settings, chat):
    original = "Revenue\nwas USD 120\u00a0million."
    index = SimpleNamespace(search=lambda *_: [(document(original, 5, 2), 0.9)])
    chat.invoke.return_value = generated_answer("Revenue was USD 120 million.")
    response = QuestionAnswerer(settings, index, chat).ask("Revenue?")
    assert response.abstained is False
    assert response.citations[0].quote == original
    assert response.citations[0].page == 5
    assert response.citations[0].chunk_id == "doc:5:2"
    assert response.citations[0].source_id == "S1"


@pytest.mark.parametrize("same_page", [False, True])
def test_qa_never_joins_quotes_across_sources_or_pdf_pages(settings, chat, same_page):
    first = document("Revenue was", 1, 0)
    second = document("USD 120 million.", 1 if same_page else 2, 1)
    index = SimpleNamespace(search=lambda *_: [(first, 0.9), (second, 0.8)])
    chat.invoke.return_value = generated_answer("Revenue was USD 120 million.")
    response = QuestionAnswerer(settings, index, chat).ask("Revenue?")
    assert response.abstained is True
    assert response.reason == "unverifiable_citation"
    assert response.citations == []


def test_qa_unknown_source_id_still_fails_closed(settings, chat):
    index = SimpleNamespace(
        search=lambda *_: [(document("Revenue\nwas USD 120 million.", 1, 0), 0.9)]
    )
    chat.invoke.return_value = generated_answer("Revenue was USD 120 million.", source_id="S99")
    response = QuestionAnswerer(settings, index, chat).ask("Revenue?")
    assert response.abstained is True
    assert response.reason == "unverifiable_citation"
    assert response.citations == []
