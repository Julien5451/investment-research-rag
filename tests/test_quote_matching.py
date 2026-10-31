"""PDF whitespace matching must preserve the exact original cited passage."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from research_qa.qa import ABSTENTION, QuestionAnswerer, match_original_quote

# Synthetic revenue-driver passages, not quotations from an annual report.
DRIVERS = (
    "Revenue growth was driven by cloud\n  services.",
    "Subscription revenue increased due to higher\tpaid seats.",
)
INTERVENING = "Operating expenses increased during the year."


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


@pytest.mark.parametrize("same_source", [True, False])
def test_one_claim_accepts_separate_passages_and_preserves_originals(settings, chat, same_source):
    sources = (
        [document("\n".join((DRIVERS[0], INTERVENING, DRIVERS[1])), 5, 2)]
        if same_source
        else [document(DRIVERS[0], 5, 2), document(DRIVERS[1], 6, 3)]
    )
    source_ids = ["S1", "S1" if same_source else "S2"]
    index = SimpleNamespace(search=lambda *_: [(source, 0.9) for source in sources])
    claim = "Revenue grew due to cloud services and higher paid seats."
    chat.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "abstained": False,
                "claims": [
                    {
                        "text": claim,
                        "evidence": [
                            {"source_id": source_id, "quote": " ".join(passage.split())}
                            for source_id, passage in zip(source_ids, DRIVERS, strict=True)
                        ],
                    }
                ],
            }
        )
    )
    response = QuestionAnswerer(settings, index, chat).ask("What drove revenue growth?")
    assert response.abstained is False
    assert response.reason is None
    assert response.answer == claim + (" [S1]" if same_source else " [S1] [S2]")
    assert [citation.quote for citation in response.citations] == list(DRIVERS)
    assert [citation.source_id for citation in response.citations] == source_ids
    for citation, source in zip(response.citations, [sources[0], sources[-1]], strict=True):
        assert citation.quote in source.page_content
        assert {key: getattr(citation, key) for key in source.metadata} == source.metadata


@pytest.mark.parametrize("separator", [" ", " ... "])
def test_discontinuous_quote_in_second_claim_rejects_entire_answer(settings, chat, separator):
    source = "\n".join((DRIVERS[0], INTERVENING, DRIVERS[1]))
    quotes = [" ".join(passage.split()) for passage in DRIVERS]
    fabricated = separator.join(quotes)
    assert match_original_quote(source, fabricated) is None
    index = SimpleNamespace(search=lambda *_: [(document(source, 5, 2), 0.9)])
    chat.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "abstained": False,
                "claims": [
                    {
                        "text": "Revenue growth was driven by cloud services.",
                        "evidence": [{"source_id": "S1", "quote": quotes[0]}],
                    },
                    {
                        "text": "Revenue grew due to cloud services and higher paid seats.",
                        "evidence": [{"source_id": "S1", "quote": fabricated}],
                    },
                ],
            }
        )
    )
    response = QuestionAnswerer(settings, index, chat).ask("What drove revenue growth?")
    assert response.abstained is True
    assert response.reason == "unverifiable_citation"
    assert response.answer == ABSTENTION
    assert response.citations == []


@pytest.mark.parametrize("score", [None, 0.2])
def test_no_valid_retrieved_evidence_abstains_without_generation(settings, chat, score):
    results = [] if score is None else [(document(DRIVERS[0], 5, 2), score)]
    index = SimpleNamespace(search=lambda *_: results)
    response = QuestionAnswerer(settings, index, chat).ask("What drove revenue growth?")
    assert response.abstained is True
    assert response.reason == "no_relevant_evidence"
    assert response.answer == ABSTENTION
    assert response.citations == []
    chat.invoke.assert_not_called()


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
