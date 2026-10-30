"""Optional retrieval ranking is offline, deterministic and source-preserving."""

import math
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from research_qa.index import VectorIndex
from research_qa.retrieval import ExperimentalRetriever, tokenize


class FixedEmbeddings(Embeddings):
    def __init__(self, vectors):
        self.vectors = vectors
        self.query_calls = 0

    def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return [1.0, 0.0]


def make_index(settings, texts_and_vectors):
    embeddings = FixedEmbeddings({text: vector for _, text, vector in texts_and_vectors})
    index = VectorIndex(settings, embeddings)
    index.add(
        [
            Document(
                page_content=text,
                metadata={
                    "document_id": "original",
                    "document_name": "annual.pdf",
                    "page": page,
                    "chunk_id": chunk_id,
                    "start_index": page * 100,
                },
            )
            for page, (chunk_id, text, _) in enumerate(texts_and_vectors, 1)
        ]
    )
    return index, embeddings


def test_tokenizer_explicit_english_chinese_and_numbers():
    assert tokenize("Revenue FY2025: $166,000 and 75.4%; 苹果年报") == [
        "revenue",
        "fy",
        "2025",
        "166,000",
        "and",
        "75.4",
        "苹",
        "果",
        "年",
        "报",
        "苹果",
        "果年",
        "年报",
    ]
    assert tokenize("员工 count 员工") == ["员", "工", "员工", "count", "员", "工", "员工"]
    assert tokenize("— % $ / ") == []


def test_dense_matches_existing_index_search(settings):
    index, embeddings = make_index(
        settings,
        [
            ("a", "Revenue increased", [1.0, 0.0]),
            ("b", "Risk discussion", [0.6, 0.8]),
            ("c", "Other disclosures", [-1.0, 0.0]),
        ],
    )
    retriever = ExperimentalRetriever(index)
    actual = retriever.search("Revenue?", 3)
    expected = index.search("Revenue?", 3)
    assert [(doc.metadata, doc.page_content) for doc, _ in actual] == [
        (doc.metadata, doc.page_content) for doc, _ in expected
    ]
    assert [score for _, score in actual] == pytest.approx([score for _, score in expected])
    assert [trace["dense_rank"] for trace in retriever.last_trace] == [1, 2, 3]
    assert all(trace["rrf_score"] is None for trace in retriever.last_trace)
    assert actual[-1][1] == 0.0
    assert embeddings.query_calls == 2


def test_search_vector_never_reembeds_documents_or_question(settings):
    index, embeddings = make_index(settings, [("a", "Employee count 166,000", [1.0, 0.0])])
    for method in ("dense", "hybrid"):
        retriever = ExperimentalRetriever(index, method)
        result = retriever.search_vector([1.0, 0.0], "Employee count", 5)
        assert len(result) == 1
    assert embeddings.query_calls == 0


def test_hybrid_can_promote_sparse_candidate_but_preserves_cosine_filter_score(settings):
    index, _ = make_index(
        settings,
        [
            ("a", "Unrelated financial discussion", [1.0, 0.0]),
            ("b", "quasar employees 166,000", [0.2, math.sqrt(0.96)]),
        ],
    )
    retriever = ExperimentalRetriever(index, "hybrid", candidate_k=1)
    actual = retriever.search_vector([1.0, 0.0], "quasar", 2)
    traces = {item["chunk_id"]: item for item in retriever.last_trace}
    assert [doc.metadata["chunk_id"] for doc, _ in actual] == ["a", "b"]
    assert traces["b"]["dense_rank"] is None
    assert traces["b"]["bm25_rank"] == 1
    assert traces["b"]["rrf_score"] == pytest.approx(1 / 61)
    scores = {doc.metadata["chunk_id"]: score for doc, score in actual}
    assert scores["b"] == pytest.approx(0.2)
    # RRF reorders candidates, but a .35 relevance gate must still reject this one.
    assert [doc.metadata["chunk_id"] for doc, score in actual if score >= 0.35] == ["a"]


def test_rrf_uses_both_rankings_and_chunk_id_ties_are_deterministic(settings):
    index, _ = make_index(
        settings,
        [
            ("z", "Employees shared", [1.0, 0.0]),
            ("a", "Employees shared elsewhere", [0.8, 0.6]),
            ("b", "No lexical overlap", [0.9, math.sqrt(0.19)]),
        ],
    )
    retriever = ExperimentalRetriever(index, "hybrid", candidate_k=3)
    result = retriever.search_vector([1.0, 0.0], "employees", 3)
    assert result[0][0].metadata["chunk_id"] == "z"
    assert retriever.last_trace[0]["rrf_score"] == pytest.approx(2 / 61)
    assert retriever.last_trace[1]["chunk_id"] == "a"

    # Synthetic ranks deliberately tie: dense nominates z and sparse nominates a.
    retriever = ExperimentalRetriever(index, "hybrid", candidate_k=1)
    retriever.refresh()
    retriever._dense = MagicMock(return_value=[("z", result[0][0], 1.0)])
    retriever._bm25 = MagicMock(return_value={"a": 10.0})
    orders = [
        [doc.metadata["chunk_id"] for doc, _ in retriever.search_vector([1.0, 0.0], "query", 2)]
        for _ in range(4)
    ]
    assert orders == [["a", "z"]] * 4


def test_metadata_text_ids_unchanged_and_external_output_mutation_is_not_cached(settings):
    index, _ = make_index(settings, [("original:7:0", "Revenue\n  42 million.", [1.0, 0.0])])
    before = deepcopy(index.collection.get(include=["documents", "metadatas"]))
    retriever = ExperimentalRetriever(index, "hybrid")
    first = retriever.search_vector([1.0, 0.0], "Revenue", 1)
    assert first[0][0].id == "original:7:0"
    first[0][0].metadata["page"] = 999
    first[0][0].page_content = "modified by caller"
    second = retriever.search_vector([1.0, 0.0], "Revenue", 1)
    assert second[0][0].metadata["page"] == 1
    assert second[0][0].page_content == "Revenue\n  42 million."
    assert index.collection.get(include=["documents", "metadatas"]) == before


def test_hybrid_refreshes_after_new_upload_changes_count(settings):
    index, embeddings = make_index(settings, [("a", "old unrelated disclosure", [1.0, 0.0])])
    retriever = ExperimentalRetriever(index, "hybrid")
    assert len(retriever.search_vector([1.0, 0.0], "newfact", 5)) == 1
    embeddings.vectors["newfact employees"] = [0.8, 0.6]
    index.add(
        [
            Document(
                page_content="newfact employees",
                metadata={
                    "chunk_id": "b",
                    "document_id": "new",
                    "document_name": "next.pdf",
                    "page": 9,
                },
            )
        ]
    )
    output = retriever.search_vector([1.0, 0.0], "newfact", 5)
    assert retriever._corpus_count == 2
    assert retriever.last_trace[0]["bm25_rank"] == 1
    assert output[0][0].metadata["chunk_id"] == "b"


def test_chinese_sparse_matching_is_explicitly_lexical_not_translation(settings):
    index, _ = make_index(
        settings,
        [
            ("a", "Employees and revenue", [1.0, 0.0]),
            ("b", "员工人数为166000", [0.8, 0.6]),
        ],
    )
    retriever = ExperimentalRetriever(index, "hybrid")
    actual = retriever.search_vector([1.0, 0.0], "员工", 2)
    assert actual[0][0].metadata["chunk_id"] == "b"
    assert "a" not in retriever._bm25("员工")


@pytest.mark.parametrize("method", ["dense", "hybrid"])
def test_empty_corpus_does_not_call_embedding_api(settings, method):
    embeddings = FixedEmbeddings({})
    index = VectorIndex(settings, embeddings)
    retriever = ExperimentalRetriever(index, method)
    assert retriever.search("anything", 3) == []
    assert retriever.last_trace == []
    assert embeddings.query_calls == 0


@pytest.mark.parametrize("query", [[], [0.0, 0.0], [float("nan"), 1], [float("inf"), 1]])
def test_invalid_vectors_rejected(settings, query):
    index, _ = make_index(settings, [("a", "Revenue", [1.0, 0.0])])
    with pytest.raises(ValueError, match="embedding"):
        ExperimentalRetriever(index).search_vector(query, "Revenue", 1)


@pytest.mark.parametrize(
    "option,value",
    [
        ("method", "unknown"),
        ("candidate_k", 0),
        ("rrf_k", -1),
        ("candidate_k", True),
    ],
)
def test_invalid_configuration_rejected(settings, option, value):
    index, _ = make_index(settings, [("a", "Revenue", [1.0, 0.0])])
    with pytest.raises(ValueError):
        ExperimentalRetriever(index, **{option: value})


def test_invalid_top_k_rejected(settings):
    index, _ = make_index(settings, [("a", "Revenue", [1.0, 0.0])])
    retriever = ExperimentalRetriever(index)
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("Revenue", 0)
