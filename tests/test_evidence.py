"""Evidence labels distinguish supporting passages from page/value coincidences."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from research_qa.evidence import EvidenceLabels, assess_chunk, evidence_recall


def source(chunk_id, text, page=1):
    return {"chunk_id": chunk_id, "text": text, "document_name": "annual.pdf", "page": page}


@pytest.fixture
def example():
    chunks = [
        source("header", "FY2025 and FY2024 revenue (USD millions).", 1),
        source("row", "Services revenue 100 90", 1),
        source("irrelevant", "Revenue growth was driven by higher services demand.", 1),
        source("alternative", "FY2025 Services revenue was USD100million.", 8),
        source("reason", "Holiday demand explains the first-quarter increase.", 2),
    ]
    corpus = {
        chunk["chunk_id"]: {
            "document_name": chunk["document_name"],
            "page": chunk["page"],
            "text_sha256": hashlib.sha256(chunk["text"].encode()).hexdigest(),
        }
        for chunk in chunks
    }
    labels = {
        "schema_version": 1,
        "corpus": corpus,
        "questions": {
            "table": {
                "answerable": True,
                "requirements": [
                    {
                        "id": "amount_period_units",
                        "status": "verified",
                        "alternatives": [
                            {
                                "pieces": [
                                    {"chunk_id": "header", "excerpt": chunks[0]["text"]},
                                    {"chunk_id": "row", "excerpt": chunks[1]["text"]},
                                ]
                            },
                            {"pieces": [{"chunk_id": "alternative", "excerpt": chunks[3]["text"]}]},
                        ],
                    }
                ],
            },
            "multipart": {
                "answerable": True,
                "requirements": [
                    {
                        "id": "amount",
                        "status": "verified",
                        "alternatives": [
                            {"pieces": [{"chunk_id": "alternative", "excerpt": chunks[3]["text"]}]},
                        ],
                    },
                    {
                        "id": "reason",
                        "status": "verified",
                        "alternatives": [
                            {"pieces": [{"chunk_id": "reason", "excerpt": chunks[4]["text"]}]},
                        ],
                    },
                ],
            },
            "unknown_fact": {
                "answerable": True,
                "requirements": [
                    {"id": "uncertain", "status": "requires_review", "alternatives": []}
                ],
            },
            "unanswerable": {"answerable": False, "requirements": []},
        },
    }
    return labels, {chunk["chunk_id"]: chunk for chunk in chunks}


def test_same_page_does_not_imply_evidence_support(example):
    data, chunks = example
    result = EvidenceLabels(data).assess_chunk("table", chunks["irrelevant"])
    assert result["support_status"] == "not_supporting"
    assert result["evidence_recall"] == 0


def test_numeric_row_without_period_units_is_only_partial(example):
    data, chunks = example
    result = EvidenceLabels(data).assess_chunk("table", chunks["row"])
    assert result["support_status"] == "partial_support"
    assert result["full_support"] is False
    assert result["evidence_recall"] == 0


def test_header_and_value_form_verified_multichunk_bundle(example):
    data, chunks = example
    result = EvidenceLabels(data).evaluate("table", [chunks["row"], chunks["header"]])
    assert result["evidence_recall"] == 1
    assert result["full_support"] is True
    assert result["selected_multiple_chunks_needed"] is True
    assert result["minimum_support_chunks"] == 1  # Alternative narrative is one chunk.


def test_alternative_page_counts_without_gold_page_change(example):
    data, chunks = example
    result = EvidenceLabels(data).evaluate("table", [chunks["alternative"]])
    assert result["evidence_recall"] == 1
    assert result["selected_multiple_chunks_needed"] is False


def test_multipart_requirement_coverage_is_not_answer_correctness(example):
    data, chunks = example
    labels = EvidenceLabels(data)
    partial = labels.evaluate("multipart", [chunks["alternative"]])
    assert partial["evidence_recall"] == 0.5
    assert partial["full_support"] is False
    complete = labels.evaluate("multipart", [chunks["alternative"], chunks["reason"]])
    assert complete["evidence_recall"] == 1
    assert complete["selected_multiple_chunks_needed"] is True
    assert complete["multi_chunk_required"] is True


@pytest.mark.parametrize("mutation", ["number", "page", "document", "chunk_id"])
def test_changed_or_unfamiliar_source_requires_review(example, mutation):
    data, chunks = example
    modified = copy.deepcopy(chunks["alternative"])
    if mutation == "number":
        modified["text"] = modified["text"].replace("100", "999")
    else:
        modified[
            {"page": "page", "document": "document_name", "chunk_id": "chunk_id"}[mutation]
        ] = {
            "page": 9,
            "document": "private.pdf",
            "chunk_id": "unseen",
        }[mutation]
    result = EvidenceLabels(data).evaluate("table", [modified])
    assert result["label_status"] == "requires_review"
    assert result["evidence_recall"] is None
    assert result["full_support"] is None


def test_no_silent_positive_label_for_unreviewed_fact(example):
    data, chunks = example
    result = EvidenceLabels(data).evaluate("unknown_fact", [chunks["alternative"]])
    assert result["label_status"] == "requires_review"
    assert result["evidence_recall"] is None


def test_unanswerable_and_unknown_questions_have_no_recall_denominator(example):
    data, chunks = example
    labels = EvidenceLabels(data)
    assert labels.evaluate("unanswerable", [chunks["alternative"]])["evidence_recall"] is None
    assert (
        labels.evaluate("missing_question", [chunks["alternative"]])["label_status"]
        == "requires_review"
    )


def test_documents_tuples_and_trace_dicts_have_same_result(example):
    data, chunks = example
    value = chunks["alternative"]
    document = Document(
        page_content=value["text"],
        metadata={key: value[key] for key in ("chunk_id", "document_name", "page")},
    )
    labels = EvidenceLabels(data)
    expected = labels.evaluate("table", [value])
    assert labels.evaluate("table", [document]) == expected
    assert evidence_recall("table", [(document, 0.9)], labels) == expected
    assert assess_chunk("table", document, labels)["support_status"] == "verified_support"


@pytest.mark.parametrize(
    "defect", ["empty_requirement", "empty_alternative", "unknown_piece", "duplicate"]
)
def test_bad_evidence_labels_fail_loudly(example, defect):
    data, _ = example
    data = copy.deepcopy(data)
    requirements = data["questions"]["table"]["requirements"]
    if defect == "empty_requirement":
        data["questions"]["table"]["requirements"] = []
    elif defect == "empty_alternative":
        requirements[0]["alternatives"][0]["pieces"] = []
    elif defect == "unknown_piece":
        requirements[0]["alternatives"][0]["pieces"][0]["chunk_id"] = "unseen"
    else:
        requirements.append(copy.deepcopy(requirements[0]))
    with pytest.raises(ValueError):
        EvidenceLabels(data)


def test_repo_labels_preserve_frozen_dataset_and_cover_all_answerable_questions():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "eval/phase3/evidence_labels.json").read_text("utf-8"))
    frozen = (root / "eval/dataset.json").read_bytes()
    assert data["dataset_sha256"] == hashlib.sha256(frozen).hexdigest()
    assert len(data["corpus"]) == 467
    labels = EvidenceLabels(data)
    assert len([q for q in labels.questions.values() if q["answerable"]]) == 18
    assert all(q["status"] == "verified" for q in labels.questions.values() if q["answerable"])
    assert len(labels.questions["a17"]["requirements"][0]["alternatives"]) >= 3
    assert labels._minimum_chunks(labels.questions["a18"]["requirements"]) == 2


def test_diagnostics_record_threshold_loss_without_page_shortcut(example, tmp_path):
    data, chunks = example
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "retrieval_diagnostics", root / "scripts/retrieval_diagnostics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(data))
    trace = {
        "id": "table",
        "question": "FY2025 revenue?",
        "answerable": True,
        "evidence": [{"document_name": "annual.pdf", "page": 1}],
        "retrieved": [
            {**chunks["header"], "relevance_score": 0.8},
            {**chunks["row"], "relevance_score": 0.2},
            {**chunks["irrelevant"], "relevance_score": 0.7},
        ],
        "response": {"answer": "Insufficient evidence", "abstained": True},
    }
    responses = tmp_path / "responses.jsonl"
    responses.write_text(json.dumps(trace) + "\n")
    output = tmp_path / "diagnostics.csv"
    result = module.build_diagnostics(responses, labels_path, output, min_score=0.35)
    assert result["evidence_recall_at_3"] == 1
    assert result["evidence_recall_at_3_after_filter"] == 0
    assert result["filtered_by_threshold_count"] == 1
    assert "partial_support" in output.read_text(encoding="utf-8-sig")
    assert result["questions_detail"][0]["after_filtering"]["full_support"] is False
