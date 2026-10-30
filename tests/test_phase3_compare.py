"""Synthetic offline checks for paired metrics, denominators and input immutability."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "phase3_compare", Path(__file__).resolve().parents[1] / "scripts/phase3_compare.py"
)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def evidence(status="verified"):
    return {
        "label_status": status,
        "evidence_recall": 1.0 if status == "verified" else None,
        "full_support": True if status == "verified" else None,
        "relevant_chunk_ids": ["c1"] if status == "verified" else [],
    }


def row(question_id, *, selected=False):
    answerable = question_id != "u1"
    text = (
        "Revenue increased by 12% in FY2025."
        if question_id == "a1"
        else "Risk is currency volatility."
    )
    final_refusal = question_id == "a2" and not selected or question_id == "u1" and selected
    failed = question_id == "u1" and not selected
    citation = {
        "source_id": "S1",
        "document_id": "public",
        "document_name": "annual.pdf",
        "page": 1 if question_id == "a1" else 2,
        "chunk_id": f"chunk-{question_id}",
        "quote": text,
        "relevance_score": 0.8,
    }
    answer = "Revenue increased by 99%." if question_id == "a1" and selected else text
    response = (
        None
        if failed
        else {
            "answer": "Insufficient evidence." if final_refusal else answer,
            "abstained": final_refusal,
            "reason": "unverifiable_citation"
            if question_id == "a2" and not selected
            else "model_insufficient_evidence"
            if final_refusal
            else None,
            "citations": [] if final_refusal else [citation],
        }
    )
    raw = {"abstained": final_refusal, "claims": []}
    if not final_refusal or question_id == "a2":
        raw["claims"] = [
            {
                "text": answer,
                "evidence": [
                    {
                        "source_id": "S1",
                        "quote": "Invented table label 12%." if final_refusal else text,
                    }
                ],
            }
        ]
    model_calls = (
        []
        if failed
        else [
            {
                "request": {"evidence": [{"source_id": "S1", "text": text}]},
                "content": json.dumps(raw),
            }
        ]
    )
    status = "requires_review" if question_id == "a2" and not selected else "verified"
    score = 1.0 if selected or question_id == "a1" else 0.0
    return {
        "id": question_id,
        "question": f"Question {question_id}",
        "answerable": answerable,
        "category": "test",
        "reference_answer": text if answerable else "Not disclosed.",
        "evidence": [{"document_name": "annual.pdf", "page": citation["page"], "quote": text}]
        if answerable
        else [],
        "required_patterns": ["12%"] if answerable else [],
        "response": response,
        "error_type": "ReadTimeout" if failed else None,
        "automatic_match": not failed and (not answerable or not final_refusal),
        "search_completed": True,
        "recall_at_3": score if answerable else None,
        "page_hit_at_3": bool(score) if answerable else None,
        "page_recall_at_k": score if answerable else None,
        "evidence_at_3": evidence(status) if answerable else None,
        "evidence_at_k": evidence(status) if answerable else None,
        "filtered_evidence_at_k": evidence(status) if answerable else None,
        "valid_citations": len((response or {}).get("citations", [])),
        "returned_citations": len((response or {}).get("citations", [])),
        "context_chars": 700 if selected else 400,
        "model_calls": model_calls,
        "latency_seconds": 20.0
        if failed
        else 1.0
        if question_id == "a1"
        else 2.0
        if question_id == "a2"
        else 3.0,
        "retrieval_latency_seconds": 0.1,
    }


def prepare(root, *, repeats=2, reviewed=True):
    for variant in ("baseline", "optimized"):
        selected = variant == "optimized"
        for repeat in range(1, repeats + 1):
            path = root / variant / f"run{repeat}"
            metadata = {
                "dataset_name": "Synthetic paired benchmark",
                "mode": "injected_test_clients",
                "status": "complete",
                "repeat": repeat,
                "started_at_shanghai": "2026-10-30T12:00:00+08:00",
                "chat_model": "test-chat",
                "embedding_model": "test-embeddings",
                "dataset_sha256": "0" * 64,
                "qa_source_sha256": variant,
                "prompt_sha256": variant,
                "retrieval_method": "dense",
                "top_k": 5 if selected else 3,
                "max_context_chars": 16000,
                "min_relevance_score": 0.35,
            }
            write(path / "metadata.json", metadata)
            rows = [row(question_id, selected=selected) for question_id in ("a1", "a2", "u1")]
            (path / "responses.jsonl").write_text(
                "\n".join(json.dumps(item) for item in rows), encoding="utf-8"
            )
            if reviewed:
                write(
                    path / "reviews.json",
                    {
                        item["id"]: {
                            "correct": (item["id"] != "a1" if selected else item["id"] == "a1"),
                            "note": "Synthetic explicit adjudication.",
                        }
                        for item in rows
                    },
                )


def test_pooled_comparison_preserves_raw_inputs_and_reports_all_question_pairs(tmp_path):
    root = tmp_path / "eval/phase3"
    prepare(root)
    originals = {
        path: path.read_bytes()
        for pattern in ("*/run*/responses.jsonl", "*/run*/reviews.json")
        for path in root.glob(pattern)
    }
    summaries = compare.aggregate(root)
    baseline, optimized = summaries["baseline"], summaries["optimized"]
    assert baseline["counts"]["questions"] == optimized["counts"]["questions"] == 6
    assert baseline["counts"]["unique_questions"] == 3
    assert baseline["counts"]["paired_repetitions"] == 2
    assert baseline["metrics"]["reviewed_accuracy"] == pytest.approx(1 / 3)
    assert optimized["metrics"]["reviewed_accuracy"] == pytest.approx(2 / 3)
    assert baseline["metrics"]["false_refusal_rate"] == 0.5
    assert optimized["metrics"]["false_refusal_rate"] == 0
    assert baseline["metrics"]["correct_refusal_rate"] == 0
    assert optimized["metrics"]["correct_refusal_rate"] == 1
    assert baseline["counts"]["errors"] == 2
    assert baseline["metrics"]["evidence_recall_at_3"] == 1
    assert baseline["metrics"]["evidence_label_coverage"]["at_3"] == {
        "verified_question_trials": 2,
        "answerable_question_trials": 4,
        "coverage": 0.5,
    }
    assert optimized["metrics"]["evidence_label_coverage"]["at_3"]["coverage"] == 1
    assert baseline["metrics"]["raw_model_quote_validity"] == 0.5
    assert optimized["metrics"]["raw_model_quote_validity"] == 1
    assert baseline["metrics"]["returned_citation_validity"] == 1
    assert optimized["metrics"]["returned_citation_validity"] == 1
    assert optimized["metrics"]["reviewed_accuracy"] < 1  # Provenance is not truth.
    assert baseline["metrics"]["context_characters"]["mean"] == 400
    assert optimized["metrics"]["context_characters"]["mean"] == 700
    assert len(baseline["per_run"]) == 2
    with (root / "paired_comparison.csv").open(encoding="utf-8-sig", newline="") as file:
        pairs = list(csv.DictReader(file))
    assert len(pairs) == 6
    assert {item["question_id"] for item in pairs} == {"a1", "a2", "u1"}
    assert [item["outcome"] for item in pairs].count("regressed") == 2
    assert [item["outcome"] for item in pairs].count("fixed") == 4
    report = (root / "comparison.md").read_text(encoding="utf-8")
    assert "not a controlled single-variable" in report
    assert "At least one paired regressed trial: a1" in report
    assert "Top-5" in report
    assert "semantic entailment" in report
    for path, original in originals.items():
        assert path.read_bytes() == original
    assert not (root / "baseline/reviews.json").exists()
    assert not (root / "optimized/reviews.json").exists()
    aggregate_csv = (root / "baseline/results.csv").read_text(encoding="utf-8-sig")
    assert "run1:a1" in aggregate_csv and "run2:a1" in aggregate_csv


def test_missing_reviews_require_explicit_preliminary_mode(tmp_path):
    prepare(tmp_path, repeats=1, reviewed=False)
    with pytest.raises(ValueError, match="review for every"):
        compare.aggregate(tmp_path)
    assert not (tmp_path / "comparison.md").exists()
    summaries = compare.aggregate(tmp_path, allow_preliminary=True)
    assert summaries["baseline"]["metrics"]["reviewed_accuracy"] is None
    assert "PRELIMINARY" in (tmp_path / "comparison.md").read_text()


@pytest.mark.parametrize(
    "field,value",
    [
        ("chat_model", "different-model"),
        ("embedding_model", "different-embedding"),
        ("max_context_chars", 8000),
        ("min_relevance_score", 0.1),
        ("dataset_sha256", "1" * 64),
    ],
)
def test_incompatible_run_metadata_is_rejected_before_outputs(tmp_path, field, value):
    prepare(tmp_path, repeats=1)
    path = tmp_path / "optimized/run1/metadata.json"
    metadata = compare.read_json(path)
    metadata[field] = value
    write(path, metadata)
    with pytest.raises(ValueError, match="differ across paired"):
        compare.aggregate(tmp_path)
    assert not (tmp_path / "comparison.md").exists()


def test_mismatched_frozen_gold_is_not_silently_paired(tmp_path):
    prepare(tmp_path, repeats=1)
    path = tmp_path / "optimized/run1/responses.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["reference_answer"] = "Changed after seeing results"
    path.write_text("\n".join(json.dumps(item) for item in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="gold labels differ"):
        compare.aggregate(tmp_path)


def test_zero_or_unavailable_citation_denominators_remain_unavailable():
    no_quotes = row("u1", selected=True)
    metrics = compare.extended_metrics([no_quotes])
    assert metrics["returned_citation_validity"] is None
    assert metrics["raw_model_quote_validity"] is None
    historical = row("a1")
    historical.pop("model_calls")
    assert compare.extended_metrics([historical])["raw_model_quote_counts"]["available"] is False
    malformed = row("a1")
    malformed["model_calls"][0]["content"] = "not JSON"
    counts = compare.raw_quote_counts([malformed])
    assert counts["unparseable_generations"] == 1
    assert counts["total"] == 0


def test_historical_reference_is_read_only_and_missing_raw_quotes_are_not_invented(tmp_path):
    root = tmp_path / "eval/phase3"
    prepare(root, repeats=1)
    previous = tmp_path / "eval/results/latest"
    write(
        previous / "summary.json",
        {
            "metrics": {
                "answer_accuracy": 17 / 24,
                "recall_at_3": 13 / 18,
            }
        },
    )
    write(previous / "reviews.json", {"a1": {"correct": True}, "a2": {"correct": False}})
    original = (previous / "summary.json").read_bytes()
    compare.aggregate(root)
    report = (root / "comparison.md").read_text(encoding="utf-8")
    assert "Historical raw model quote validity: **unavailable**" in report
    assert "incorrect trial: a1" in report
    assert (previous / "summary.json").read_bytes() == original


def test_holdout_scope_and_absent_evidence_labels_are_reported_separately(tmp_path):
    root = tmp_path / "holdout"
    prepare(root, repeats=1)
    for path in root.glob("*/run1/responses.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for item in rows:
            for field in ("evidence_at_3", "evidence_at_k", "filtered_evidence_at_k"):
                item[field] = None
        path.write_text("\n".join(json.dumps(item) for item in rows), encoding="utf-8")
    summaries = compare.aggregate(root)
    for summary in summaries.values():
        assert summary["run"]["evaluation_scope"] == "holdout"
        assert "holdout" in summary["methodology"]["scope"]
        assert summary["metrics"]["evidence_recall_at_3"] is None
        assert summary["metrics"]["evidence_recall_at_k"] is None
        assert summary["metrics"]["evidence_label_coverage"]["at_3"] == {
            "verified_question_trials": 0,
            "answerable_question_trials": 2,
            "coverage": 0,
        }
    report = (root / "comparison.md").read_text()
    assert report.startswith("# Phase 3 independent-company holdout evaluation")
    assert "Evidence Recall@3 | N/A | N/A | N/A" in report
    assert "small development set" not in report
    assert "same fixed development questions" not in json.dumps(summaries)


def test_blank_raw_quotes_are_not_counted_as_exact_valid_quotations():
    captured = {
        "model_calls": [
            {
                "request": {"evidence": [{"source_id": "S1", "text": "Revenue        12%"}]},
                "content": json.dumps(
                    {
                        "claims": [
                            {
                                "evidence": [
                                    {
                                        "source_id": "S1",
                                        "quote": " " * 8,
                                    }
                                ]
                            }
                        ]
                    }
                ),
            }
        ]
    }
    counts = compare.raw_quote_counts([captured])
    assert counts["total"] == 1
    assert counts["valid"] == counts["exact"] == 0
