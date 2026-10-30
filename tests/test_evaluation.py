"""Offline checks for metric definitions, provenance, and the real QA runner path."""

import hashlib
import json

import httpx
import pytest
from conftest import REVENUE, RISK, pdf_bytes, supported_answer
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from research_qa import evaluation


def gold(page=2, name="annual.pdf"):
    return {"document_name": name, "page": page, "quote": REVENUE}


def question(question_id="q1", *, answerable=True):
    return evaluation.EvaluationQuestion(
        id=question_id,
        question="What happened to revenue?" if answerable else "What are the emissions?",
        answerable=answerable,
        reference_answer=REVENUE if answerable else "Insufficient evidence.",
        evidence=[gold()] if answerable else [],
        required_patterns=[r"12\s*%", r"FY\s*2025"] if answerable else [],
        category="financial" if answerable else "unanswerable",
    )


def response(*, abstained=False):
    return {
        "answer": "Insufficient evidence." if abstained else REVENUE,
        "abstained": abstained,
        "reason": "model_insufficient_evidence" if abstained else None,
        "citations": [],
    }


def row(
    question_id,
    *,
    answerable=True,
    abstained=False,
    error=False,
    search_completed=True,
    recall=1.0,
    latency=1.0,
):
    final = None if error else response(abstained=abstained)
    return {
        "id": question_id,
        "category": "example",
        "question": "Revenue?",
        "reference_answer": REVENUE,
        "answerable": answerable,
        "response": final,
        "error_type": "ReadTimeout" if error else None,
        "automatic_match": not error and (abstained != answerable),
        "recall_at_3": recall if answerable else None,
        "page_hit_at_3": bool(recall) if answerable else None,
        "latency_seconds": latency,
        "retrieval_latency_seconds": latency / 10,
        "search_completed": search_completed,
    }


def metadata():
    return {
        "dataset_name": "Unit benchmark",
        "mode": "injected_test_clients",
        "status": "complete",
        "chat_model": "test-chat",
        "embedding_model": "test-embedding",
        "started_at_shanghai": "2026-10-30T12:00:00+08:00",
    }


def test_page_recall_deduplicates_pages_and_uses_only_top_three():
    evidence = [gold(2), gold(2), gold(3)]
    retrieved = [gold(2), gold(2), gold(2, name="other.pdf"), gold(3)]
    assert evaluation.page_recall_at_3(evidence, retrieved) == 0.5
    assert evaluation.page_recall_at_3([gold()], [gold(name="other.pdf")]) == 0
    assert evaluation.page_recall_at_3([], retrieved) is None


def test_regex_check_uses_answer_only_and_abstention_labels():
    assert evaluation.automatic_match(question(), response()) is True
    unrelated = response()
    unrelated["answer"] = "Revenue grew."
    unrelated["citations"] = [{"quote": REVENUE}]
    assert evaluation.automatic_match(question(), unrelated) is False
    assert evaluation.automatic_match(question(), response(abstained=True)) is False
    assert evaluation.automatic_match(question(answerable=False), response(abstained=True)) is True
    assert evaluation.automatic_match(question(answerable=False), response()) is False
    assert evaluation.automatic_match(question(), None) is False


def metric_rows():
    return [
        row("a1", latency=1),
        row("a2", abstained=True, recall=0, latency=2),
        row("u1", answerable=False, abstained=True, latency=3),
        row("u2", answerable=False, error=True, latency=10),
        row("a3", error=True, search_completed=False, recall=0, latency=11),
    ]


def test_metrics_keep_errors_out_of_refusals_and_in_denominators():
    summary = evaluation.summarize(metric_rows(), metadata())
    counts, metrics = summary["counts"], summary["metrics"]
    assert counts["errors"] == 2
    assert counts["abstained"] == 2
    assert counts["abstention_reasons"] == {"model_insufficient_evidence": 2}
    assert metrics["recall_at_3"] == pytest.approx(1 / 3)
    assert metrics["correct_refusal_rate"] == 0.5
    assert metrics["false_refusal_rate"] == pytest.approx(1 / 3)
    assert metrics["abstention_precision"] == 0.5
    assert metrics["automatic_answer_accuracy"] == 0.4
    assert metrics["automatic_answerable_accuracy"] == pytest.approx(1 / 3)
    assert metrics["answer_accuracy_method"] == "regex_preliminary"
    assert metrics["reviewed_accuracy"] is None
    assert metrics["latency_seconds"]["ask"] == {
        "count": 3,
        "mean": 2,
        "p50": 2,
        "p95": pytest.approx(2.9),
    }
    assert metrics["latency_seconds"]["retrieval"]["count"] == 4
    assert metrics["latency_seconds"]["failed_ask"]["count"] == 2


def test_absent_denominators_are_none_not_fabricated_zeroes():
    metrics = evaluation.summarize([], metadata())["metrics"]
    assert metrics["recall_at_3"] is None
    assert metrics["correct_refusal_rate"] is None
    assert metrics["false_refusal_rate"] is None
    assert metrics["abstention_precision"] is None
    assert metrics["answer_accuracy"] is None
    assert metrics["latency_seconds"]["ask"]["p95"] is None


def test_complete_agent_adjudication_replaces_preliminary_accuracy():
    rows = metric_rows()
    reviews = {item["id"]: {"correct": False, "note": "Checked against gold."} for item in rows}
    reviews["a1"]["correct"] = True
    summary = evaluation.summarize(rows, metadata(), reviews)
    assert summary["metrics"]["reviewed_accuracy"] == 0.2
    assert summary["metrics"]["answer_accuracy"] == 0.2
    assert summary["metrics"]["answer_accuracy_method"] == "agent_adjudicated"
    assert summary["counts"]["review_complete"] is True
    partial = evaluation.summarize(rows, metadata(), {"a1": reviews["a1"]})
    assert partial["metrics"]["reviewed_accuracy"] is None
    assert partial["metrics"]["reviewed_subset_accuracy"] == 1
    assert partial["counts"]["review_complete"] is False


@pytest.mark.parametrize(
    "reviews",
    [
        {"unknown": {"correct": True, "note": "No such question."}},
        {"a1": {"correct": 1, "note": "Boolean required."}},
        {"a1": {"correct": True, "note": " "}},
        {"a1": {"correct": True, "note": "Checked.", "unexpected": True}},
        {"u2": {"correct": True, "note": "Error is not a refusal."}},
    ],
)
def test_invalid_adjudications_are_rejected(reviews):
    with pytest.raises(ValueError):
        evaluation.summarize(metric_rows(), metadata(), reviews)


@pytest.mark.parametrize(
    "updates",
    [
        {"required_patterns": ["["]},
        {"required_patterns": []},
        {"evidence": []},
        {"evidence": [{"document_name": "annual.pdf", "page": 0, "quote": REVENUE}]},
        {"evidence": [{"document_name": "annual.pdf", "page": 2, "quote": " "}]},
        {"answerable": False},
    ],
)
def test_invalid_gold_labels_are_rejected(updates):
    with pytest.raises(ValidationError):
        evaluation.EvaluationQuestion.model_validate({**question().model_dump(), **updates})


def save_dataset(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    data = pdf_bytes("", REVENUE, RISK)
    (pdf_dir / "annual.pdf").write_bytes(data)
    dataset = {
        "schema_version": 1,
        "name": "Offline integration benchmark",
        "documents": [
            {
                "document_name": "annual.pdf",
                "url": "https://example.com/annual.pdf",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
        "questions": [
            question().model_dump(),
            question("q2", answerable=False).model_dump(),
            question("q3").model_dump(),
        ],
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    return dataset_path, pdf_dir, dataset


def test_runner_persists_actual_trace_and_summary_cli_needs_no_provider(
    tmp_path, settings, embeddings, chat, monkeypatch, capsys
):
    dataset_path, pdf_dir, _ = save_dataset(tmp_path)
    output = tmp_path / "run"
    chat.invoke.side_effect = [
        AIMessage(content=supported_answer()),
        httpx.ReadTimeout("secret-provider-body-must-never-be-saved"),
    ]
    summary = evaluation.run_benchmark(
        dataset_path,
        pdf_dir,
        output,
        settings,
        embeddings=embeddings,
        chat=chat,
    )
    assert summary["counts"]["questions"] == 3
    assert summary["counts"]["errors"] == 1
    assert summary["counts"]["correct_refusals"] == 1
    assert summary["run"]["mode"] == "injected_test_clients"
    assert summary["run"]["documents"][0]["pages"] == 3
    assert summary["run"]["documents"][0]["chunks"] == 2
    assert chat.invoke.call_count == 2
    assert settings.data_dir != output / "index"
    assert (output / "index/chroma/chroma.sqlite3").exists()
    rows = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
    assert rows[0]["retrieved"][0]["page"] == 2
    assert rows[0]["retrieved"][0]["text"] == REVENUE
    assert rows[0]["response"]["citations"][0]["page"] == 2
    assert rows[2]["response"] is None
    assert rows[2]["error_type"] == "ReadTimeout"
    assert rows[2]["recall_at_3"] == 1
    for path in output.glob("*.*"):
        assert "secret-provider-body-must-never-be-saved" not in path.read_text(
            encoding="utf-8-sig"
        )
    with pytest.raises(FileExistsError):
        evaluation.run_benchmark(
            dataset_path, pdf_dir, output, settings, embeddings=embeddings, chat=chat
        )
    reviews = {
        "q1": {"correct": True, "note": "Revenue and period match evidence."},
        "q2": {"correct": True, "note": "No emissions evidence in this corpus."},
        "q3": {"correct": False, "note": "Provider timed out, no answer."},
    }
    review_path = tmp_path / "reviews.json"
    review_path.write_text(json.dumps(reviews), encoding="utf-8")
    monkeypatch.setattr(evaluation, "build_chat", lambda _: pytest.fail("No API on summarize"))
    monkeypatch.setattr(
        evaluation, "build_embeddings", lambda _: pytest.fail("No API on summarize")
    )
    assert evaluation.main(["--summarize", str(output), "--review", str(review_path)]) == 1
    final = json.loads((output / "summary.json").read_text())
    assert final["metrics"]["answer_accuracy"] == pytest.approx(2 / 3)
    assert final["metrics"]["answer_accuracy_method"] == "agent_adjudicated"
    assert final["counts"]["review_complete"] is True
    assert (output / "results.csv").is_file()
    assert (output / "report.md").is_file()
    assert "secret-provider-body-must-never-be-saved" not in capsys.readouterr().out


@pytest.mark.parametrize("bad_kind", ["hash", "quote", "name", "duplicate_id"])
def test_invalid_dataset_fails_before_provider_calls(tmp_path, settings, monkeypatch, bad_kind):
    dataset_path, pdf_dir, data = save_dataset(tmp_path)
    if bad_kind == "hash":
        data["documents"][0]["sha256"] = "0" * 64
    elif bad_kind == "quote":
        data["questions"][0]["evidence"][0]["quote"] = "Fabricated fact not in PDF"
    elif bad_kind == "name":
        data["documents"][0]["document_name"] = "../annual.pdf"
    else:
        data["questions"][1]["id"] = data["questions"][0]["id"]
    dataset_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(evaluation, "build_chat", lambda _: pytest.fail("No API on bad gold"))
    monkeypatch.setattr(evaluation, "build_embeddings", lambda _: pytest.fail("No API on bad gold"))
    with pytest.raises(ValueError):
        evaluation.run_benchmark(dataset_path, pdf_dir, tmp_path / "run", settings)


def test_ingestion_failure_records_safe_failure_metadata(tmp_path, settings, embeddings, chat):
    dataset_path, pdf_dir, _ = save_dataset(tmp_path)

    class BrokenEmbeddings(type(embeddings)):
        def embed_documents(self, texts):
            raise httpx.ConnectError("secret-upstream-body")

    output = tmp_path / "failed"
    with pytest.raises(httpx.ConnectError):
        evaluation.run_benchmark(
            dataset_path, pdf_dir, output, settings, embeddings=BrokenEmbeddings(), chat=chat
        )
    saved = json.loads((output / "metadata.json").read_text())
    assert saved["status"] == "failed"
    assert saved["error_type"] == "ConnectError"
    assert "secret-upstream-body" not in (output / "metadata.json").read_text()
    assert json.loads((output / "summary.json").read_text())["counts"]["questions"] == 0
    chat.invoke.assert_not_called()
