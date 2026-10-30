"""Reproducible, isolated evaluation of the actual retrieval and QA pipeline.

No judge model or hidden fallback is used. Regex matches are preliminary labels;
complete agent adjudications can replace them without making any API requests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from research_qa.config import Settings
from research_qa.index import VectorIndex
from research_qa.ingestion import chunk_pages, extract_pages
from research_qa.providers import build_chat, build_embeddings, embedding_fingerprint
from research_qa.qa import SYSTEM_PROMPT, QuestionAnswerer


class GoldEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    document_name: str
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)

    @field_validator("quote")
    @classmethod
    def quote_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence quotes must not be blank")
        return value


class EvaluationQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000)
    answerable: bool
    reference_answer: str = Field(min_length=1)
    evidence: list[GoldEvidence]
    required_patterns: list[str]
    category: str = "general"

    @field_validator("required_patterns")
    @classmethod
    def valid_patterns(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value:
                raise ValueError("Patterns must not be empty")
            try:
                re.compile(value, flags=re.IGNORECASE)
            except re.error as exc:
                raise ValueError("Invalid answer regex") from exc
        return values

    @model_validator(mode="after")
    def consistent_labels(self) -> EvaluationQuestion:
        if self.answerable and (not self.evidence or not self.required_patterns):
            raise ValueError("Answerable questions need evidence and answer patterns")
        if not self.answerable and (self.evidence or self.required_patterns):
            raise ValueError("Unanswerable questions must have no evidence or answer patterns")
        if not self.question.strip():
            raise ValueError("Question must not be blank")
        return self


class EvaluationDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    document_name: str = Field(min_length=1)
    url: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    company: str | None = None
    fiscal_year: int | None = None
    pages: int | None = None
    source_page: str | None = None

    @field_validator("document_name")
    @classmethod
    def basename_only(cls, value: str) -> str:
        if "/" in value or "\\" in value or not value.lower().endswith(".pdf"):
            raise ValueError("Document names must be PDF basenames")
        return value


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: int
    name: str
    description: str | None = None
    documents: list[EvaluationDocument] = Field(min_length=1, max_length=2)
    questions: list[EvaluationQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_references(self) -> EvaluationDataset:
        if self.schema_version != 1:
            raise ValueError("Unsupported dataset schema")
        names = [doc.document_name for doc in self.documents]
        ids = [question.id for question in self.questions]
        if len(set(names)) != len(names) or len(set(ids)) != len(ids):
            raise ValueError("Duplicate document name or question ID")
        if any(
            evidence.document_name not in names
            for question in self.questions
            for evidence in question.evidence
        ):
            raise ValueError("Unknown evidence document")
        return self


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def page_recall_at_3(
    evidence: list[dict[str, Any]], retrieved: list[dict[str, Any]]
) -> float | None:
    """Recall of unique annotated (document, physical page) pairs in raw top-3.

    This is page-level recall, not exact-chunk recall. Duplicate chunks on a page
    do not increase the score. Unanswerable questions have no recall denominator.
    """
    gold = {(item["document_name"], item["page"]) for item in evidence}
    if not gold:
        return None
    found = {(item["document_name"], item["page"]) for item in retrieved[:3]}
    return len(gold & found) / len(gold)


def automatic_match(question: EvaluationQuestion, response: dict[str, Any] | None) -> bool:
    """Deterministic, deliberately preliminary answer check; not a semantic judge."""
    if response is None:
        return False
    if not question.answerable:
        return response.get("abstained") is True
    return response.get("abstained") is False and all(
        re.search(pattern, response["answer"], flags=re.IGNORECASE)
        for pattern in question.required_patterns
    )


class TracedIndex:
    """Capture the one real search that QA performs, without a duplicate API call."""

    def __init__(self, index: VectorIndex):
        self.index = index
        self.reset()

    def reset(self) -> None:
        self.results: list[tuple[Document, float]] = []
        self.latency_seconds: float | None = None
        self.search_completed = False

    def search(self, question: str, top_k: int) -> list[tuple[Document, float]]:
        started = perf_counter()
        try:
            self.results = self.index.search(question, top_k)
            self.search_completed = True
            return self.results
        finally:
            self.latency_seconds = perf_counter() - started


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], fraction: float) -> float | None:
    """Linear interpolation between sorted observations (NumPy's default method)."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _latencies(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
    }


def validate_reviews(
    rows: list[dict[str, Any]], reviews: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    if reviews is None:
        return {}
    if not isinstance(reviews, dict):
        raise ValueError("Reviews must map question IDs to {correct, note}")
    known = {row["id"] for row in rows}
    for question_id, review in reviews.items():
        if question_id not in known:
            raise ValueError("Review contains an unknown question ID")
        if (
            not isinstance(review, dict)
            or type(review.get("correct")) is not bool
            or not isinstance(review.get("note"), str)
            or not review["note"].strip()
            or set(review) != {"correct", "note"}
        ):
            raise ValueError("Every review needs boolean correct and a nonempty note")
    for row in rows:
        if row["error_type"] and reviews.get(row["id"], {}).get("correct"):
            raise ValueError("An API/internal error cannot be reviewed as a correct answer")
    return reviews


def summarize(
    rows: list[dict[str, Any]], metadata: dict[str, Any], reviews: dict[str, Any] | None = None
) -> dict[str, Any]:
    reviews = validate_reviews(rows, reviews)
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    successful = [row for row in rows if not row["error_type"]]
    abstained = [row for row in successful if row["response"]["abstained"]]
    false_refusals = [row for row in abstained if row["answerable"]]
    correct_refusals = [row for row in abstained if not row["answerable"]]
    recall_values = [row["recall_at_3"] for row in answerable]
    reviewed_correct = sum(review["correct"] for review in reviews.values())
    complete_review = bool(rows) and len(reviews) == len(rows)
    automatic_accuracy = _ratio(sum(row["automatic_match"] for row in rows), len(rows))
    reviewed_accuracy = _ratio(reviewed_correct, len(rows)) if complete_review else None
    metrics = {
        "recall_at_3": statistics.mean(recall_values) if recall_values else None,
        "page_hit_at_3": _ratio(sum(row["page_hit_at_3"] for row in answerable), len(answerable)),
        "automatic_answer_accuracy": automatic_accuracy,
        "automatic_answerable_accuracy": _ratio(
            sum(row["automatic_match"] for row in answerable), len(answerable)
        ),
        "reviewed_accuracy": reviewed_accuracy,
        "reviewed_answerable_accuracy": (
            _ratio(sum(reviews[row["id"]]["correct"] for row in answerable), len(answerable))
            if complete_review
            else None
        ),
        "reviewed_subset_accuracy": _ratio(reviewed_correct, len(reviews)),
        "answer_accuracy": reviewed_accuracy if complete_review else automatic_accuracy,
        "answer_accuracy_method": "agent_adjudicated" if complete_review else "regex_preliminary",
        "correct_refusal_rate": _ratio(len(correct_refusals), len(unanswerable)),
        "false_refusal_rate": _ratio(len(false_refusals), len(answerable)),
        "abstention_precision": _ratio(len(correct_refusals), len(abstained)),
        "error_rate": _ratio(len(rows) - len(successful), len(rows)),
        "latency_seconds": {
            "ask": _latencies([row["latency_seconds"] for row in successful]),
            "retrieval": _latencies(
                [row["retrieval_latency_seconds"] for row in rows if row["search_completed"]]
            ),
            "failed_ask": _latencies([row["latency_seconds"] for row in rows if row["error_type"]]),
        },
    }
    return {
        "schema_version": 1,
        "run": metadata,
        "counts": {
            "questions": len(rows),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "errors": len(rows) - len(successful),
            "abstained": len(abstained),
            "false_refusals": len(false_refusals),
            "correct_refusals": len(correct_refusals),
            "abstention_reasons": dict(Counter(row["response"]["reason"] for row in abstained)),
            "reviewed": len(reviews),
            "review_complete": complete_review,
        },
        "metrics": metrics,
        "methodology": {
            "recall_at_3": (
                "Macro-average page recall over answerable questions: unique annotated "
                "(document_name, physical 1-based page) pairs found in raw top-3 chunks / "
                "all annotated page pairs. It does not verify exact supporting passage. "
                "Search failures score zero; generation failures retain measured retrieval."
            ),
            "automatic_answer_accuracy": (
                "Preliminary regex check of final answer only, or explicit abstention for "
                "unanswerable questions. It does not establish semantic correctness."
            ),
            "reviewed_accuracy": (
                "Correct agent-adjudicated answers / all questions, including errors. "
                "Published only when every response has an explicit review note. Reviews "
                "are agent-assisted checks against gold answers, not independent human "
                "labels, and use no additional judge-model API calls."
            ),
            "refusals": (
                "Correct refusal rate = unanswerable explicit abstentions / all unanswerable; "
                "false refusal rate = answerable explicit abstentions / all answerable; "
                "abstention precision = correct explicit refusals / all explicit abstentions. "
                "Request errors never count as refusals and remain in rate denominators."
            ),
            "latency": (
                "Sequential single-run wall-clock seconds, no warmup excluded. Ask includes "
                "one query embedding/retrieval, generation when invoked, retries and final "
                "validation; excludes PDF indexing, HTTP and UI. Retrieval measures the same "
                "search inside ask, not a second search. p50/p95 use linear interpolation. "
                "Successful and failed ask latencies are reported separately."
            ),
            "scope": "Small curated development benchmark, not an independent held-out estimate.",
        },
    }


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_summary(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_dir: Path,
    reviews: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write machine and human-readable summaries; makes no API requests."""
    summary = summarize(rows, metadata, reviews)
    _json_write(output_dir / "summary.json", summary)
    if reviews is not None:
        _json_write(output_dir / "reviews.json", reviews)
    fields = [
        "id",
        "category",
        "question",
        "answerable",
        "reference_answer",
        "answer",
        "abstained",
        "reason",
        "recall_at_3",
        "automatic_match",
        "reviewed_correct",
        "review_note",
        "latency_seconds",
        "retrieval_latency_seconds",
        "error_type",
    ]
    with (output_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            response = row["response"] or {}
            review = (reviews or {}).get(row["id"], {})
            writer.writerow(
                {
                    **{key: row.get(key) for key in fields},
                    "answer": response.get("answer"),
                    "abstained": response.get("abstained"),
                    "reason": response.get("reason"),
                    "reviewed_correct": review.get("correct"),
                    "review_note": review.get("note"),
                }
            )
    metrics = summary["metrics"]
    lines = [
        "# Research QA evaluation",
        "",
        f"Dataset: {metadata['dataset_name']}",
        f"Mode: {metadata['mode']}; run status: {metadata['status']}",
        f"Chat: {metadata['chat_model']}; embeddings: {metadata['embedding_model']}",
        f"Started: {metadata['started_at_shanghai']}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "recall_at_3",
        "page_hit_at_3",
        "answer_accuracy",
        "answer_accuracy_method",
        "automatic_answer_accuracy",
        "automatic_answerable_accuracy",
        "reviewed_accuracy",
        "reviewed_answerable_accuracy",
        "correct_refusal_rate",
        "false_refusal_rate",
        "abstention_precision",
        "error_rate",
    ):
        lines.append(f"| {key} | {_display(metrics[key])} |")
    for stage, timing in metrics["latency_seconds"].items():
        for key in ("count", "mean", "p50", "p95"):
            lines.append(
                f"| {stage} latency {key} (seconds except count) | {_display(timing[key])} |"
            )
    lines.extend(
        [
            "",
            "## Counts",
            "",
            json.dumps(summary["counts"], ensure_ascii=False),
            "",
            "## Methodology",
            "",
        ]
    )
    for key, description in summary["methodology"].items():
        lines.extend([f"**{key}**: {description}", ""])
    lines.extend(
        [
            "## Per-question audit",
            "",
            "See `responses.jsonl` for raw retrieved chunks, final responses and timings; "
            "`results.csv` for a compact worksheet and `reviews.json` for adjudications.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_dataset(path: Path) -> EvaluationDataset:
    return EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


def _prepare_documents(
    dataset: EvaluationDataset, pdf_dir: Path, settings: Settings
) -> list[tuple[EvaluationDocument, list[Document], int]]:
    """Verify hashes and gold excerpts locally before invoking either provider."""
    prepared = []
    extracted: dict[tuple[str, int], str] = {}
    for document in dataset.documents:
        data = (pdf_dir / document.document_name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != document.sha256:
            raise ValueError("PDF checksum differs from dataset manifest")
        if len(data) > settings.max_upload_bytes or not data.startswith(b"%PDF-"):
            raise ValueError("Evaluation PDF violates application upload limits")
        pages, page_count = extract_pages(
            data,
            digest,
            document.document_name,
            max_pages=settings.max_pdf_pages,
            max_extracted_chars=settings.max_extracted_chars,
        )
        for page in pages:
            extracted[(document.document_name, page.metadata["page"])] = page.page_content
        prepared.append((document, pages, page_count))
    for question in dataset.questions:
        for evidence in question.evidence:
            page_text = extracted.get((evidence.document_name, evidence.page), "")
            if normalize_whitespace(evidence.quote) not in normalize_whitespace(page_text):
                raise ValueError(f"Gold excerpt not found on its annotated page: {question.id}")
    return prepared


def run_benchmark(
    dataset_path: Path,
    pdf_dir: Path,
    output_dir: Path,
    settings: Settings,
    *,
    limit: int | None = None,
    embeddings: Embeddings | None = None,
    chat: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Run the pipeline once per question, retaining every response and failure."""
    dataset = load_dataset(dataset_path)
    if limit is not None and limit < 1:
        raise ValueError("Question limit must be positive")
    if (embeddings is None) != (chat is None):
        raise ValueError("Inject both clients or neither")
    prepared = _prepare_documents(dataset, pdf_dir, settings)
    output_dir.mkdir(parents=True, exist_ok=False)
    isolated = settings.model_copy(
        update={
            "data_dir": output_dir / "index",
            "collection_name": "evaluation",
            "default_top_k": 3,
        }
    )
    metadata = {
        "dataset_name": dataset.name,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "mode": "live" if embeddings is None else "injected_test_clients",
        "status": "indexing",
        "started_at_utc": _utc_now(),
        "started_at_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "chat_provider": settings.chat_provider,
        "chat_model": settings.chat_model,
        "chat_host": urlsplit(settings.chat_base_url).hostname,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_host": urlsplit(settings.embedding_base_url).hostname,
        "embedding_fingerprint": embedding_fingerprint(settings),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "top_k": 3,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "min_relevance_score": settings.min_relevance_score,
        "max_context_chars": settings.max_context_chars,
        "request_timeout_seconds": settings.request_timeout_seconds,
        "question_limit": limit,
        "dataset_question_count": len(dataset.questions),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "versions": {
            package: version(package)
            for package in (
                "langchain-core",
                "langchain-openai",
                "langchain-chroma",
                "chromadb",
                "pymupdf",
            )
        },
        "documents": [],
    }
    rows: list[dict[str, Any]] = []
    _json_write(output_dir / "metadata.json", metadata)
    _json_write(output_dir / "dataset.json", dataset.model_dump())
    try:
        embedding_client = embeddings if embeddings is not None else build_embeddings(isolated)
        chat_client = chat if chat is not None else build_chat(isolated)
        index = VectorIndex(isolated, embedding_client)
        ingestion_started = perf_counter()
        for document, pages, page_count in prepared:
            chunks = chunk_pages(
                pages, chunk_size=isolated.chunk_size, chunk_overlap=isolated.chunk_overlap
            )
            index.add(chunks)
            metadata["documents"].append(
                {
                    **document.model_dump(),
                    "document_id": document.sha256,
                    "pages": page_count,
                    "chunks": len(chunks),
                }
            )
            print(
                f"Indexed {document.document_name}: {page_count} pages, {len(chunks)} chunks",
                flush=True,
            )
        metadata["ingestion_seconds"] = perf_counter() - ingestion_started
        metadata["status"] = "evaluating"
        _json_write(output_dir / "metadata.json", metadata)
        traced = TracedIndex(index)
        answerer = QuestionAnswerer(isolated, traced, chat_client)
        selected = dataset.questions[:limit] if limit is not None else dataset.questions
        with (output_dir / "responses.jsonl").open("w", encoding="utf-8") as file:
            for number, question in enumerate(selected, 1):
                traced.reset()
                started = perf_counter()
                response = None
                error_type = None
                try:
                    response = answerer.ask(question.question, top_k=3).model_dump()
                except Exception as exc:
                    # Provider messages may contain request bodies or credentials.
                    # Retain only the exception class; errors are never abstentions.
                    error_type = type(exc).__name__
                elapsed = perf_counter() - started
                retrieved = [
                    {
                        "document_name": doc.metadata["document_name"],
                        "page": doc.metadata["page"],
                        "chunk_id": doc.metadata["chunk_id"],
                        "relevance_score": score,
                        "text": doc.page_content,
                    }
                    for doc, score in traced.results
                ]
                evidence = [item.model_dump() for item in question.evidence]
                recall = page_recall_at_3(evidence, retrieved)
                row = {
                    **question.model_dump(),
                    "retrieved": retrieved,
                    "search_completed": traced.search_completed,
                    "recall_at_3": recall,
                    "page_hit_at_3": bool(recall and recall > 0) if question.answerable else None,
                    "response": response,
                    "automatic_match": automatic_match(question, response),
                    "error_type": error_type,
                    "latency_seconds": elapsed,
                    "retrieval_latency_seconds": traced.latency_seconds,
                }
                rows.append(row)
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                print(
                    f"[{number}/{len(selected)}] {question.id}: "
                    f"{'error ' + error_type if error_type else 'completed'} ({elapsed:.2f}s)",
                    flush=True,
                )
        metadata["status"] = "complete"
        metadata["finished_at_utc"] = _utc_now()
        _json_write(output_dir / "metadata.json", metadata)
        return write_summary(rows, metadata, output_dir)
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["error_type"] = type(exc).__name__
        metadata["finished_at_utc"] = _utc_now()
        _json_write(output_dir / "metadata.json", metadata)
        write_summary(rows, metadata, output_dir)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.json"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("eval/pdfs"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summarize", type=Path, metavar="OUTPUT_DIR")
    parser.add_argument("--review", type=Path)
    args = parser.parse_args(argv)
    if not args.summarize and args.output is None:
        parser.error("--output is required for a new run")
    if args.review and not args.summarize:
        parser.error("--review requires --summarize")
    try:
        if args.summarize:
            metadata = json.loads((args.summarize / "metadata.json").read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in (args.summarize / "responses.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            review_path = args.review or args.summarize / "reviews.json"
            if args.review and not review_path.exists():
                raise ValueError("Explicit review file does not exist")
            reviews = (
                json.loads(review_path.read_text(encoding="utf-8"))
                if review_path.exists()
                else None
            )
            summary = write_summary(rows, metadata, args.summarize, reviews)
        else:
            summary = run_benchmark(
                args.dataset, args.pdf_dir, args.output, Settings(), limit=args.limit
            )
        print(
            json.dumps(
                {"counts": summary["counts"], "metrics": summary["metrics"]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return (
            0 if summary["run"]["status"] == "complete" and not summary["counts"]["errors"] else 1
        )
    except Exception as exc:
        print(
            f"Evaluation failed ({type(exc).__name__}). Check configuration, dataset and "
            "the run's metadata.json; upstream messages are suppressed.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
