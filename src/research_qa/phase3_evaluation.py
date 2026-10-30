"""Paired live evaluation using frozen Phase 2 QA and a selected Phase 3 variant.

This reuses the existing evaluator's dataset and scoring contracts. It never
rewrites Phase 2 artifacts. Raw generations are retained to audit rejected quotes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from research_qa.config import Settings
from research_qa.evaluation import automatic_match, load_dataset, write_summary
from research_qa.evidence import EvidenceLabels
from research_qa.index import VectorIndex
from research_qa.providers import build_chat, build_embeddings
from research_qa.qa import INJECTION
from research_qa.retrieval import ExperimentalRetriever


def load_qa(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load frozen QA implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CapturedChat:
    def __init__(self, model):
        self.model = model
        self.calls: list[dict] = []

    def invoke(self, messages):
        started = perf_counter()
        message = self.model.invoke(messages)
        try:
            request = json.loads(messages[1].content)
        except (ValueError, IndexError, TypeError):
            request = {"content": messages[-1].content}
        self.calls.append(
            {
                "request": request,
                "content": message.content,
                "elapsed_seconds": perf_counter() - started,
                "usage": message.usage_metadata,
                "resolved_model": message.response_metadata.get("model_name"),
            }
        )
        return message


class CapturedRetrieval:
    def __init__(self, retriever):
        self.retriever = retriever
        self.results = []
        self.elapsed = None
        self.completed = False

    def search(self, question: str, top_k: int):
        started = perf_counter()
        try:
            self.results = self.retriever.search(question, top_k)
            self.completed = True
            return self.results
        finally:
            self.elapsed = perf_counter() - started


def page_recall(question, results, k: int):
    gold = {(e.document_name, e.page) for e in question.evidence}
    found = {(doc.metadata["document_name"], doc.metadata["page"]) for doc, _ in results[:k]}
    return len(gold & found) / len(gold) if gold else None


def citation_counts(response: dict | None, results: list) -> tuple[int, int]:
    """Check returned source identity and contiguous raw quotation, not entailment."""
    sources = {doc.metadata["chunk_id"]: doc for doc, _ in results}
    citations = (response or {}).get("citations", [])
    valid = 0
    for citation in citations:
        source = sources.get(citation["chunk_id"])
        if (
            source is not None
            and citation["quote"] in source.page_content
            and all(
                citation[key] == source.metadata[key]
                for key in ("document_id", "document_name", "page", "chunk_id")
            )
        ):
            valid += 1
    return valid, len(citations)


def run_question(question, module, retriever, chat, settings, labels, top_k):
    captured_chat = CapturedChat(chat)
    captured_index = CapturedRetrieval(retriever)
    started = perf_counter()
    response, error = None, None
    try:
        response = (
            module.QuestionAnswerer(settings, captured_index, captured_chat)
            .ask(question.question, top_k=top_k)
            .model_dump()
        )
    except Exception as exc:
        error = type(exc).__name__
    elapsed = perf_counter() - started
    results = captured_index.results
    filtered = [
        (doc, score)
        for doc, score in results
        if score >= settings.min_relevance_score and not INJECTION.search(doc.page_content)
    ]
    valid, total = citation_counts(response, results)
    return {
        **question.model_dump(),
        "response": response,
        "error_type": error,
        "automatic_match": automatic_match(question, response),
        "latency_seconds": elapsed,
        "retrieval_latency_seconds": captured_index.elapsed,
        "search_completed": captured_index.completed,
        "recall_at_3": page_recall(question, results, 3),
        "page_hit_at_3": bool(page_recall(question, results, 3)) if question.answerable else None,
        "page_recall_at_k": page_recall(question, results, top_k),
        "evidence_at_3": labels.evaluate(question.id, results[:3]) if labels else None,
        "evidence_at_k": labels.evaluate(question.id, results) if labels else None,
        "filtered_evidence_at_k": labels.evaluate(question.id, filtered) if labels else None,
        "valid_citations": valid,
        "returned_citations": total,
        "context_chars": min(
            settings.max_context_chars, sum(len(doc.page_content) for doc, _ in filtered)
        ),
        "model_calls": captured_chat.calls,
        "ranking_trace": retriever.last_trace,
        "retrieved": [
            {**doc.metadata, "text": doc.page_content, "relevance_score": score}
            for doc, score in results
        ],
    }


def json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_paired(
    dataset_path: Path,
    index_dir: Path,
    output: Path,
    selection: dict,
    repeats: int,
    *,
    embeddings=None,
    chat=None,
    labels_path=None,
):
    dataset = load_dataset(dataset_path)
    settings = Settings().model_copy(
        update={
            "data_dir": index_dir,
            "collection_name": "evaluation",
            "default_top_k": 3,
            "min_relevance_score": selection["min_relevance_score"],
            "max_context_chars": selection["max_context_chars"],
        }
    )
    embeddings = embeddings if embeddings is not None else build_embeddings(settings)
    live = chat is None
    chat = chat if chat is not None else build_chat(settings)
    index = VectorIndex(settings, embeddings)
    labels = EvidenceLabels(labels_path) if labels_path else None
    baseline_module = load_qa(output / "frozen/qa_phase2.py", "phase3_baseline_qa")
    optimized_module = load_qa(output / "frozen/qa_selected.py", "phase3_selected_qa")
    retrieval = {
        "baseline": ExperimentalRetriever(index, method="dense"),
        "optimized": ExperimentalRetriever(index, method=selection["retrieval_method"]),
    }
    # Build BM25 before timed questions; separately recorded as startup cost.
    startup_cost = {}
    for name, retriever in retrieval.items():
        start = perf_counter()
        if retriever.method == "hybrid":
            retriever.refresh()
        startup_cost[name] = perf_counter() - start
    for repeat in range(1, repeats + 1):
        rows = {name: [] for name in retrieval}
        paths = {name: output / name / f"run{repeat}" for name in retrieval}
        metadata = {}
        for name, path in paths.items():
            path.mkdir(parents=True, exist_ok=False)
            source_path = (
                output / "frozen" / ("qa_phase2.py" if name == "baseline" else "qa_selected.py")
            )
            qa_module = baseline_module if name == "baseline" else optimized_module
            metadata[name] = {
                "dataset_name": dataset.name,
                "mode": "live" if live else "injected_test_clients",
                "status": "evaluating",
                "repeat": repeat,
                "started_at_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "started_at_utc": datetime.now(UTC).isoformat(),
                "chat_model": settings.chat_model,
                "embedding_model": settings.embedding_model,
                "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                "qa_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "prompt_sha256": hashlib.sha256(qa_module.SYSTEM_PROMPT.encode()).hexdigest(),
                "documents": [doc.model_dump() for doc in dataset.documents],
                "retrieval_method": retrieval[name].method,
                "top_k": 3 if name == "baseline" else selection["top_k"],
                "max_context_chars": settings.max_context_chars,
                "min_relevance_score": settings.min_relevance_score,
                "selection": selection,
                "sparse_startup_seconds": startup_cost[name],
                "paired_order": "Alternate which variant goes first by question and repeat parity",
                "latency_scope": "Full live query embedding, retrieval, generation and validation; "
                "no HTTP/UI or indexing; BM25 startup recorded separately.",
                "citation_validity_scope": "Returned raw source identity and substring; "
                "not a semantic-entailment guarantee.",
            }
            json_write(path / "metadata.json", metadata[name])
        for position, question in enumerate(dataset.questions):
            order = (
                ("baseline", "optimized") if (position + repeat) % 2 else ("optimized", "baseline")
            )
            for name in order:
                top_k = metadata[name]["top_k"]
                module = baseline_module if name == "baseline" else optimized_module
                row = run_question(question, module, retrieval[name], chat, settings, labels, top_k)
                rows[name].append(row)
                with (paths[name] / "responses.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                status = (
                    "ERROR " + row["error_type"]
                    if row["error_type"]
                    else (row["response"]["reason"] or "answer")
                )
                print(
                    f"repeat{repeat} {name} {question.id}: {status} {row['latency_seconds']:.2f}s",
                    flush=True,
                )
        for name, path in paths.items():
            metadata[name]["status"] = "complete"
            metadata[name]["finished_at_utc"] = datetime.now(UTC).isoformat()
            json_write(path / "metadata.json", metadata[name])
            write_summary(rows[name], metadata[name], path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.json"))
    parser.add_argument("--index-dir", type=Path, default=Path("eval/phase3/workspace/index"))
    parser.add_argument("--output", type=Path, default=Path("eval/phase3"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--selection", type=Path, default=Path("eval/phase3/selection.json"))
    parser.add_argument("--labels", type=Path, default=Path("eval/phase3/evidence_labels.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    try:
        run_paired(
            args.dataset,
            args.index_dir,
            args.output,
            selection,
            args.repeats,
            labels_path=args.labels,
        )
    except Exception as exc:
        print(
            f"Paired evaluation failed ({type(exc).__name__}); raw upstream errors suppressed.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
