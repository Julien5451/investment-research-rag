"""Compare fixed-vector dense and BM25/RRF retrieval before chat evaluation."""

import csv
import hashlib
import json
import statistics
from pathlib import Path
from time import perf_counter

from research_qa.config import Settings
from research_qa.evaluation import load_dataset
from research_qa.evidence import EvidenceLabels
from research_qa.index import VectorIndex
from research_qa.providers import build_embeddings, embedding_fingerprint
from research_qa.qa import INJECTION
from research_qa.retrieval import ExperimentalRetriever

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "eval/phase3"


def page_recall(question, results, k):
    gold = {(item.document_name, item.page) for item in question.evidence}
    found = {(doc.metadata["document_name"], doc.metadata["page"]) for doc, _ in results[:k]}
    return len(gold & found) / len(gold) if gold else None


def main() -> None:
    dataset_path = ROOT / "eval/dataset.json"
    dataset = load_dataset(dataset_path)
    settings = Settings().model_copy(
        update={
            "data_dir": PHASE / "workspace/index",
            "collection_name": "evaluation",
        }
    )
    embeddings = build_embeddings(settings)
    index = VectorIndex(settings, embeddings)
    cache_path = PHASE / "workspace/query_vectors.json"
    identity = {
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "embedding_fingerprint": embedding_fingerprint(settings),
    }
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache["identity"] != identity:
            raise ValueError("Cached query vectors belong to a different frozen dataset/model")
    else:
        started = perf_counter()
        vectors = embeddings.embed_documents([q.question for q in dataset.questions])
        cache = {
            "identity": identity,
            "batch_embedding_seconds": perf_counter() - started,
            "vectors": dict(zip([q.id for q in dataset.questions], vectors, strict=True)),
        }
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    labels = EvidenceLabels(PHASE / "evidence_labels.json")
    rows, summary = [], []
    retrievers = {
        method: ExperimentalRetriever(index, method=method) for method in ("dense", "hybrid")
    }
    started = perf_counter()
    retrievers["hybrid"].refresh()
    sparse_startup_seconds = perf_counter() - started
    for question in dataset.questions:
        for method, retriever in retrievers.items():
            for k in (3, 5):
                started = perf_counter()
                results = retriever.search_vector(
                    cache["vectors"][question.id], question.question, k
                )
                elapsed = perf_counter() - started
                filtered = [
                    (doc, score)
                    for doc, score in results
                    if score >= settings.min_relevance_score
                    and not INJECTION.search(doc.page_content)
                ]
                raw_support = labels.evaluate(question.id, results)
                surviving_support = labels.evaluate(question.id, filtered)
                rows.append(
                    {
                        "id": question.id,
                        "question": question.question,
                        "answerable": question.answerable,
                        "method": method,
                        "top_k": k,
                        "page_recall": page_recall(question, results, k),
                        "evidence": raw_support,
                        "filtered_evidence": surviving_support,
                        "ranking_seconds": elapsed,
                        "context_chars": min(
                            settings.max_context_chars,
                            sum(len(doc.page_content) for doc, _ in filtered),
                        ),
                        "relevant_chunks_surviving": len(surviving_support["relevant_chunk_ids"]),
                        "retrieved": [
                            {
                                "chunk_id": doc.metadata["chunk_id"],
                                **doc.metadata,
                                "relevance_score": score,
                                "text": doc.page_content,
                                "filtered_by_threshold": score < settings.min_relevance_score,
                                "filtered_by_injection": bool(INJECTION.search(doc.page_content)),
                            }
                            for doc, score in results
                        ],
                        "ranking_trace": retriever.last_trace,
                    }
                )
    for method in ("dense", "hybrid"):
        for k in (3, 5):
            subset = [r for r in rows if r["method"] == method and r["top_k"] == k]
            answerable = [r for r in subset if r["answerable"]]
            verified = [r for r in answerable if r["evidence"]["evidence_recall"] is not None]
            summary.append(
                {
                    "method": method,
                    "top_k": k,
                    "questions": len(subset),
                    "verified_answerable_questions": len(verified),
                    "page_recall": statistics.mean(r["page_recall"] for r in answerable),
                    "evidence_recall": statistics.mean(
                        r["evidence"]["evidence_recall"] for r in verified
                    ),
                    "filtered_evidence_recall": statistics.mean(
                        r["filtered_evidence"]["evidence_recall"] for r in verified
                    ),
                    "fully_supported_questions": sum(
                        r["evidence"]["full_support"] for r in verified
                    ),
                    "fully_supported_after_filtering": sum(
                        r["filtered_evidence"]["full_support"] for r in verified
                    ),
                    "mean_ranking_seconds": statistics.mean(r["ranking_seconds"] for r in subset),
                    "p50_ranking_seconds": statistics.median(r["ranking_seconds"] for r in subset),
                    "p95_ranking_seconds": statistics.quantiles(
                        [r["ranking_seconds"] for r in subset], n=20, method="inclusive"
                    )[18],
                    "mean_context_chars": statistics.mean(r["context_chars"] for r in subset),
                    "mean_relevant_chunks_surviving": statistics.mean(
                        r["relevant_chunks_surviving"] for r in answerable
                    ),
                }
            )
    payload = {
        "schema_version": 1,
        **identity,
        "corpus_chunks": index.count(),
        "candidate_k": 20,
        "rrf_k": 60,
        "min_relevance_score": settings.min_relevance_score,
        "max_context_chars": settings.max_context_chars,
        "query_embedding_batch_seconds": cache["batch_embedding_seconds"],
        "sparse_startup_seconds": sparse_startup_seconds,
        "latency_scope": "Local ranking only with the same cached real query vectors. "
        "Batch embedding time reported separately, not divided into query latency.",
        "top_k_tradeoff": "Top5 may supply up to5000 chunk characters vs3000 forTop3, "
        "under the same16000-character cap; not a single-variable context comparison.",
        "summaries": summary,
    }
    (PHASE / "retrieval_experiments.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (PHASE / "retrieval_runs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    with (PHASE / "retrieval_comparison.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
