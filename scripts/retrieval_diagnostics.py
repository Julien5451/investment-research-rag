"""Offline evidence-level audit of saved Phase 2 retrieval traces; no API calls."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from research_qa.evidence import EvidenceLabels
from research_qa.qa import INJECTION


def build_diagnostics(
    responses: Path, labels_path: Path, output: Path, *, min_score: float = 0.35
) -> dict:
    labels = EvidenceLabels(labels_path)
    rows = [json.loads(line) for line in responses.read_text("utf-8").splitlines() if line.strip()]
    csv_rows, question_rows = [], []
    for row in rows:
        selected = row["retrieved"]
        surviving = [
            chunk
            for chunk in selected
            if chunk["relevance_score"] >= min_score and not INJECTION.search(chunk["text"])
        ]
        before = labels.evaluate(row["id"], selected)
        after = labels.evaluate(row["id"], surviving)
        relevant_surviving = set(after.get("relevant_chunk_ids", []))
        question_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answerable": row["answerable"],
                "raw": before,
                "after_filtering": after,
                "relevant_chunks_surviving": len(relevant_surviving),
                "phase2_response": row["response"],
            }
        )
        for rank, chunk in enumerate(selected, 1):
            assessment = labels.assess_chunk(row["id"], chunk)
            csv_rows.append(
                {
                    "question_id": row["id"],
                    "query": row["question"],
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "document_name": chunk["document_name"],
                    "physical_page": chunk["page"],
                    "similarity_score": chunk["relevance_score"],
                    "gold_page_match": any(
                        item["document_name"] == chunk["document_name"]
                        and item["page"] == chunk["page"]
                        for item in row["evidence"]
                    ),
                    "support_status": assessment["support_status"],
                    "matched_required_groups": ";".join(assessment.get("matched_groups", [])),
                    "contains_all_required_evidence": assessment.get("full_support"),
                    "filtered_by_threshold": chunk["relevance_score"] < min_score,
                    "filtered_by_injection_rule": bool(INJECTION.search(chunk["text"])),
                    "relevant_and_survives": chunk["chunk_id"] in relevant_surviving,
                    "question_evidence_recall_before_filter": before["evidence_recall"],
                    "question_evidence_recall_after_filter": after["evidence_recall"],
                    "question_full_support": after.get("full_support"),
                    "multiple_chunks_required_in_corpus": before.get("multi_chunk_required"),
                    "selected_multiple_chunks_needed": after.get("selected_multiple_chunks_needed"),
                    "label_status": assessment["label_status"],
                    "source_text_sha256": hashlib.sha256(chunk["text"].encode()).hexdigest(),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    verified = [
        q for q in question_rows if q["answerable"] and q["raw"]["label_status"] == "verified"
    ]
    summary = {
        "source": str(responses),
        "source_sha256": hashlib.sha256(responses.read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "min_relevance_score": min_score,
        "questions": len(rows),
        "verified_answerable_questions": len(verified),
        "review_required_questions": [
            q["id"] for q in question_rows if q["raw"]["label_status"] == "requires_review"
        ],
        "evidence_recall_at_3": statistics.mean(q["raw"]["evidence_recall"] for q in verified),
        "evidence_recall_at_3_after_filter": statistics.mean(
            q["after_filtering"]["evidence_recall"] for q in verified
        ),
        "fully_supported_questions": sum(q["raw"]["full_support"] for q in verified),
        "filtered_by_threshold_count": sum(r["filtered_by_threshold"] for r in csv_rows),
        "filtered_by_injection_count": sum(r["filtered_by_injection_rule"] for r in csv_rows),
        "questions_detail": question_rows,
        "methodology": (
            "Reconstruction from saved raw top3 only; no new embeddings/chat calls. "
            "Evidence bundles verify exact source identity, wording, financial period/units "
            "and required context. Page identity alone never establishes evidence support. "
            "Unanswerable questions have no recall denominator. Historical failed raw "
            "model quotes were not saved and cannot be recovered by this diagnostic."
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses", type=Path, default=Path("eval/results/latest/responses.jsonl")
    )
    parser.add_argument("--labels", type=Path, default=Path("eval/phase3/evidence_labels.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("eval/phase3/retrieval_diagnostics.csv")
    )
    parser.add_argument("--min-score", type=float, default=0.35)
    args = parser.parse_args()
    summary = build_diagnostics(args.responses, args.labels, args.output, min_score=args.min_score)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "questions_detail"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
