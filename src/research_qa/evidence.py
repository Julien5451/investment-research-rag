"""Offline, reviewed evidence bundles for a fixed benchmark corpus.

This module never infers relevance from page identity or answer-token presence.
Labels are a separate versioned sidecar, not edits to the Phase 2 gold dataset.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

DEFAULT_LABEL_PATH = Path(__file__).resolve().parents[2] / "eval/phase3/evidence_labels.json"


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _chunk(value: Any) -> dict[str, Any]:
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, Document):
        return {**value.metadata, "text": value.page_content}
    if isinstance(value, dict):
        metadata = value.get("metadata", {})
        return {**metadata, **value, "text": value.get("text", value.get("page_content", ""))}
    raise TypeError("Evidence assessment requires Documents or retrieved trace dictionaries")


class EvidenceLabels:
    """Evaluate explicit source bundles, including contextual multi-chunk tables.

    A requirement is supported when any reviewed alternative bundle is completely
    present. A bundle may require a value row and its year/unit header from two
    separate chunks. Exact original-text hashes protect against using an old label
    on a changed corpus; unfamiliar evidence is flagged for review, never guessed.
    """

    def __init__(self, path: Path | str | dict[str, Any] = DEFAULT_LABEL_PATH):
        self.data = path if isinstance(path, dict) else json.loads(Path(path).read_text("utf-8"))
        if self.data.get("schema_version") != 1:
            raise ValueError("Unsupported evidence-label schema")
        self.questions = self.data["questions"]
        self.corpus = self.data["corpus"]
        for question in self.questions.values():
            requirements = question.get("requirements", [])
            if question["answerable"] and not requirements:
                raise ValueError("Answerable evidence labels need at least one requirement")
            ids = [requirement["id"] for requirement in requirements]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate evidence requirement ID")
            for requirement in requirements:
                if requirement.get("status") == "verified" and not requirement.get("alternatives"):
                    raise ValueError("Verified evidence requirements need supporting alternatives")
                for alternative in requirement.get("alternatives", []):
                    if not alternative.get("pieces"):
                        raise ValueError("Evidence alternatives need source pieces")
                    for piece in alternative["pieces"]:
                        if (
                            piece["chunk_id"] not in self.corpus
                            or not piece.get("excerpt", "").strip()
                        ):
                            raise ValueError(
                                "Evidence piece needs a known chunk and nonempty excerpt"
                            )

    def _known(self, chunk: dict[str, Any]) -> bool:
        known = self.corpus.get(chunk.get("chunk_id"))
        return bool(
            known
            and known["document_name"] == chunk.get("document_name")
            and known["page"] == chunk.get("page")
            and known["text_sha256"] == hashlib.sha256(chunk["text"].encode()).hexdigest()
        )

    def _piece_present(self, piece: dict[str, Any], chunks: dict[str, dict[str, Any]]) -> bool:
        chunk = chunks.get(piece["chunk_id"])
        return bool(
            chunk
            and self._known(chunk)
            and normalize_whitespace(piece["excerpt"]) in normalize_whitespace(chunk["text"])
        )

    def _matched(self, requirements: list[dict[str, Any]], chunks: dict) -> list[str]:
        return [
            requirement["id"]
            for requirement in requirements
            if requirement.get("status") == "verified"
            and any(
                all(self._piece_present(piece, chunks) for piece in alternative["pieces"])
                for alternative in requirement["alternatives"]
            )
        ]

    @staticmethod
    def _minimum_chunks(requirements: list[dict[str, Any]]) -> int | None:
        choices = [
            [{piece["chunk_id"] for piece in alt["pieces"]} for alt in req["alternatives"]]
            for req in requirements
            if req.get("status") == "verified"
        ]
        if len(choices) != len(requirements) or not choices:
            return None
        return min(len(set().union(*combination)) for combination in itertools.product(*choices))

    def evaluate(self, question_id: str, docs: list[Any]) -> dict[str, Any]:
        """Return requirement recall and full-question support over this selection.

        Evidence Recall is a macro-ready fraction of all required evidence groups;
        it is None for unanswerable/unknown/partially unreviewed labels. A known
        value row without its period/unit context is partial, not full support.
        """
        question = self.questions.get(question_id)
        if question is None:
            return {
                "label_status": "requires_review",
                "evidence_recall": None,
                "full_support": None,
                "review_reason": "question_not_labeled",
            }
        if not question["answerable"]:
            return {
                "label_status": "not_applicable",
                "evidence_recall": None,
                "full_support": None,
                "required_groups": 0,
                "matched_groups": [],
                "relevant_chunk_ids": [],
                "multi_chunk_required": None,
                "selected_multiple_chunks_needed": None,
            }
        chunks = {_chunk(value)["chunk_id"]: _chunk(value) for value in docs}
        requirements = question["requirements"]
        matched = self._matched(requirements, chunks)
        unknown = [key for key, chunk in chunks.items() if not self._known(chunk)]
        verified = all(req.get("status") == "verified" for req in requirements)
        status = "verified" if verified and not unknown else "requires_review"
        relevant = sorted(
            key
            for key in chunks
            if any(
                piece["chunk_id"] == key and self._piece_present(piece, chunks)
                for requirement in requirements
                for alternative in requirement.get("alternatives", [])
                for piece in alternative["pieces"]
            )
        )
        full = len(matched) == len(requirements) if status == "verified" else None
        minimum = self._minimum_chunks(requirements)
        single_has_all = any(
            len(self._matched(requirements, {key: value})) == len(requirements)
            for key, value in chunks.items()
        )
        return {
            "label_status": status,
            "evidence_recall": len(matched) / len(requirements) if status == "verified" else None,
            "full_support": full,
            "required_groups": len(requirements),
            "matched_groups": matched,
            "relevant_chunk_ids": relevant,
            "multi_chunk_required": minimum > 1 if minimum is not None else None,
            "minimum_support_chunks": minimum,
            "selected_multiple_chunks_needed": bool(full and not single_has_all),
            "review_chunk_ids": unknown,
            "review_reason": "unfamiliar_source_or_unreviewed_requirement"
            if status != "verified"
            else None,
        }

    def assess_chunk(self, question_id: str, doc: Any) -> dict[str, Any]:
        result = self.evaluate(question_id, [doc])
        if result["label_status"] != "verified":
            support = result["label_status"]
        elif result["full_support"]:
            support = "verified_support"
        elif result["relevant_chunk_ids"]:
            support = "partial_support"
        else:
            support = "not_supporting"
        return {**result, "support_status": support}


def load_evidence_labels(path: Path | str = DEFAULT_LABEL_PATH) -> EvidenceLabels:
    return EvidenceLabels(path)


def evidence_recall(
    question_id: str, docs: list[Any], labels: EvidenceLabels | None = None
) -> dict[str, Any]:
    return (labels or load_evidence_labels()).evaluate(question_id, docs)


def assess_chunk(
    question_id: str, doc: Any, labels: EvidenceLabels | None = None
) -> dict[str, Any]:
    return (labels or load_evidence_labels()).assess_chunk(question_id, doc)
