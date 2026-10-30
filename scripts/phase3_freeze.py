"""Freeze the development-selected candidate before paired and holdout evaluation."""

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from research_qa.config import Settings


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    phase = root / "eval/phase3"
    source = root / "src/research_qa/qa.py"
    frozen = phase / "frozen/qa_selected.py"
    if frozen.exists() and digest(frozen) != digest(source):
        raise ValueError("Candidate already frozen with different source; do not retune a holdout")
    shutil.copy2(source, frozen)
    retrieval = json.loads((phase / "retrieval_experiments.json").read_text())
    settings = Settings()
    selection = {
        "schema_version": 1,
        "state": "candidate_frozen_before_end_to_end_and_holdout",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "retrieval_method": "dense",
        "top_k": 5,
        "min_relevance_score": 0.35,
        "max_context_chars": 16000,
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "citation_retry_limit": 0,
        "rationale": "Dense5 adds one fully supported question vsDense3; hybrid has no "
        "evidence-recall benefit at either k and adds sparse startup/ranking cost. "
        "Keep hybrid experimental only; trial generic completeness/table-copy prompt.",
        "interventions": [
            "Top3 toTop5 (more available context)",
            "generic multipart completeness and contiguous table-copy prompt",
        ],
        "unchanged": [
            "corpus/chunkIDs/text/metadata",
            "models",
            "gold/rubric",
            "cosine threshold0.35",
            "citationvalidator",
            "maxcontext16000",
        ],
        "retrieval_experiment": retrieval,
        "source_hashes": {
            path: digest(root / path)
            for path in (
                "src/research_qa/qa.py",
                "src/research_qa/retrieval.py",
                "src/research_qa/index.py",
                "src/research_qa/ingestion.py",
                "uv.lock",
            )
        },
        "dataset_sha256": digest(root / "eval/dataset.json"),
        "evidence_labels_sha256": digest(phase / "evidence_labels.json"),
        "holdout_dataset_sha256": digest(phase / "holdout_dataset.json"),
        "holdout_selection_policy": "No holdout retrieval or chat outputs observed before freeze; "
        "no implementation or parameter tuning on holdout results.",
    }
    target = phase / "selection.json"
    if target.exists():
        raise ValueError("Selection already frozen; preserve the existing decision record")
    target.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Candidate frozen: dense5 + prompt-only coverage/table-copy; no citation repair retry")


if __name__ == "__main__":
    main()
