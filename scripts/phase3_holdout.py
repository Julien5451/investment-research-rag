"""Evaluate the frozen implementation on the separately authored Costco holdout."""

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from research_qa.config import Settings
from research_qa.evaluation import _prepare_documents, load_dataset
from research_qa.index import VectorIndex
from research_qa.ingestion import chunk_pages
from research_qa.phase3_evaluation import json_write, run_paired
from research_qa.providers import build_embeddings


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    phase = root / "eval/phase3"
    selection = json.loads((phase / "selection.json").read_text())
    decision = json.loads((phase / "selection_decision.json").read_text())
    if decision["selected"] != "optimized":
        raise ValueError("This holdout command requires an accepted, frozen candidate")
    for relative, expected in selection["source_hashes"].items():
        if digest(root / relative) != expected:
            raise ValueError("Frozen implementation changed; do not tune on holdout outputs")
    dataset_path = phase / "holdout_dataset.json"
    if digest(dataset_path) != selection["holdout_dataset_sha256"]:
        raise ValueError("Holdout labels changed after the candidate freeze")
    output = phase / "holdout"
    if output.exists():
        raise ValueError("Holdout output already exists; preserve the original run")
    output.mkdir()
    frozen = output / "frozen"
    frozen.mkdir()
    for name in ("qa_phase2.py", "qa_selected.py"):
        shutil.copy2(phase / "frozen" / name, frozen / name)
    dataset = load_dataset(dataset_path)
    index_dir = phase / "workspace/holdout_index"
    settings = Settings().model_copy(
        update={
            "data_dir": index_dir,
            "collection_name": "evaluation",
        }
    )
    if (settings.chat_model, settings.embedding_model) != (
        selection["chat_model"],
        selection["embedding_model"],
    ):
        raise ValueError("Model identities changed after freeze")
    prepared = _prepare_documents(dataset, root / "eval/pdfs", settings)
    embeddings = build_embeddings(settings)
    index = VectorIndex(settings, embeddings)
    if index.count():
        raise ValueError("Holdout index must start empty")
    started = perf_counter()
    documents = []
    for document, pages, count in prepared:
        chunks = chunk_pages(
            pages, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        index.add(chunks)
        documents.append({**document.model_dump(), "pages": count, "chunks": len(chunks)})
    json_write(
        output / "freeze_check.json",
        {
            "implementation_unchanged": True,
            "checked_at_utc": datetime.now(UTC).isoformat(),
            "selection_decision": decision,
            "source_hashes": selection["source_hashes"],
            "dataset_sha256": digest(dataset_path),
            "documents": documents,
            "indexing_seconds": perf_counter() - started,
            "scope": "One holdout pass per frozen variant; no tuning on these outputs. "
            "Holdout evidence-level labels are unavailable; do not substitute page labels.",
        },
    )
    run_paired(
        dataset_path, index_dir, output, selection, 1, embeddings=embeddings, labels_path=None
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Holdout failed ({type(exc).__name__}); upstream details suppressed.")
        raise SystemExit(1) from None
