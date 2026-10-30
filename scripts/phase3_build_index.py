"""Rebuild the ignored Phase 3 index from pinned PDFs; embeddings only, no chat."""

import argparse
import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings

from research_qa.config import Settings
from research_qa.evaluation import _prepare_documents, load_dataset
from research_qa.index import VectorIndex
from research_qa.ingestion import chunk_pages
from research_qa.providers import build_embeddings, embedding_fingerprint


def validate_index(root: Path, target: Path) -> int:
    """Read and verify an existing corpus without constructing model clients."""
    if not (target / "chroma/chroma.sqlite3").is_file():
        raise ValueError("Existing index is incomplete; files were preserved")
    labels = json.loads((root / "eval/phase3/evidence_labels.json").read_text("utf-8"))
    original = json.loads((root / "eval/results/latest/metadata.json").read_text("utf-8"))
    try:
        client = chromadb.PersistentClient(
            path=str(target / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
        )
        collection = client.get_collection("evaluation", embedding_function=None)
        data = collection.get(include=["documents", "metadatas"])
    except Exception:
        raise ValueError(
            "Existing index is unreadable or incomplete; files were preserved"
        ) from None
    metadata = collection.metadata or {}
    if original.get("embedding_fingerprint") not in {None, metadata.get("embedding_fingerprint")}:
        raise ValueError("Existing index embedding fingerprint differs; files were preserved")
    actual = {}
    for chunk_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"], strict=True):
        if not text or not meta or meta.get("chunk_id") != chunk_id:
            raise ValueError(
                "Existing index does not match the frozen corpus; files were preserved"
            )
        actual[chunk_id] = {
            "document_name": meta.get("document_name"),
            "page": meta.get("page"),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    if (
        not actual
        or metadata.get("hnsw:space") != "cosine"
        or metadata.get("schema_version") != 1
        or actual != labels["corpus"]
    ):
        raise ValueError("Existing index does not match the frozen corpus; files were preserved")
    return len(actual)


def build_index(root: Path, settings: Settings, *, embeddings: Embeddings | None = None) -> dict:
    """Validate the frozen corpus completely before any paid embedding request."""
    phase = root / "eval/phase3"
    target = phase / "workspace/index"
    if target.exists():
        raise FileExistsError("Existing Phase 3 index was preserved")
    dataset_path = phase / "frozen/apple_dataset.json"
    labels = json.loads((phase / "evidence_labels.json").read_text("utf-8"))
    original = json.loads((root / "eval/results/latest/metadata.json").read_text("utf-8"))
    if settings.embedding_model != original["embedding_model"]:
        raise ValueError("Embedding model differs from the frozen Phase 2 configuration")
    if original.get("embedding_fingerprint") not in {None, embedding_fingerprint(settings)}:
        raise ValueError("Embedding provider/endpoint differs from the frozen configuration")
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if digest != labels["dataset_sha256"]:
        raise ValueError("Frozen dataset differs from the reviewed evidence labels")
    pinned = settings.model_copy(
        update={
            "data_dir": target,
            "collection_name": "evaluation",
            "chunk_size": original["chunk_size"],
            "chunk_overlap": original["chunk_overlap"],
        }
    )
    prepared = _prepare_documents(load_dataset(dataset_path), root / "eval/pdfs", pinned)
    chunks = [
        chunk
        for _, pages, _ in prepared
        for chunk in chunk_pages(
            pages, chunk_size=pinned.chunk_size, chunk_overlap=pinned.chunk_overlap
        )
    ]
    actual = {
        chunk.metadata["chunk_id"]: {
            "document_name": chunk.metadata["document_name"],
            "page": chunk.metadata["page"],
            "text_sha256": hashlib.sha256(chunk.page_content.encode()).hexdigest(),
        }
        for chunk in chunks
    }
    if len(actual) != len(chunks) or actual != labels["corpus"]:
        raise ValueError("Extracted chunk IDs/text differ from the frozen reviewed corpus")
    # All local source checks precede even client construction. Existing indexes
    # are never opened by this builder, and the normal fingerprint guard remains.
    client = embeddings if embeddings is not None else build_embeddings(pinned)
    index = VectorIndex(pinned, client)
    index.add(chunks)
    validate_index(root, target)
    metadata = {
        "mode": "rebuilt",
        "chunks": index.count(),
        "dataset_sha256": digest,
        "embedding_model": pinned.embedding_model,
        "embedding_fingerprint": embedding_fingerprint(pinned),
        "collection_name": pinned.collection_name,
    }
    (target / "build_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", "utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        print(json.dumps(build_index(args.root, Settings()), indent=2))
    except Exception as exc:
        raise SystemExit(
            f"Index rebuild failed ({type(exc).__name__}); verify pinned PDFs, "
            "corpus labels and embedding settings. Upstream details are suppressed."
        ) from None


if __name__ == "__main__":
    main()
