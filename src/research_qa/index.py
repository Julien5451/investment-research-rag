"""Persistent cosine search with a guard against incompatible embedding models."""

from threading import RLock

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from research_qa.config import Settings
from research_qa.providers import embedding_fingerprint


class VectorIndex:
    def __init__(self, settings: Settings, embeddings: Embeddings):
        self.lock = RLock()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(settings.data_dir / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        fingerprint = embedding_fingerprint(settings)
        try:
            existing = self.client.get_collection(settings.collection_name)
        except chromadb.errors.NotFoundError:
            existing = None
        if existing is not None:
            metadata = existing.metadata or {}
            if metadata.get("embedding_fingerprint") != fingerprint:
                raise ValueError(
                    "Embedding configuration differs from this collection. "
                    "Use a new COLLECTION_NAME and re-upload the PDFs."
                )
            if metadata.get("schema_version") != 1 or metadata.get("hnsw:space") != "cosine":
                raise ValueError("Incompatible index schema; use a new COLLECTION_NAME.")
        self.collection = self.client.get_or_create_collection(
            settings.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "schema_version": 1,
                "embedding_fingerprint": fingerprint,
            },
            embedding_function=None,
        )
        self.store = Chroma(
            client=self.client,
            collection_name=settings.collection_name,
            embedding_function=embeddings,
        )

    def add(self, chunks: list[Document]) -> None:
        """Rollback all chunks from a failed, newly assigned upload ID."""
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        with self.lock:
            try:
                for offset in range(0, len(chunks), 64):
                    self.store.add_documents(
                        chunks[offset : offset + 64], ids=ids[offset : offset + 64]
                    )
            except Exception:
                self.collection.delete(ids=ids)
                raise

    def search(self, question: str, top_k: int) -> list[tuple[Document, float]]:
        with self.lock:
            count = self.collection.count()
            if not count:
                return []
            results = self.store.similarity_search_with_score(question, k=min(top_k, count))
        # Chroma's cosine distance is 1 - cosine similarity. Negative similarities
        # are clipped to zero; this is a retrieval score, not a truth probability.
        return [(doc, max(0.0, min(1.0, 1.0 - distance))) for doc, distance in results]

    def count(self) -> int:
        with self.lock:
            return self.collection.count()
