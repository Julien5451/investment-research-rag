"""Optional dense/BM25+RRF experiments over unchanged Chroma chunks.

The production default remains ``VectorIndex.search``. This small adapter also
implements its search contract, so experiments can reuse the existing QA layer.
RRF controls ranking only: the returned relevance score is still clipped cosine
similarity, including for candidates nominated only by BM25. A relevance cutoff
such as 0.35 must never be applied to an RRF score.

English terms are lowercased without stemming or translation. Numeric tokens
retain decimal/thousands separators. Contiguous Han text contributes unigrams
and adjacent bigrams; this is explicit lexical matching, not a Chinese segmenter
or cross-language sparse search. It cannot translate Chinese questions into an
English report. No language model or external search service is used by BM25.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from research_qa.index import VectorIndex

_TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[,.]\d+)*|[\u3400-\u4dbf\u4e00-\u9fff]+")
_HAN = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


def tokenize(text: str) -> list[str]:
    """Lowercase Latin/numeric terms; split Han runs into unigram/bigram terms."""
    tokens = []
    for match in _TOKEN.finditer(text):
        token = match.group().lower()
        if _HAN.fullmatch(token):
            tokens.extend(token)
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return tokens


@dataclass(frozen=True)
class _Chunk:
    id: str
    document: Document
    embedding: tuple[float, ...]
    term_counts: Counter[str]
    token_count: int

    @property
    def chunk_id(self) -> str:
        return str(self.document.metadata["chunk_id"])


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _vector(values: list[float]) -> tuple[tuple[float, ...], float]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Query embedding must be a finite numeric vector") from exc
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("Query embedding must be a finite numeric vector")
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Query embedding must have a finite, nonzero norm")
    return vector, norm


class ExperimentalRetriever:
    """Rank existing chunks with dense cosine or unweighted BM25/dense RRF.

    Corpus data is read lazily for hybrid search and refreshed after append-only
    ingestion changes the collection count. Call ``refresh`` explicitly after
    an external same-count replacement. Existing IDs, text and metadata remain
    unchanged. ``search_vector`` accepts a cached query embedding so every
    experimental variant can use the identical query vector.
    """

    def __init__(
        self,
        index: VectorIndex,
        method: Literal["dense", "hybrid"] = "dense",
        candidate_k: int = 20,
        rrf_k: int = 60,
        *,
        embeddings: Embeddings | None = None,
    ):
        if method not in {"dense", "hybrid"}:
            raise ValueError("Retrieval method must be dense or hybrid")
        _positive_integer(candidate_k, "candidate_k")
        _positive_integer(rrf_k, "rrf_k")
        self.index = index
        self.method = method
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.embeddings = embeddings if embeddings is not None else index.store.embeddings
        self.last_trace: list[dict] = []
        self._chunks: dict[str, _Chunk] = {}
        self._corpus_count = -1
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0

    def refresh(self) -> None:
        """Rebuild the local sparse snapshot from existing Chroma data only."""
        with self.index.lock:
            data = self.index.collection.get(include=["documents", "metadatas", "embeddings"])
            documents = data["documents"]
            metadatas = data["metadatas"]
            embeddings = data["embeddings"]
            chunks: dict[str, _Chunk] = {}
            frequencies: Counter[str] = Counter()
            lengths = []
            for offset, stored_id in enumerate(data["ids"]):
                text, metadata = documents[offset], metadatas[offset]
                if text is None or metadata is None or "chunk_id" not in metadata:
                    raise ValueError("Indexed chunks need text and original chunk_id metadata")
                if embeddings is None:
                    raise ValueError("Indexed chunks need stored embeddings for cosine filtering")
                embedding = tuple(float(value) for value in embeddings[offset])
                tokens = tokenize(text)
                counts = Counter(tokens)
                chunks[stored_id] = _Chunk(
                    id=stored_id,
                    document=Document(id=stored_id, page_content=text, metadata=dict(metadata)),
                    embedding=embedding,
                    term_counts=counts,
                    token_count=len(tokens),
                )
                frequencies.update(counts.keys())
                lengths.append(len(tokens))
            self._chunks = chunks
            self._document_frequency = frequencies
            self._average_length = math.fsum(lengths) / len(lengths) if lengths else 0.0
            self._corpus_count = len(chunks)

    def _bm25(self, question: str) -> dict[str, float]:
        # BM25Okapi with positive Robertson IDF, k1=1.5 and b=0.75.
        # Each distinct query term contributes once; no question-specific tuning.
        terms = set(tokenize(question))
        if not terms or not self._average_length:
            return {}
        n = len(self._chunks)
        scores = {}
        for stored_id, chunk in self._chunks.items():
            contributions = []
            for term in sorted(terms):
                frequency = chunk.term_counts[term]
                if not frequency:
                    continue
                df = self._document_frequency[term]
                inverse_frequency = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * chunk.token_count / self._average_length
                )
                contributions.append(inverse_frequency * frequency * 2.5 / denominator)
            score = math.fsum(contributions)
            if score > 0:
                scores[stored_id] = score
        return scores

    def _dense(self, vector: tuple[float, ...], limit: int) -> list[tuple[str, Document, float]]:
        result = self.index.collection.query(
            query_embeddings=[list(vector)],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        return [
            (
                stored_id,
                Document(
                    id=stored_id,
                    page_content=result["documents"][0][offset],
                    metadata=dict(result["metadatas"][0][offset]),
                ),
                1.0 - float(result["distances"][0][offset]),
            )
            for offset, stored_id in enumerate(result["ids"][0])
        ]

    def search(self, question: str, top_k: int) -> list[tuple[Document, float]]:
        """Embed a question once, then apply the selected ranking strategy."""
        _positive_integer(top_k, "top_k")
        self.last_trace = []
        with self.index.lock:
            if not self.index.collection.count():
                return []
        if self.embeddings is None:
            raise ValueError("A query embedding provider is required for search")
        return self.search_vector(self.embeddings.embed_query(question), question, top_k)

    def search_vector(
        self, query_vector: list[float], question: str, top_k: int
    ) -> list[tuple[Document, float]]:
        """Search with an already computed vector; does not call model APIs."""
        _positive_integer(top_k, "top_k")
        self.last_trace = []
        vector, query_norm = _vector(query_vector)
        with self.index.lock:
            count = self.index.collection.count()
            if not count:
                return []
            if self.method == "dense":
                dense = self._dense(vector, min(top_k, count))
                self.last_trace = [
                    {
                        "chunk_id": doc.metadata["chunk_id"],
                        "rank": rank,
                        "dense_rank": rank,
                        "bm25_rank": None,
                        "bm25_score": None,
                        "rrf_score": None,
                        "raw_cosine_similarity": cosine,
                        "relevance_score": max(0.0, min(1.0, cosine)),
                    }
                    for rank, (_, doc, cosine) in enumerate(dense, 1)
                ]
                return [
                    (doc, item["relevance_score"])
                    for (_, doc, _), item in zip(dense, self.last_trace, strict=True)
                ]

            if self._corpus_count != count:
                self.refresh()
            dense = self._dense(vector, min(self.candidate_k, count))
            dense_ranks = {stored_id: rank for rank, (stored_id, _, _) in enumerate(dense, 1)}
            bm25 = self._bm25(question)
            sparse = sorted(bm25, key=lambda sid: (-bm25[sid], self._chunks[sid].chunk_id))
            sparse_ranks = {sid: rank for rank, sid in enumerate(sparse[: self.candidate_k], 1)}
            fused = {}
            for stored_id in dense_ranks.keys() | sparse_ranks.keys():
                ranks = [
                    ranks[stored_id] for ranks in (dense_ranks, sparse_ranks) if stored_id in ranks
                ]
                fused[stored_id] = math.fsum(1.0 / (self.rrf_k + rank) for rank in ranks)
            ordered = sorted(fused, key=lambda sid: (-fused[sid], self._chunks[sid].chunk_id))
            output = []
            for rank, stored_id in enumerate(ordered[:top_k], 1):
                chunk = self._chunks[stored_id]
                if len(chunk.embedding) != len(vector):
                    raise ValueError("Query embedding dimension differs from stored embeddings")
                norm = math.sqrt(math.fsum(value * value for value in chunk.embedding))
                if not norm:
                    raise ValueError("Indexed embedding has zero norm")
                cosine = math.fsum(a * b for a, b in zip(vector, chunk.embedding, strict=True))
                cosine /= query_norm * norm
                score = max(0.0, min(1.0, cosine))
                document = Document(
                    id=chunk.document.id,
                    page_content=chunk.document.page_content,
                    metadata=dict(chunk.document.metadata),
                )
                output.append((document, score))
                self.last_trace.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "rank": rank,
                        "dense_rank": dense_ranks.get(stored_id),
                        "bm25_rank": sparse_ranks.get(stored_id),
                        "bm25_score": bm25.get(stored_id),
                        "rrf_score": fused[stored_id],
                        "raw_cosine_similarity": cosine,
                        "relevance_score": score,
                    }
                )
            return output
