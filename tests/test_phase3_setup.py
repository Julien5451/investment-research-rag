"""Fresh-clone bootstrap checks use only generated PDFs and local test vectors."""

import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pymupdf
import pytest
from langchain_core.embeddings import Embeddings

from research_qa.config import Settings
from research_qa.index import VectorIndex
from research_qa.ingestion import chunk_pages, extract_pages
from research_qa.providers import embedding_fingerprint


class OfflineEmbeddings(Embeddings):
    def __init__(self):
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[1.0, 0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        raise AssertionError("Index rebuilding must not query or invoke chat")


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    builder = importlib.import_module("phase3_build_index")
    setup = importlib.import_module("phase3_setup")
    root = tmp_path / "repo"
    frozen = root / "eval/phase3/frozen"
    frozen.mkdir(parents=True)
    pdf_dir = root / "eval/pdfs"
    pdf_dir.mkdir(parents=True)
    text = "FY2025 revenue was USD100million. Revenue grew because cloud demand increased."
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 50), text)
        data = pdf.tobytes()
    (pdf_dir / "annual.pdf").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    dataset = {
        "schema_version": 1,
        "name": "Offline bootstrap fixture",
        "documents": [
            {
                "document_name": "annual.pdf",
                "url": "https://example.invalid/a.pdf",
                "sha256": digest,
            }
        ],
        "questions": [
            {
                "id": "q1",
                "question": "What was FY2025 revenue?",
                "answerable": True,
                "reference_answer": "USD100million",
                "category": "financial",
                "evidence": [
                    {
                        "document_name": "annual.pdf",
                        "page": 1,
                        "quote": "FY2025 revenue was USD100million.",
                    }
                ],
                "required_patterns": ["100"],
            }
        ],
    }
    dataset_bytes = json.dumps(dataset).encode()
    (root / "eval/dataset.json").write_bytes(dataset_bytes)
    (frozen / "apple_dataset.json").write_bytes(dataset_bytes)
    pages, _ = extract_pages(data, digest, "annual.pdf", max_pages=10, max_extracted_chars=10000)
    chunks = chunk_pages(pages, chunk_size=55, chunk_overlap=5)
    labels = {
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "corpus": {
            c.metadata["chunk_id"]: {
                "document_name": "annual.pdf",
                "page": 1,
                "text_sha256": hashlib.sha256(c.page_content.encode()).hexdigest(),
            }
            for c in chunks
        },
    }
    labels_path = frozen.parent / "evidence_labels.json"
    labels_path.write_text(json.dumps(labels))
    settings = Settings(
        _env_file=None,
        embedding_provider="openai-compatible",
        embedding_model="frozen-model",
        embedding_base_url="http://localhost/v1",
        chunk_size=500,
        chunk_overlap=10,
    )
    result_dir = root / "eval/results/latest"
    result_dir.mkdir(parents=True)
    (result_dir / "metadata.json").write_text(
        json.dumps(
            {
                "embedding_model": settings.embedding_model,
                "embedding_fingerprint": embedding_fingerprint(settings),
                "chunk_size": 55,
                "chunk_overlap": 5,
            }
        )
    )
    source_dir = root / "src/research_qa"
    source_dir.mkdir(parents=True)
    (source_dir / "qa.py").write_text("# frozen QA fixture\n")
    (source_dir / "config.py").write_text("# frozen configuration fixture\n")
    return root, settings, builder, setup, labels_path, len(chunks)


def test_builds_real_local_index_using_original_chunk_settings_and_injected_embeddings(bootstrap):
    root, settings, builder, _, _, count = bootstrap
    embeddings = OfflineEmbeddings()
    result = builder.build_index(root, settings, embeddings=embeddings)
    assert result["mode"] == "rebuilt"
    assert result["chunks"] == count > 1
    assert result["embedding_model"] == "frozen-model"
    assert embeddings.document_calls == 1
    assert (root / "eval/phase3/workspace/index/chroma/chroma.sqlite3").is_file()
    assert "api_key" not in json.dumps(result).lower()


@pytest.mark.parametrize("change", ["hash", "missing", "page", "document"])
def test_changed_corpus_fails_before_embedding_client_construction(bootstrap, monkeypatch, change):
    root, settings, builder, _, labels_path, _ = bootstrap
    labels = json.loads(labels_path.read_text())
    first = next(iter(labels["corpus"]))
    if change == "missing":
        del labels["corpus"][first]
    else:
        field = {"hash": "text_sha256", "page": "page", "document": "document_name"}[change]
        labels["corpus"][first][field] = {"hash": "0" * 64, "page": 2, "document": "different.pdf"}[
            change
        ]
    labels_path.write_text(json.dumps(labels))
    factory = MagicMock(side_effect=AssertionError("Must validate before API client construction"))
    monkeypatch.setattr(builder, "build_embeddings", factory)
    with pytest.raises(ValueError, match="chunk IDs/text"):
        builder.build_index(root, settings)
    factory.assert_not_called()
    assert not (root / "eval/phase3/workspace/index").exists()


@pytest.mark.parametrize("change", ["embedding_model", "embedding_base_url"])
def test_wrong_embedding_space_rejected_before_api(bootstrap, monkeypatch, change):
    root, settings, builder, _, _, _ = bootstrap
    factory = MagicMock()
    monkeypatch.setattr(builder, "build_embeddings", factory)
    value = "other-model" if change == "embedding_model" else "http://other.invalid/v1"
    with pytest.raises(ValueError, match="Embedding"):
        builder.build_index(root, settings.model_copy(update={change: value}))
    factory.assert_not_called()


def test_changed_dataset_and_pdf_rejected_before_api(bootstrap, monkeypatch):
    root, settings, builder, _, _, _ = bootstrap
    factory = MagicMock()
    monkeypatch.setattr(builder, "build_embeddings", factory)
    dataset = root / "eval/phase3/frozen/apple_dataset.json"
    original = dataset.read_bytes()
    dataset.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="Frozen dataset"):
        builder.build_index(root, settings)
    dataset.write_bytes(original)
    pdf = root / "eval/pdfs/annual.pdf"
    pdf.write_bytes(pdf.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        builder.build_index(root, settings)
    factory.assert_not_called()


def test_existing_target_is_never_opened_or_overwritten(bootstrap, monkeypatch):
    root, settings, builder, _, _, _ = bootstrap
    target = root / "eval/phase3/workspace/index"
    target.mkdir(parents=True)
    sentinel = target / "keep.txt"
    sentinel.write_text("existing index contents")
    factory = MagicMock()
    monkeypatch.setattr(builder, "build_embeddings", factory)
    with pytest.raises(FileExistsError):
        builder.build_index(root, settings)
    assert sentinel.read_text() == "existing index contents"
    factory.assert_not_called()


def test_setup_default_still_copies_original_index_and_reuses_target(bootstrap, monkeypatch):
    root, settings, builder, setup, _, count = bootstrap
    original = root / "eval/results/latest/index"
    dataset = json.loads((root / "eval/dataset.json").read_text())
    digest = dataset["documents"][0]["sha256"]
    pages, _ = extract_pages(
        (root / "eval/pdfs/annual.pdf").read_bytes(),
        digest,
        "annual.pdf",
        max_pages=10,
        max_extracted_chars=10000,
    )
    index = VectorIndex(
        settings.model_copy(update={"data_dir": original, "collection_name": "evaluation"}),
        OfflineEmbeddings(),
    )
    index.add(chunk_pages(pages, chunk_size=55, chunk_overlap=5))
    factory = MagicMock(side_effect=AssertionError("No rebuild when original exists"))
    monkeypatch.setattr(builder, "build_index", factory)
    result = setup.prepare(root)
    assert result["mode"] == "copied" and result["chunks"] == count
    target = root / "eval/phase3/workspace/index"
    assert (target / "chroma/chroma.sqlite3").is_file()
    assert setup.prepare(root, rebuild_index=True)["mode"] == "existing"
    factory.assert_not_called()


def test_fresh_clone_rebuild_is_explicit_and_preserves_frozen_manifest(bootstrap):
    root, settings, _, setup, _, count = bootstrap
    with pytest.raises(FileNotFoundError, match="--rebuild-index"):
        setup.prepare(root)
    frozen = root / "eval/phase3/frozen"
    before = {p.name: p.read_bytes() for p in frozen.iterdir() if p.is_file()}
    result = setup.prepare(
        root, rebuild_index=True, settings=settings, embeddings=OfflineEmbeddings()
    )
    assert result["mode"] == "rebuilt" and result["chunks"] == count
    assert before == {p.name: p.read_bytes() for p in frozen.iterdir() if p.is_file()}


def test_setup_rejects_modified_phase2_artifact_without_rewriting_manifest(bootstrap):
    root, _, _, setup, _, _ = bootstrap
    with pytest.raises(FileNotFoundError):
        setup.prepare(root)
    manifest = root / "eval/phase3/frozen/phase2_manifest.json"
    before = manifest.read_bytes()
    metadata = root / "eval/results/latest/metadata.json"
    metadata.write_text(metadata.read_text() + " ")
    with pytest.raises(ValueError, match="contents changed"):
        setup.prepare(root, rebuild_index=True)
    assert manifest.read_bytes() == before


def test_failed_embedding_rebuild_cannot_be_mistaken_for_ready_on_retry(bootstrap, monkeypatch):
    root, settings, builder, setup, _, _ = bootstrap

    class FailingEmbeddings(OfflineEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("simulated embedding outage")

    with pytest.raises(RuntimeError, match="simulated embedding"):
        setup.prepare(root, rebuild_index=True, settings=settings, embeddings=FailingEmbeddings())
    target = root / "eval/phase3/workspace/index"
    assert (target / "chroma/chroma.sqlite3").is_file()
    sentinel = target / "keep.txt"
    sentinel.write_text("preserve failed build for inspection")
    factory = MagicMock(side_effect=AssertionError("Retry must not call embeddings"))
    monkeypatch.setattr(builder, "build_embeddings", factory)
    with pytest.raises(ValueError, match="frozen corpus"):
        setup.prepare(root, rebuild_index=True, settings=settings)
    assert sentinel.read_text() == "preserve failed build for inspection"
    factory.assert_not_called()


@pytest.mark.parametrize("missing_database_folder", [False, True])
def test_setup_rejects_empty_or_missing_database_without_creating_one(
    bootstrap, missing_database_folder
):
    root, _, _, setup, _, _ = bootstrap
    target = root / "eval/phase3/workspace/index"
    target.mkdir(parents=True)
    if missing_database_folder:
        (target / "chroma").mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("unchanged")
    with pytest.raises(ValueError, match="incomplete"):
        setup.prepare(root, rebuild_index=True)
    assert sentinel.read_text() == "unchanged"
    assert not (target / "chroma/chroma.sqlite3").exists()


@pytest.mark.parametrize("change", ["text", "missing_chunk", "fingerprint"])
def test_setup_revalidates_real_existing_index_content(bootstrap, change):
    root, settings, builder, setup, _, _ = bootstrap
    setup.prepare(root, rebuild_index=True, settings=settings, embeddings=OfflineEmbeddings())
    target = root / "eval/phase3/workspace/index"
    index = VectorIndex(
        settings.model_copy(update={"data_dir": target, "collection_name": "evaluation"}),
        OfflineEmbeddings(),
    )
    chunk_id = index.collection.get(include=[])["ids"][0]
    if change == "text":
        index.collection.update(
            ids=[chunk_id],
            documents=["Changed unsupported source text"],
            embeddings=[[1.0, 0.1, 0.2]],
        )
    elif change == "missing_chunk":
        index.collection.delete(ids=[chunk_id])
    else:
        index.collection.modify(
            # Chroma disallows hnsw:space in modify(), even when unchanged.
            metadata={"schema_version": 1, "embedding_fingerprint": "wrong-space"}
        )
    expected = "embedding fingerprint" if change == "fingerprint" else "frozen corpus"
    with pytest.raises(ValueError, match=expected):
        setup.prepare(root)
    assert (target / "chroma/chroma.sqlite3").is_file()
