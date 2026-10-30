"""FastAPI entry point. Run one local worker against each Chroma directory."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from openai import OpenAIError
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from research_qa.config import Settings
from research_qa.index import VectorIndex
from research_qa.ingestion import InvalidPDF, chunk_pages, extract_pages
from research_qa.models import AskRequest, AskResponse, DocumentResponse, HealthResponse
from research_qa.providers import build_chat, build_embeddings
from research_qa.qa import QuestionAnswerer

logger = logging.getLogger(__name__)


class UploadBodyLimit:
    """Bound the raw multipart body even without Content-Length, before parsing.

    The small overhead allowance covers multipart headers/boundaries. The endpoint
    separately enforces the exact file-byte limit. Peak buffered body is bounded.
    """

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes + 65536

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/documents":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        try:
            announced = int(headers.get(b"content-length", b"0"))
        except ValueError:
            await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                scope, receive, send
            )
            return
        if announced < 0:
            await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                scope, receive, send
            )
            return
        parts = []
        size = 0
        if announced <= self.max_bytes:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                part = message.get("body", b"")
                size += len(part)
                if size > self.max_bytes:
                    break
                parts.append(part)
                if not message.get("more_body", False):
                    body = b"".join(parts)

                    async def replay() -> dict:
                        nonlocal body
                        chunk, body = body, b""
                        return {"type": "http.request", "body": chunk, "more_body": False}

                    await self.app(scope, replay, send)
                    return
        await JSONResponse({"detail": "Upload exceeds size limit"}, status_code=413)(
            scope, receive, send
        )


def create_app(
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
    chat: BaseChatModel | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        embedding_client = embeddings if embeddings is not None else build_embeddings(settings)
        model = chat if chat is not None else build_chat(settings)
        index = VectorIndex(settings, embedding_client)
        app.state.index = index
        app.state.qa = QuestionAnswerer(settings, index, model)
        yield

    app = FastAPI(title="Investment Research Document QA", version="0.1.0", lifespan=lifespan)
    app.add_middleware(UploadBodyLimit, max_bytes=settings.max_upload_bytes)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        # Local liveness/index check only. No remote model requests or secret output.
        return HealthResponse(status="ok", indexed_chunks=app.state.index.count())

    @app.post("/documents", response_model=DocumentResponse, status_code=201)
    def upload_document(file: Annotated[UploadFile, File()]) -> DocumentResponse:
        # Resolve either Windows or POSIX client paths. Client names never become
        # storage paths; UUIDs prevent traversal, collisions and accidental overwrite.
        document_name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        stored: Path | None = None
        try:
            if (
                not document_name
                or len(document_name) > 255
                or any(ord(char) < 32 for char in document_name)
                or Path(document_name).suffix.lower() != ".pdf"
                or file.content_type not in {"application/pdf", "application/octet-stream"}
            ):
                raise HTTPException(415, "A PDF filename and PDF content type are required")
            data = file.file.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, "PDF exceeds size limit")
            if not data.startswith(b"%PDF-"):
                raise HTTPException(415, "File does not have a PDF signature")
            document_id = uuid4().hex
            pages, total_pages = extract_pages(
                data,
                document_id,
                document_name,
                max_pages=settings.max_pdf_pages,
                max_extracted_chars=settings.max_extracted_chars,
            )
            chunks = chunk_pages(
                pages, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
            )
            upload_dir = settings.data_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            stored = upload_dir / f"{document_id}.pdf"
            stored.write_bytes(data)
            app.state.index.add(chunks)
            return DocumentResponse(
                document_id=document_id,
                document_name=document_name,
                pages=total_pages,
                chunks=len(chunks),
            )
        except InvalidPDF as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OpenAIError, httpx.HTTPError) as exc:
            if stored:
                stored.unlink(missing_ok=True)
            raise HTTPException(502, "Embedding provider request failed") from exc
        except HTTPException:
            raise
        except Exception as exc:
            if stored:
                stored.unlink(missing_ok=True)
            logger.error("Document indexing failed (%s)", type(exc).__name__)
            raise HTTPException(500, "Document indexing failed") from exc
        finally:
            file.file.close()

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            return app.state.qa.ask(request.question, request.top_k)
        except (OpenAIError, httpx.HTTPError) as exc:
            raise HTTPException(502, "Model provider request failed") from exc
        except Exception as exc:
            logger.error("Question answering failed (%s)", type(exc).__name__)
            raise HTTPException(500, "Question answering failed") from exc

    return app


app = create_app()
