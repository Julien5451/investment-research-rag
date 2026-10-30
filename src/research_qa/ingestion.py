"""Bounded PDF extraction and page-preserving text splitting."""

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class InvalidPDF(ValueError):
    """The submitted PDF cannot be safely indexed as text."""


def extract_pages(
    data: bytes,
    document_id: str,
    document_name: str,
    *,
    max_pages: int,
    max_extracted_chars: int,
) -> tuple[list[Document], int]:
    """Extract nonblank pages while retaining physical, one-based PDF page numbers.

    The returned count includes blank pages. Image-only PDFs need OCR, which this
    service intentionally does not perform. Password-protected files are rejected
    rather than attempting to decrypt them.
    """
    if max_pages < 1 or max_extracted_chars < 1:
        raise ValueError("PDF extraction limits must be positive.")
    if not data:
        raise InvalidPDF("The PDF is empty.")

    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except (RuntimeError, ValueError) as exc:
        raise InvalidPDF("The file is not a readable PDF.") from exc

    with pdf:
        if not pdf.is_pdf:
            raise InvalidPDF("The file is not a PDF.")
        if pdf.needs_pass:
            raise InvalidPDF("Password-protected PDFs are not supported.")
        page_count = pdf.page_count
        if page_count < 1:
            raise InvalidPDF("The PDF contains no pages.")
        if page_count > max_pages:
            raise InvalidPDF(f"The PDF exceeds the {max_pages}-page limit.")

        pages: list[Document] = []
        extracted_chars = 0
        for page_number in range(page_count):
            try:
                text = pdf.load_page(page_number).get_text("text", sort=True)
            except (RuntimeError, ValueError) as exc:
                raise InvalidPDF("A PDF page could not be read.") from exc
            extracted_chars += len(text)
            if extracted_chars > max_extracted_chars:
                raise InvalidPDF("The PDF exceeds the extracted-text character limit.")
            text = text.replace("\x00", "").strip()
            if text:
                pages.append(
                    Document(
                        page_content=text,
                        metadata={
                            "document_id": document_id,
                            "document_name": document_name,
                            "page": page_number + 1,
                        },
                    )
                )
        if not pages:
            raise InvalidPDF("The PDF has no extractable text; scanned PDFs require OCR.")
        return pages, page_count


def chunk_pages(pages: list[Document], *, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split each page independently so every chunk has one unambiguous citation."""
    if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks: list[Document] = []
    for page in pages:
        for index, chunk in enumerate(splitter.split_documents([page])):
            chunk.metadata["chunk_id"] = (
                f"{page.metadata['document_id']}:{page.metadata['page']}:{index}"
            )
            chunks.append(chunk)
    return chunks
