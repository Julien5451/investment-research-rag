import pytest
from conftest import REVENUE, RISK, pdf_bytes
from langchain_core.documents import Document

from research_qa.ingestion import InvalidPDF, chunk_pages, extract_pages


def test_pages_keep_physical_numbers_and_chunks_do_not_cross_pages():
    pages, count = extract_pages(
        pdf_bytes("", REVENUE, RISK),
        "doc",
        "report.pdf",
        max_pages=10,
        max_extracted_chars=1000,
    )
    assert count == 3
    assert [page.metadata["page"] for page in pages] == [2, 3]
    chunks = chunk_pages(pages, chunk_size=25, chunk_overlap=5)
    assert len(chunks) > 2
    for chunk in chunks:
        original = REVENUE if chunk.metadata["page"] == 2 else RISK
        assert chunk.page_content in original
        assert len(chunk.page_content) <= 25
        assert chunk.metadata["document_id"] == "doc"
        assert chunk.metadata["document_name"] == "report.pdf"
        assert chunk.metadata["chunk_id"].startswith(f"doc:{chunk.metadata['page']}:")
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    repeated = chunk_pages(pages, chunk_size=25, chunk_overlap=5)
    assert [c.metadata["chunk_id"] for c in repeated] == [c.metadata["chunk_id"] for c in chunks]


def test_unicode_chunking_retains_text_and_metadata():
    text = "营业收入增长。风险来自汇率。利润增长。"
    original = Document(
        page_content=text,
        metadata={
            "document_id": "zh",
            "document_name": "中文报告.pdf",
            "page": 7,
        },
    )
    chunks = chunk_pages([original], chunk_size=9, chunk_overlap=2)
    assert len(chunks) > 1
    assert all(chunk.metadata["page"] == 7 for chunk in chunks)
    assert all(chunk.page_content in text for chunk in chunks)


@pytest.mark.parametrize(
    "data", [b"", b"%PDF-1.7 invalid", pdf_bytes(""), pdf_bytes(REVENUE, encrypted=True)]
)
def test_extractor_rejects_unreadable_or_textless_pdfs(data):
    with pytest.raises(InvalidPDF):
        extract_pages(data, "doc", "bad.pdf", max_pages=10, max_extracted_chars=1000)
