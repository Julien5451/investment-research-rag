"""Offline transport and rendered-interface checks for the HTTP-only demo."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from conftest import pdf_bytes

from research_qa.demo_client import DemoAPIError, QAClient
from research_qa.models import AskResponse, Citation, DocumentResponse, HealthResponse

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def response(*, abstained=False):
    return AskResponse(
        answer="Revenue was 100 million. [S1]" if not abstained else "Insufficient evidence.",
        abstained=abstained,
        reason="no_relevant_evidence" if abstained else None,
        citations=[]
        if abstained
        else [
            Citation(
                source_id="S1",
                document_id="demo-doc",
                document_name="annual.pdf",
                page=7,
                chunk_id="demo-doc:7:0",
                quote="Revenue was 100 million.",
                relevance_score=0.9,
            )
        ],
    )


def test_demo_client_calls_only_backend_endpoints():
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "indexed_chunks": 12})
        if request.url.path == "/documents":
            assert "multipart/form-data" in request.headers["content-type"]
            assert b'filename="annual.pdf"' in request.content
            assert b"%PDF-demo" in request.content
            return httpx.Response(
                201,
                json={
                    "document_id": "demo-doc",
                    "document_name": "annual.pdf",
                    "pages": 9,
                    "chunks": 12,
                },
            )
        assert request.url.path == "/ask"
        assert json.loads(request.content) == {"question": "Revenue?", "top_k": 3}
        return httpx.Response(200, json=response().model_dump())

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        client = QAClient("http://backend.invalid", http_client=transport)
        assert client.health().indexed_chunks == 12
        assert client.upload("annual.pdf", b"%PDF-demo").document_id == "demo-doc"
        result = client.ask("Revenue?", 3)
        assert result.citations[0].page == 7
    assert [request.url.path for request in requests] == ["/health", "/documents", "/ask"]
    assert all("authorization" not in request.headers for request in requests)


@pytest.mark.parametrize("status", [413, 415, 422, 500, 502])
def test_demo_client_does_not_display_raw_backend_errors(status):
    def handle(_request):
        return httpx.Response(status, json={"detail": "private-token-never-display"})

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        client = QAClient("http://backend.invalid", http_client=transport)
        with pytest.raises(DemoAPIError) as caught:
            client.ask("Revenue?", 3)
        assert "private-token" not in str(caught.value)


def test_demo_client_reports_connection_error_safely():
    def handle(request):
        raise httpx.ConnectError("secret-upstream-host", request=request)

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        with pytest.raises(DemoAPIError, match="无法连接") as caught:
            QAClient("http://backend.invalid", http_client=transport).health()
        assert "secret-upstream-host" not in str(caught.value)


def test_demo_client_validates_response_contract():
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"answer": "missing citation schema"})
        )
    ) as transport:
        with pytest.raises(DemoAPIError, match="格式错误"):
            QAClient("http://backend.invalid", http_client=transport).ask("Revenue?", 3)


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "http://user:secret@localhost", "http://x?q=1"]
)
def test_demo_rejects_urls_containing_secrets_or_invalid_protocols(url):
    with pytest.raises(DemoAPIError):
        QAClient(url)


@pytest.fixture
def healthy_backend(monkeypatch):
    monkeypatch.setenv("QA_API_URL", "http://backend.invalid")
    monkeypatch.setattr(
        QAClient, "health", lambda _: HealthResponse(status="ok", indexed_chunks=42)
    )
    monkeypatch.setattr(QAClient, "ask", lambda _self, _question, _top_k: response())
    monkeypatch.setattr(
        QAClient,
        "upload",
        lambda _self, _name, _data: DocumentResponse(
            document_id="demo-doc",
            document_name="annual.pdf",
            pages=9,
            chunks=12,
        ),
    )


def test_demo_renders_health_question_and_page_citation(healthy_backend):
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "研报问答"
    assert any(metric.value == "42" for metric in app.metric)
    app.chat_input[0].set_value("Revenue?").run(timeout=20)
    assert not app.exception
    assert len(app.chat_message) == 2
    assert any("Revenue was 100 million" in block.value for block in app.markdown)
    assert any(
        "annual.pdf" in expander.label and "第 7 页" in expander.label for expander in app.expander
    )
    assert any(text.value == "Revenue was 100 million." for text in app.text)
    app.run(timeout=20)
    assert len(app.session_state["turns"]) == 1  # a rerun cannot resubmit the question


def test_demo_upload_requires_click_and_deduplicates_reruns(healthy_backend, monkeypatch):
    upload = MagicMock(
        return_value=DocumentResponse(
            document_id="demo-doc",
            document_name="annual.pdf",
            pages=1,
            chunks=1,
        )
    )
    monkeypatch.setattr(QAClient, "upload", upload)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    data = pdf_bytes("Revenue was 100 million.")
    app.file_uploader[0].set_value(("annual.pdf", data, "application/pdf")).run(timeout=20)
    upload.assert_not_called()
    next(button for button in app.button if button.label == "上传并建立索引").click().run(
        timeout=20
    )
    assert not app.exception
    upload.assert_called_once_with("annual.pdf", data)
    assert app.session_state["uploaded_documents"]["demo-doc"]["data"] == data
    app.run(timeout=20)
    next(button for button in app.button if button.label == "上传并建立索引").click().run(
        timeout=20
    )
    assert not app.exception
    upload.assert_called_once()
    assert any("无需重复建库" in info.value for info in app.info)


def test_demo_renders_explicit_abstention(healthy_backend, monkeypatch):
    monkeypatch.setattr(QAClient, "ask", lambda _self, _question, _top_k: response(abstained=True))
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    app.chat_input[0].set_value("Unknown private metric?").run(timeout=20)
    assert not app.exception
    assert any("证据不足" in warning.value for warning in app.warning)
    assert any("no_relevant_evidence" in caption.value for caption in app.caption)


def test_demo_failed_backend_disables_upload_and_questions(monkeypatch):
    def fail(_self):
        raise DemoAPIError("无法连接问答服务")

    monkeypatch.setattr(QAClient, "health", fail)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    assert not app.exception
    assert app.chat_input[0].disabled
    assert any("无法连接问答服务" in error.value for error in app.error)
    assert next(button for button in app.button if button.label == "上传并建立索引").disabled


def test_demo_question_failure_remains_visible_in_history(healthy_backend, monkeypatch):
    def fail(_self, _question, _top_k):
        raise DemoAPIError("模型服务暂时不可用")

    monkeypatch.setattr(QAClient, "ask", fail)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    app.chat_input[0].set_value("Revenue?").run(timeout=20)
    assert not app.exception
    assert any("模型服务暂时不可用" in error.value for error in app.error)
    app.run(timeout=20)
    assert len(app.session_state["turns"]) == 1


def test_demo_can_preview_only_session_uploaded_pdf(healthy_backend):
    result = response().model_dump()
    result["citations"][0]["page"] = 1
    data = pdf_bytes("Revenue was 100 million.")
    app = AppTest.from_file(str(APP_PATH))
    app.session_state["uploaded_documents"] = {
        "demo-doc": {"data": data, "document_name": "annual.pdf", "pages": 1, "chunks": 1}
    }
    app.session_state["turns"] = [
        {
            "question": "Revenue?",
            "response": result,
            "elapsed_seconds": 1.5,
        }
    ]
    app.run(timeout=20)
    assert not app.exception
    assert app.checkbox[0].label == "预览引用页面"
    app.checkbox[0].check().run(timeout=20)
    assert not app.exception
    assert len(app.get("image")) == 1


@pytest.mark.parametrize(
    "mode,status,shows_metrics",
    [("live", "complete", True), ("mock", "complete", False), ("live", "failed", False)],
)
def test_demo_displays_only_saved_evaluation_metrics(
    healthy_backend, monkeypatch, mode, status, shows_metrics
):
    original_is_file = Path.is_file
    original_read_text = Path.read_text
    summary_path = APP_PATH.parent / "eval/results/latest/summary.json"
    summary = {
        "schema_version": 1,
        "run": {
            "mode": mode,
            "status": status,
            "chat_model": "real-model",
            "embedding_model": "real-embeddings",
            "started_at_shanghai": "2026-10-30T03:00:00+08:00",
        },
        "counts": {"questions": 24},
        "metrics": {
            "recall_at_3": 0.5,
            "answer_accuracy": 0.25,
            "correct_refusal_rate": 1.0,
            "latency_seconds": {"ask": {"p95": 2.0}},
            "answer_accuracy_method": "project_review",
        },
    }

    def exists(path):
        if path == APP_PATH.parent / "eval/phase3/optimized/summary.json":
            return False
        return True if path == summary_path else original_is_file(path)

    def read(path, *args, **kwargs):
        return (
            json.dumps(summary)
            if path == summary_path
            else original_read_text(path, *args, **kwargs)
        )

    monkeypatch.setattr(Path, "is_file", exists)
    monkeypatch.setattr(Path, "read_text", read)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    assert not app.exception
    shown = {metric.label: metric.value for metric in app.metric}
    if not shows_metrics:
        assert "页级 Recall@3" not in shown
        assert any("尚无完整的真实模型评测结果" in info.value for info in app.info)
        return
    assert shown["页级 Recall@3"] == "50.0%"
    assert shown["答案正确率"] == "25.0%"
    assert shown["正确拒答率"] == "100.0%"
    assert shown["问答延迟 P95"] == "2.00 秒"
    assert any("real-model" in caption.value and "24" in caption.value for caption in app.caption)


def test_demo_labels_repeated_phase3_trials_and_evidence_metric(healthy_backend, monkeypatch):
    summary_path = APP_PATH.parent / "eval/phase3/optimized/summary.json"
    original_read_text = Path.read_text
    original_is_file = Path.is_file
    summary = {
        "run": {
            "mode": "live",
            "status": "complete",
            "top_k": 5,
            "repeat_ids": [1, 2, 3],
            "chat_model": "chat",
            "embedding_model": "embedding",
        },
        "counts": {"questions": 72, "unique_questions": 24},
        "metrics": {
            "evidence_recall_at_k": 0.75,
            "recall_at_3": 0.5,
            "answer_accuracy": 0.8,
            "correct_refusal_rate": 1.0,
        },
    }
    monkeypatch.setattr(Path, "is_file", lambda p: p == summary_path or original_is_file(p))
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *args, **kwargs: (
            json.dumps(summary) if p == summary_path else original_read_text(p, *args, **kwargs)
        ),
    )
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    assert not app.exception
    shown = {metric.label: metric.value for metric in app.metric}
    assert shown["证据 Recall@5"] == "75.0%"
    assert any("24 题 × 3 轮 = 72 次答题" in caption.value for caption in app.caption)
    assert app.slider[0].value == 5
