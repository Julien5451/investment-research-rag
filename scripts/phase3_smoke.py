"""Exercise actual upload/ask/citation/abstention routes with real providers."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from fastapi.testclient import TestClient

from research_qa.config import Settings
from research_qa.main import create_app


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = {"mode": "live", "transport": "FastAPI TestClient HTTP + real remote model APIs"}
    with TemporaryDirectory(prefix="research-qa-live-smoke-") as temporary:
        settings = Settings().model_copy(
            update={
                "data_dir": Path(temporary),
                "collection_name": "phase3-smoke",
                "default_top_k": 5,
            }
        )
        app = create_app(settings)
        with TestClient(app) as client:
            started = perf_counter()
            upload = client.post(
                "/documents",
                files={
                    "file": (
                        "apple_2025_10k.pdf",
                        (root / "eval/pdfs/apple_2025_10k.pdf").read_bytes(),
                        "application/pdf",
                    )
                },
            )
            assert upload.status_code == 201
            report["upload"] = {
                "status_code": upload.status_code,
                **upload.json(),
                "elapsed_seconds": perf_counter() - started,
            }
            assert client.get("/health").json()["indexed_chunks"] == upload.json()["chunks"]
            answer = client.post(
                "/ask",
                json={
                    "question": "Which fiscal quarter historically has higher net sales, "
                    "and what seasonal reason does Apple give?",
                    "top_k": 5,
                },
            )
            assert answer.status_code == 200 and not answer.json()["abstained"]
            assert "holiday" in answer.json()["answer"].lower()
            for citation in answer.json()["citations"]:
                stored = app.state.index.collection.get(ids=[citation["chunk_id"]])
                assert citation["quote"] in stored["documents"][0]
                assert citation["page"] == stored["metadatas"][0]["page"]
            report["grounded_answer"] = answer.json()
            refusal = client.post(
                "/ask",
                json={
                    "question": "What will Apple's exact total net sales be in fiscal 2026?",
                    "top_k": 5,
                },
            )
            assert refusal.status_code == 200 and refusal.json()["abstained"]
            report["explicit_abstention"] = refusal.json()
            invalid = client.post(
                "/documents",
                files={
                    "file": (
                        "wrong.txt",
                        b"this is not PDF",
                        "text/plain",
                    )
                },
            )
            assert invalid.status_code == 415
            report["invalid_upload_status"] = invalid.status_code
    report["checks_passed"] = True
    (root / "eval/phase3/live_smoke.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Live route smoke passed: PDF upload, health, complete answer, "
        "source quote/page, abstention"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Live smoke failed ({type(exc).__name__}); upstream details suppressed.")
        raise SystemExit(1) from None
