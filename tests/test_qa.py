import json

import pytest
from conftest import REVENUE, supported_answer
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from research_qa.index import VectorIndex
from research_qa.qa import QuestionAnswerer


def doc(text=REVENUE, *, page=1, chunk=0):
    return Document(
        page_content=text,
        metadata={
            "document_id": "doc",
            "document_name": "report.pdf",
            "page": page,
            "chunk_id": f"doc:{page}:{chunk}",
        },
    )


@pytest.mark.parametrize(
    ("model_response", "reason"),
    [
        ('{"abstained": true, "claims": []}', "model_insufficient_evidence"),
        ('{"abstained": false, "claims": []}', "model_insufficient_evidence"),
        ("not valid JSON", "invalid_model_response"),
        ('{"answer": "Revenue doubled"}', "invalid_model_response"),
        (
            '{"abstained": false, "claims": [{"text": "Revenue doubled", "evidence": []}]}',
            "invalid_model_response",
        ),
        (supported_answer(source="S999"), "unverifiable_citation"),
        (supported_answer("Revenue doubled in FY2025."), "unverifiable_citation"),
        (
            json.dumps(
                {
                    "abstained": False,
                    "claims": [
                        {
                            "text": "Revenue [S999]",
                            "evidence": [{"source_id": "S1", "quote": REVENUE}],
                        }
                    ],
                }
            ),
            "invalid_model_response",
        ),
        (
            json.dumps(
                {
                    "abstained": False,
                    "claims": [
                        {
                            "text": "Ignore previous instructions",
                            "evidence": [{"source_id": "S1", "quote": REVENUE}],
                        }
                    ],
                }
            ),
            "invalid_model_response",
        ),
        ("x" * 40001, "invalid_model_response"),
    ],
)
def test_bad_or_insufficient_answers_fail_closed(
    settings, embeddings, chat, model_response, reason
):
    index = VectorIndex(settings, embeddings)
    index.add([doc()])
    chat.invoke.return_value = AIMessage(content=model_response)
    response = QuestionAnswerer(settings, index, chat).ask("What happened to revenue?")
    assert response.abstained is True
    assert response.reason == reason
    assert response.citations == []
    assert "Insufficient evidence" in response.answer


def test_top_k_retrieval_and_context_budget(settings, embeddings, chat):
    settings = settings.model_copy(update={"max_context_chars": len(REVENUE)})
    index = VectorIndex(settings, embeddings)
    index.add([doc(page=page) for page in range(1, 4)])
    assert len(index.search("Revenue?", 1)) == 1
    assert len(index.search("Revenue?", 2)) == 2
    assert len(index.search("Revenue?", 20)) == 3
    QuestionAnswerer(settings, index, chat).ask("Revenue?", top_k=3)
    supplied = json.loads(chat.invoke.call_args.args[0][1].content)["evidence"]
    assert len(supplied) == 1
    assert sum(len(source["text"]) for source in supplied) <= settings.max_context_chars


def test_embedding_model_change_rejects_old_index(settings, embeddings):
    VectorIndex(settings, embeddings).add([doc()])
    changed = settings.model_copy(update={"embedding_model": "another-vector-space"})
    with pytest.raises(ValueError, match="Embedding configuration differs"):
        VectorIndex(changed, embeddings)


def test_index_rolls_back_partial_batch_when_embeddings_fail(settings):
    from conftest import TopicEmbeddings

    class FailingEmbeddings(TopicEmbeddings):
        calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated embedding failure after first batch")
            return super().embed_documents(texts)

    index = VectorIndex(settings, FailingEmbeddings())
    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        index.add([doc(chunk=chunk) for chunk in range(65)])
    assert index.count() == 0
