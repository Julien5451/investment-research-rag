"""Public API contracts; source identities are assigned by the server."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class Citation(BaseModel):
    source_id: str
    document_id: str
    document_name: str
    page: int
    chunk_id: str
    quote: str
    relevance_score: float


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[Citation] = Field(default_factory=list)
    reason: str | None = None


class DocumentResponse(BaseModel):
    document_id: str
    document_name: str
    pages: int
    chunks: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
