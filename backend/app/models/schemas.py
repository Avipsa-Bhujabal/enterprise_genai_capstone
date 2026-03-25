from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=10)


class Citation(BaseModel):
    source: str
    chunk_id: str
    preview: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    agent_trace: list[str]


class IngestResponse(BaseModel):
    message: str
    files_processed: int
    chunks_created: int
    sources: list[str]
