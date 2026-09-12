"""Request and response models.

Validation lives at the edge: routers and services can assume well-formed
input, and every rejection returns the same envelope with per-field details
(see errors.register_exception_handlers).

Responses are modelled too, so the OpenAPI schema FastAPI generates at /docs is
accurate rather than a pile of ``dict``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field

from .config import settings


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


# Applied per field rather than as a wildcard validator: pydantic forbids a
# mode="before" validator on a discriminator field, and trimming `type` would
# be meaningless anyway.
TrimmedStr = Annotated[str, BeforeValidator(_strip)]
OptionalTrimmedStr = Annotated[str | None, BeforeValidator(_strip)]


# ---------- ingest ----------


class NoteIngestRequest(BaseModel):
    type: Literal["note"]
    content: TrimmedStr = Field(min_length=1, max_length=settings.max_note_chars)
    title: OptionalTrimmedStr = Field(default=None, max_length=200)


class UrlIngestRequest(BaseModel):
    type: Literal["url"]
    url: TrimmedStr = Field(min_length=1)
    title: OptionalTrimmedStr = Field(default=None, max_length=200)


# A discriminated union on `type` keeps the two ingestion modes honest: you
# cannot post a note with a url field and have it quietly ignored, and the error
# message names which variant failed.
IngestRequest = Annotated[NoteIngestRequest | UrlIngestRequest, Field(discriminator="type")]


# ---------- query ----------


class QueryRequest(BaseModel):
    question: TrimmedStr = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=settings.retrieval_top_k, ge=1, le=settings.retrieval_max_top_k, alias="topK")

    model_config = {"populate_by_name": True}


# ---------- responses ----------


class ItemError(BaseModel):
    code: str
    message: str


class ItemSummary(BaseModel):
    id: str
    sourceType: str
    title: str | None
    url: str | None
    status: str
    error: ItemError | None
    charCount: int
    chunkCount: int
    preview: str | None
    createdAt: str
    updatedAt: str
    processedAt: str | None


class ItemDetail(ItemSummary):
    content: str | None


class IngestResponse(BaseModel):
    item: ItemSummary
    message: str


class ItemResponse(BaseModel):
    item: ItemDetail


class ItemListResponse(BaseModel):
    items: list[ItemSummary]
    nextCursor: str | None
    counts: dict[str, int]


class Source(BaseModel):
    citation: int
    itemId: str
    chunkId: str
    sourceType: str | None
    title: str | None
    url: str | None
    snippet: str
    score: float
    charRange: list[int]
    cited: bool


class QueryResponse(BaseModel):
    question: str
    answered: bool
    answer: str
    reason: str | None = None
    truncated: bool | None = None
    lowConfidence: bool | None = None
    sources: list[Source]
    stats: dict[str, Any]


class SearchResult(BaseModel):
    itemId: str
    chunkId: str
    chunkIndex: int
    score: float
    title: str | None
    url: str | None
    sourceType: str | None
    snippet: str


class SearchResponse(BaseModel):
    question: str
    results: list[SearchResult]
    stats: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    uptimeSeconds: int
    providers: dict[str, Any]
    index: dict[str, Any]
    items: dict[str, int]
    ingestion: dict[str, int]
