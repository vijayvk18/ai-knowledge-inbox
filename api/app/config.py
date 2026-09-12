"""Central configuration.

Everything tunable lives here, so the whole knob surface fits on one screen and
no other module reads os.environ directly. Values come from the environment or
from api/.env; real environment variables always win.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- server ----------
    env: str = Field(default="development", alias="APP_ENV")
    port: int = 8787
    cors_origins: str = "http://localhost:5173"
    log_level: str = "info"
    # Pretty lines locally; JSON lines are what a log shipper wants.
    log_pretty: bool = True

    # ':memory:' is supported and is what the tests use.
    db_file: str = str(PROJECT_ROOT / "data" / "knowledge-inbox.db")

    # ---------- ingestion ----------
    ingest_concurrency: int = 2
    max_note_chars: int = 100_000
    url_fetch_timeout_seconds: float = 10.0
    url_fetch_max_bytes: int = 2_000_000
    url_fetch_max_redirects: int = 3
    url_fetch_user_agent: str = "KnowledgeInboxBot/1.0 (+https://example.invalid)"
    # Disables the SSRF guard. Local testing only.
    url_fetch_allow_private: bool = False

    # ---------- chunking ----------
    # ~1000 chars ~= 250 tokens: small enough that a retrieved chunk is mostly
    # signal, large enough to hold a whole idea.
    chunk_target_chars: int = 1000
    # Overlap keeps a sentence that straddles a boundary retrievable from both sides.
    chunk_overlap_chars: int = 150
    chunk_min_chars: int = 120

    # ---------- embeddings ----------
    # "openai" | "local". Resolved in embedding_provider_name below so the app
    # still boots, and stays fully usable, with no credentials at all.
    embedding_provider: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 64
    embedding_timeout_seconds: float = 30.0
    # 2048, not 512: the hashing trick collides, and at 512 a collision can make
    # an irrelevant chunk outrank the right one (measured, not theoretical).
    # Costs 8 KB per chunk instead of 2 KB - cheap for a dev-time fallback.
    local_embedding_dimensions: int = 2048

    # ---------- answers ----------
    # "anthropic" | "echo"
    answer_provider: str | None = None
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"
    answer_max_tokens: int = 2000
    # Grounded Q&A over a handful of chunks is not deep reasoning; low effort
    # keeps p95 latency down. Raise for harder corpora.
    answer_effort: str = "low"
    answer_timeout_seconds: float = 60.0

    # ---------- retrieval ----------
    retrieval_top_k: int = 5
    retrieval_max_top_k: int = 20
    # Left unset so each embedding provider supplies its own floor: dense and
    # lexical scores are not on the same scale.
    retrieval_min_score: float | None = None
    # Cap on chunks from any single item, so one long article cannot crowd every
    # other source out of the context window.
    retrieval_max_per_item: int = 3
    retrieval_max_context_chars: int = 12_000

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def embedding_provider_name(self) -> str:
        if self.embedding_provider:
            return self.embedding_provider
        return "openai" if self.openai_api_key else "local"

    @property
    def answer_provider_name(self) -> str:
        if self.answer_provider:
            return self.answer_provider
        return "anthropic" if self.anthropic_api_key else "echo"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
