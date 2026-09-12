"""Embedding provider selection.

One place decides which provider is live, so the rest of the app depends on the
interface (name, model, dimensions, embed_documents, embed_query) and never on
which vendor is configured.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..config import settings
from ..logging_setup import get_logger
from ..providers.local_embeddings import LocalEmbeddingProvider
from ..providers.openai_embeddings import OpenAIEmbeddingProvider

logger = get_logger(__name__)


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimensions: int | None
    default_min_score: float

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...

    async def embed_query(self, text: str) -> np.ndarray: ...


_FACTORIES = {
    "openai": OpenAIEmbeddingProvider,
    "local": LocalEmbeddingProvider,
}

_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _provider
    if _provider is not None:
        return _provider

    name = settings.embedding_provider_name
    factory = _FACTORIES.get(name)
    if factory is None:
        raise RuntimeError(
            f'Unknown EMBEDDING_PROVIDER "{name}". Supported: {", ".join(_FACTORIES)}.'
        )

    _provider = factory()  # type: ignore[assignment]
    logger.info("embedding provider ready", extra={"provider": _provider.name, "model": _provider.model})

    if _provider.name == "local":
        logger.warning(
            "using the local lexical embedder - matches wording, not meaning. "
            "Set OPENAI_API_KEY for semantic search."
        )
    return _provider


async def close_embedding_provider() -> None:
    global _provider
    if _provider is not None and hasattr(_provider, "aclose"):
        await _provider.aclose()  # type: ignore[attr-defined]
    _provider = None


def reset_embedding_provider() -> None:
    """Test seam: drop the memoized provider so a different one can be selected."""
    global _provider
    _provider = None
