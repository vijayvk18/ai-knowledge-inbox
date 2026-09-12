"""OpenAI embedding provider.

One POST to /v1/embeddings via httpx rather than pulling in the SDK for a
single endpoint. Batched, timed out, and mapped onto our error types so the
route layer never sees a raw upstream failure.
"""

from __future__ import annotations

import httpx
import numpy as np

from ..config import settings
from ..errors import UpstreamError
from ..logging_setup import get_logger

logger = get_logger(__name__)

_DIMENSIONS_BY_MODEL = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider:
    name = "openai"
    default_min_score = 0.15

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY. "
                "Set it, or use EMBEDDING_PROVIDER=local."
            )
        self.model = settings.openai_embedding_model
        self.dimensions = _DIMENSIONS_BY_MODEL.get(self.model)
        self._client = httpx.AsyncClient(
            base_url=settings.openai_base_url,
            timeout=settings.embedding_timeout_seconds,
            headers={"authorization": f"Bearer {settings.openai_api_key}"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, inputs: list[str]) -> list[np.ndarray]:
        try:
            response = await self._client.post("/embeddings", json={"model": self.model, "input": inputs})
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                f"Embedding request timed out after {settings.embedding_timeout_seconds}s",
                code="embedding_timeout",
                status=504,
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(
                "Could not reach the embedding API", code="embedding_unreachable", status=504
            ) from exc

        if response.status_code >= 400:
            logger.error(
                "embedding api error",
                extra={"status": response.status_code, "body": response.text[:400]},
            )
            # 429 and 5xx are worth retrying upstream; 4xx is our request or our key.
            raise UpstreamError(
                f"Embedding API returned {response.status_code}",
                code="embedding_rate_limited" if response.status_code == 429 else "embedding_failed",
                status=429 if response.status_code == 429 else 502,
            )

        payload = response.json()
        # The API documents order preservation, but the index field is authoritative.
        entries = sorted(payload["data"], key=lambda entry: entry["index"])
        return [np.asarray(entry["embedding"], dtype=np.float32) for entry in entries]

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        size = settings.embedding_batch_size
        for start in range(0, len(texts), size):
            vectors.extend(await self._request(texts[start : start + size]))
        return vectors

    async def embed_query(self, text: str) -> np.ndarray:
        vectors = await self._request([text])
        return vectors[0]
