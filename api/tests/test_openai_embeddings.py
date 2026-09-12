"""OpenAI embedding provider.

Like the answer provider, this path has never run against the real API, so the
tests cover our half of it: batching, how we read the response, and how every
failure maps onto our error types. httpx's MockTransport stands in for the
network so the request we actually send can be inspected.
"""

import httpx
import numpy as np
import pytest

from app.config import settings
from app.errors import UpstreamError
from app.providers.openai_embeddings import OpenAIEmbeddingProvider


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "openai_embedding_model", "text-embedding-3-small")
    return OpenAIEmbeddingProvider()


def _install(provider, handler):
    """Route the provider's client at a mock transport, recording every request."""
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(record),
        base_url=settings.openai_base_url,
        headers={"authorization": f"Bearer {settings.openai_api_key}"},
    )
    return requests


def _embeddings_response(vectors, *, shuffle=False):
    data = [{"index": i, "embedding": vector} for i, vector in enumerate(vectors)]
    if shuffle:  # the API documents order, but `index` is authoritative
        data = list(reversed(data))
    return httpx.Response(200, json={"data": data, "model": "text-embedding-3-small"})


# ---------- the request we send ----------


async def test_sends_the_model_and_input_to_the_embeddings_endpoint(provider):
    import json

    requests = _install(provider, lambda _: _embeddings_response([[1.0, 0.0], [0.0, 1.0]]))

    await provider.embed_documents(["first", "second"])

    assert len(requests) == 1
    assert requests[0].url.path.endswith("/embeddings")
    assert requests[0].headers["authorization"] == "Bearer sk-test"

    body = json.loads(requests[0].content)
    assert body["model"] == "text-embedding-3-small"
    assert body["input"] == ["first", "second"]


async def test_documents_are_batched_to_the_configured_size(provider, monkeypatch):
    import json

    monkeypatch.setattr(settings, "embedding_batch_size", 2)
    requests = _install(
        provider,
        lambda request: _embeddings_response([[1.0, 0.0]] * len(json.loads(request.content)["input"])),
    )

    vectors = await provider.embed_documents(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert len(requests) == 3, "5 documents at batch size 2 is three round trips"
    assert [len(json.loads(r.content)["input"]) for r in requests] == [2, 2, 1]


# ---------- how we read the response ----------


async def test_vectors_are_returned_in_input_order_not_response_order(provider):
    """`index` is authoritative: a reordered response must not mis-assign vectors."""
    _install(provider, lambda _: _embeddings_response([[1.0, 0.0], [0.0, 1.0]], shuffle=True))

    first, second = await provider.embed_documents(["first", "second"])

    assert first.tolist() == [1.0, 0.0]
    assert second.tolist() == [0.0, 1.0]


async def test_vectors_come_back_as_float32(provider):
    _install(provider, lambda _: _embeddings_response([[0.5, 0.25]]))

    (vector,) = await provider.embed_documents(["only"])

    assert vector.dtype == np.float32, "the store writes raw float32 bytes"


async def test_embed_query_returns_a_single_vector(provider):
    _install(provider, lambda _: _embeddings_response([[0.1, 0.2, 0.3]]))

    vector = await provider.embed_query("a question")

    assert vector.tolist() == pytest.approx([0.1, 0.2, 0.3])


# ---------- how upstream failures map ----------


@pytest.mark.parametrize(
    ("status", "expected_status", "expected_code"),
    [
        (429, 429, "embedding_rate_limited"),
        (500, 502, "embedding_failed"),
        (503, 502, "embedding_failed"),
        (401, 502, "embedding_failed"),
    ],
)
async def test_http_errors_map_to_our_error_types(provider, status, expected_status, expected_code):
    _install(provider, lambda _: httpx.Response(status, json={"error": {"message": "nope"}}))

    with pytest.raises(UpstreamError) as excinfo:
        await provider.embed_documents(["text"])

    assert excinfo.value.status == expected_status
    assert excinfo.value.code == expected_code


async def test_a_timeout_maps_to_504(provider):
    def timeout(_):
        raise httpx.ReadTimeout("too slow")

    _install(provider, timeout)

    with pytest.raises(UpstreamError) as excinfo:
        await provider.embed_documents(["text"])

    assert excinfo.value.status == 504
    assert excinfo.value.code == "embedding_timeout"


async def test_a_connection_failure_maps_to_504(provider):
    def unreachable(_):
        raise httpx.ConnectError("no route")

    _install(provider, unreachable)

    with pytest.raises(UpstreamError) as excinfo:
        await provider.embed_documents(["text"])

    assert excinfo.value.code == "embedding_unreachable"


async def test_missing_credentials_fail_at_construction_not_at_query_time(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None)

    with pytest.raises(RuntimeError) as excinfo:
        OpenAIEmbeddingProvider()
    assert "OPENAI_API_KEY" in str(excinfo.value)


# ---------- identity, which the staleness check depends on ----------


def test_known_models_report_their_dimensions(provider):
    assert provider.dimensions == 1536
    assert provider.name == "openai"
    assert provider.model == "text-embedding-3-small"


def test_an_unknown_model_reports_unknown_dimensions_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "openai_embedding_model", "text-embedding-4-enormous")

    assert OpenAIEmbeddingProvider().dimensions is None
