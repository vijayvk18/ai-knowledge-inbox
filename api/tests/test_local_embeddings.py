"""Retrieval-quality guards for the keyless embedder.

The closest thing here to a retrieval eval set. Every case exists because it
failed at some point:

* at 512 dimensions, hash collisions let an unrelated document outrank the
  correct one, which scored *negative* against a query containing its keyword;
* character 4-grams made "quarterly" match "starter" (shared "arte"/"rter")
  strongly enough to clear the relevance floor, so unrelated chunks were
  retrieved and cited.
"""

import numpy as np
import pytest

from app.providers.local_embeddings import LocalEmbeddingProvider

NOTE = "Aurora ships March 14. Rollback: revert flag aurora_enabled and page the on-call SRE."
PAGE = (
    "Example Domain This domain is for use in illustrative examples in documents. "
    "You may use this domain in literature without prior coordination or asking for permission."
)
HIRING = "The hiring loop has four stages: recruiter screen, technical screen, onsite, and debrief."


@pytest.fixture
def provider() -> LocalEmbeddingProvider:
    return LocalEmbeddingProvider()


async def score(provider: LocalEmbeddingProvider, query: str, document: str) -> float:
    query_vector = await provider.embed_query(query)
    (document_vector,) = await provider.embed_documents([document])
    return float(np.dot(query_vector, document_vector))


@pytest.mark.parametrize(
    ("query", "expected", "other"),
    [
        ("What is the rollback plan?", NOTE, PAGE),
        ("When does Aurora ship?", NOTE, PAGE),
        ("who do I page during an incident", NOTE, PAGE),
        ("hiring loops and debriefs", HIRING, NOTE),
        ("recruiter screens", HIRING, PAGE),
    ],
)
async def test_relevant_document_outranks_an_irrelevant_one(provider, query, expected, other):
    relevant = await score(provider, query, expected)
    irrelevant = await score(provider, query, other)

    assert relevant > irrelevant, f"{query!r} ranked the wrong document first"
    # Above the provider's own floor, or retrieval would discard it entirely.
    assert relevant > provider.default_min_score


@pytest.mark.parametrize(
    ("query", "document"),
    [
        ("recruiter screens", HIRING),       # plural -> singular
        ("hiring loops", HIRING),            # plural -> singular
        ("Aurora shipping", NOTE),           # -ing -> bare stem
    ],
)
async def test_plural_and_inflected_forms_still_match(provider, query, document):
    """Suffix stripping earns its keep here; exact word matching would score 0."""
    assert await score(provider, query, document) > 0.1


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("quarterly revenue forecast", "Sourdough starter needs feeding every twelve hours."),
        ("photosynthesis in arctic lichen", "The deploy freeze runs December 18 to January 5."),
        ("best pizza in Naples", NOTE),
        ("kubernetes ingress controller", HIRING),
    ],
)
async def test_texts_with_no_shared_words_score_zero(provider, left, right):
    """The n-gram regression: these scored up to 0.076 and cleared the floor."""
    assert abs(await score(provider, left, right)) < provider.default_min_score





async def test_embedding_is_deterministic_and_unit_length(provider):
    first = await provider.embed_query("stable across processes")
    second = await provider.embed_query("stable across processes")

    assert np.array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-5)
    assert first.shape[0] == provider.dimensions
