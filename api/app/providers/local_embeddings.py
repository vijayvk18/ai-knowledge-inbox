"""Local embedding provider - no credentials, no network, deterministic.

A hashed bag-of-words vector ("hashing trick"): lightly stemmed word tokens
hashed into a fixed-width vector with sub-linear term weighting, then
L2-normalized so cosine similarity is a dot product.

Be clear about what this is: lexical similarity, not semantic. It matches
"invoice deadline" to a chunk containing those words, and it will NOT match
"when is payment due". It exists so the whole pipeline - ingest, index,
retrieve, cite - is runnable and reviewable offline. Point EMBEDDING_PROVIDER
at openai for real semantics.

**Why stemming and not character n-grams.** The first version hashed character
4-grams alongside words, to tolerate plurals and typos. Measured, that traded
one problem for a worse one: "quarterly" and "starter" share the 4-grams "arte"
and "rter", so two texts with no word in common scored 0.076 - above the
relevance floor, so unrelated chunks were retrieved and cited. Light suffix
stripping delivers the plural/inflection tolerance that motivated n-grams
("budget caps" -> "budget cap") while scoring *exactly zero* on unrelated text,
which is what makes a relevance floor mean anything. The cost is typo tolerance:
"recuiter" no longer matches "recruiter". For searching your own notes, a false
match that produces a confidently wrong citation is worse than a missed typo.
"""

from __future__ import annotations

import math
import re

import numpy as np

from ..config import settings

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this "
    "to was what when where which who why will with your you do does did i me my we our "
    "can could should would there their them they he she his her".split()
)

_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)

# Order matters: longest suffix first, so "flies" -> "fly" not "flie".
_SUFFIXES = (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""))

# Keep a stem meaningful - never strip a word down to a fragment.
_MIN_STEM_LENGTH = 3


def _hash32(text: str) -> int:
    """FNV-1a: fast, well-distributed, and stable across processes."""
    h = 0x811C9DC5
    for char in text:
        h ^= ord(char) & 0xFF
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _stem(token: str) -> str:
    """Strip common English inflections. Deliberately crude - not a Porter stemmer.

    It only needs to make a query token and a document token collide when they
    are the same word in a different form, which is where most missed lexical
    matches come from in practice.
    """
    for suffix, replacement in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM_LENGTH:
            return token[: -len(suffix)] + replacement
    return token


def _tokenize(text: str) -> list[str]:
    cleaned = _NON_WORD.sub(" ", text.lower())
    return [
        _stem(token)
        for token in cleaned.split()
        if len(token) > 1 and token not in _STOPWORDS
    ]


def _embed(text: str, dimensions: int) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=np.float32)
    counts: dict[str, float] = {}

    for token in _tokenize(text):
        counts[token] = counts.get(token, 0.0) + 1.0

    for feature, count in counts.items():
        h = _hash32(feature)
        bucket = h % dimensions
        # Signed hashing: halves collision bias by letting colliding features cancel.
        sign = -1.0 if (h >> 31) & 1 else 1.0
        # Sub-linear term weighting - a word repeated 20 times is not 20x as relevant.
        vector[bucket] += sign * (1.0 + math.log(count))

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


class LocalEmbeddingProvider:
    name = "local"
    # Unrelated text scores ~0 under this scheme, so the floor exists to separate
    # "shares one incidental word" from "actually on topic" rather than to fight
    # noise. Anything below it still reaches the user through the low-confidence
    # path in answer_service, flagged rather than hidden.
    default_min_score = 0.10

    # Bump whenever tokenization or feature naming changes. Stored on every
    # chunk, so a change makes existing vectors detectably stale instead of
    # silently incomparable - dimensions stay the same width and cannot tell.
    version = "v2"

    def __init__(self) -> None:
        self.dimensions = settings.local_embedding_dimensions
        self.model = f"hashed-bow-{self.version}-{self.dimensions}"

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [_embed(text, self.dimensions) for text in texts]

    async def embed_query(self, text: str) -> np.ndarray:
        return _embed(text, self.dimensions)
