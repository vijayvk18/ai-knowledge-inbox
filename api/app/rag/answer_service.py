"""The query pipeline: retrieve -> ground -> generate -> cite.

Retrieval outcomes are graded rather than binary, because "I found nothing"
is a bad answer to give someone with a five-item inbox:

* **confident** - at least one chunk cleared the relevance floor;
* **low confidence** - nothing cleared it, but something shares wording with
  the question, so the best few are answered over and flagged;
* **no match at all** - not one chunk shares a word; ``answered: false``;
* **empty inbox** - nothing saved yet; ``answered: false``.

The last two return 200 with an empty source list rather than inventing an
answer or raising a 404: both are ordinary states for a knowledge inbox, and
the UI renders them as notices.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..config import settings
from ..logging_setup import get_logger
from ..providers.anthropic_answers import AnthropicAnswerProvider
from ..providers.echo_answers import EchoAnswerProvider
from .retriever import retrieve
from .vector_store import vector_store

logger = get_logger(__name__)

_CITATION = re.compile(r"\[(\d{1,2})\]")

# Floor for the low-confidence retry. Strictly above zero: a chunk sharing no
# word at all with the question is not a weak match, it is not a match, and
# handing it to the model invites an answer grounded in nothing.
_MINIMUM_USEFUL_SCORE = 0.01

_FACTORIES = {
    "anthropic": AnthropicAnswerProvider,
    "echo": EchoAnswerProvider,
}

_provider = None


def get_answer_provider():
    global _provider
    if _provider is not None:
        return _provider

    name = settings.answer_provider_name
    factory = _FACTORIES.get(name)
    if factory is None:
        raise RuntimeError(f'Unknown ANSWER_PROVIDER "{name}". Supported: {", ".join(_FACTORIES)}.')

    _provider = factory()
    logger.info("answer provider ready", extra={"provider": _provider.name, "model": _provider.model})
    if _provider.name == "echo":
        logger.warning(
            "no answer model configured - /query will return the top passage instead of a generated answer"
        )
    return _provider


async def close_answer_provider() -> None:
    global _provider
    if _provider is not None:
        await _provider.aclose()
    _provider = None


def reset_answer_provider() -> None:
    """Test seam."""
    global _provider
    _provider = None


def _cited_numbers(text: str) -> set[int]:
    """Which citation numbers the model actually used, so the UI can dim the rest."""
    return {int(match.group(1)) for match in _CITATION.finditer(text)}


async def answer_question(*, question: str, top_k: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()

    if vector_store.size() == 0:
        return {
            "answered": False,
            "answer": "There is nothing in your inbox yet. Save a note or a URL first, then ask again.",
            "reason": "empty_index",
            "sources": [],
            "stats": {
                "indexedChunks": 0,
                "retrievedChunks": 0,
                "totalMs": round((time.perf_counter() - started) * 1000),
            },
        }

    passages, stats = await retrieve(question=question, top_k=top_k)
    low_confidence = False

    if not passages:
        # Nothing cleared the relevance floor. Refusing outright is the wrong
        # call: the floor is tuned to keep noise out of a large corpus, but in a
        # small inbox the best chunk is often just below it, and the user can
        # see for themselves whether it answers their question. So retry without
        # the floor, hand the model the best few, and let it say what is missing
        # - which the system prompt already requires it to do.
        passages, stats = await retrieve(
            question=question,
            top_k=min(top_k or settings.retrieval_top_k, 3),
            min_score=_MINIMUM_USEFUL_SCORE,
        )
        low_confidence = True

        if not passages:
            logger.info("query matched no chunks at all", extra=stats)
            return {
                "answered": False,
                "answer": (
                    "Nothing in your saved items matches that question at all. "
                    "Try rephrasing, or save a source that covers it."
                ),
                "reason": "no_relevant_context",
                "sources": [],
                "stats": {
                    **stats,
                    "retrievedChunks": 0,
                    "totalMs": round((time.perf_counter() - started) * 1000),
                },
            }

        logger.info(
            "no passage cleared the relevance floor - answering at low confidence",
            extra={**stats, "topScore": passages[0]["score"]},
        )

    provider = get_answer_provider()
    generation_started = time.perf_counter()
    generated = await provider.generate(question=question, passages=passages)
    generation_ms = round((time.perf_counter() - generation_started) * 1000)

    cited = _cited_numbers(generated.text)

    logger.info(
        "query answered",
        extra={
            "retrievedChunks": len(passages),
            "topScore": passages[0]["score"],
            "citedCount": len(cited),
            "generationMs": generation_ms,
            "answerProvider": provider.name,
        },
    )

    return {
        "answered": True,
        "answer": generated.text,
        "truncated": generated.truncated,
        # True when nothing cleared the relevance floor and these are best-effort
        # matches. The UI warns; the answer is still grounded and cited.
        "lowConfidence": low_confidence,
        "sources": [
            {
                "citation": passage["citation"],
                "itemId": passage["item_id"],
                "chunkId": passage["chunk_id"],
                "sourceType": passage["source_type"],
                "title": passage["title"],
                "url": passage["url"],
                "snippet": passage["content"],
                "score": passage["score"],
                "charRange": [passage["char_start"], passage["char_end"]],
                # False means the model did not lean on this passage - useful when
                # tuning topK, and the UI dims these.
                "cited": passage["citation"] in cited,
            }
            for passage in passages
        ],
        "stats": {
            **stats,
            "retrievedChunks": len(passages),
            "generationMs": generation_ms,
            "answerProvider": provider.name,
            "answerModel": generated.usage["model"],
            "inputTokens": generated.usage["input_tokens"],
            "outputTokens": generated.usage["output_tokens"],
            "totalMs": round((time.perf_counter() - started) * 1000),
        },
    }
