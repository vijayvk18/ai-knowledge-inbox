"""Answer generation with Claude.

The contract with the model is narrow on purpose: answer only from the numbered
context blocks, cite them as [1], [2], and say so when the context does not
contain the answer. Grounding is enforced by the prompt and made checkable by
the citations the UI renders beside the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic

from ..config import settings
from ..errors import UpstreamError
from ..logging_setup import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You answer questions strictly from a user's saved notes and saved web pages.

Rules:
- Use ONLY the numbered context blocks provided. Never use outside knowledge.
- Cite every claim with the bracketed number of the block it came from, e.g. [2]. Cite multiple as [1][3].
- If the context does not contain the answer, say plainly what is missing. Do not guess or fill gaps.
- If the context answers only part of the question, answer that part and state what is unsupported.
- Be direct and concise. No preamble, no restating the question."""


@dataclass(slots=True)
class GeneratedAnswer:
    text: str
    truncated: bool
    usage: dict[str, Any]


def _build_user_message(question: str, passages: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[{passage['citation']}] {passage['source_label']}\n{passage['content']}" for passage in passages
    )
    return f"Context blocks:\n\n{context}\n\n---\n\nQuestion: {question}"


class AnthropicAnswerProvider:
    name = "anthropic"

    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANSWER_PROVIDER=anthropic requires ANTHROPIC_API_KEY. "
                "Set it, or use ANSWER_PROVIDER=echo."
            )
        self.model = settings.anthropic_model
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.answer_timeout_seconds,
            max_retries=2,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def generate(self, *, question: str, passages: list[dict]) -> GeneratedAnswer:
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=settings.answer_max_tokens,
                # Grounded extraction over a few short blocks. Thinking stays on
                # (the default for this model family); low effort keeps latency sane.
                output_config={"effort": settings.answer_effort},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _build_user_message(question, passages)}],
            )
        except anthropic.RateLimitError as exc:
            raise UpstreamError(
                "The answer model is rate limited. Try again shortly.",
                code="answer_rate_limited",
                status=429,
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise UpstreamError(
                f"Answer generation timed out after {settings.answer_timeout_seconds}s",
                code="answer_timeout",
                status=504,
            ) from exc
        except anthropic.AuthenticationError as exc:
            raise UpstreamError(
                "The answer model rejected our credentials.", code="answer_unauthorized", status=502
            ) from exc
        except anthropic.APIError as exc:
            logger.error("answer provider failed", extra={"error": str(exc)})
            raise UpstreamError("Answer generation failed.", code="answer_failed") from exc

        # A policy decline arrives as HTTP 200 with stop_reason 'refusal'. Check it
        # before reading content, which is empty in that case.
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise UpstreamError(
                "The answer model declined to answer this question.",
                code="answer_refused",
                status=422,
                details={"category": category},
            )

        text = "\n".join(block.text for block in response.content if block.type == "text").strip()

        return GeneratedAnswer(
            text=text,
            truncated=response.stop_reason == "max_tokens",
            usage={
                "model": response.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
