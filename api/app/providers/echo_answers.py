"""Keyless answer provider.

Retrieval is the half of a RAG system most worth inspecting, and it needs no
LLM. With no ANTHROPIC_API_KEY set, POST /query still returns the full response
shape - ranked passages, citations, scores - with generated prose replaced by
the top passage and an explicit notice. Nothing downstream special-cases a
missing key.
"""

from __future__ import annotations

from .anthropic_answers import GeneratedAnswer


class EchoAnswerProvider:
    name = "echo"
    model = "none"

    async def aclose(self) -> None:  # symmetry with the real providers
        return None

    async def generate(self, *, question: str, passages: list[dict]) -> GeneratedAnswer:
        top = passages[0]
        lines = [
            f'No answer model is configured, so this is the highest-scoring retrieved passage '
            f'for "{question}" rather than a generated answer.',
            "",
            f"[{top['citation']}] {top['source_label']}",
            top["content"],
            "",
            "Set ANTHROPIC_API_KEY and restart the server to get a synthesized, cited answer.",
        ]
        return GeneratedAnswer(
            text="\n".join(lines),
            truncated=False,
            usage={"model": "none", "input_tokens": 0, "output_tokens": 0},
        )
