"""Chunking.

Strategy: structure-aware packing with a sliding overlap.

1. Split the document on blank lines into paragraphs - the author's own units
   of meaning. A paragraph rarely changes topic halfway through, so paragraph
   boundaries are the cheapest good boundaries available.
2. Greedily pack whole paragraphs into a chunk until adding the next would
   exceed the target size.
3. A paragraph bigger than the target is split on sentence boundaries, and a
   sentence bigger than the target is hard-split at word boundaries. These
   fallbacks exist so one wall-of-text input cannot produce a chunk too large
   to embed.
4. Each chunk carries a tail of the previous chunk as overlap, so a fact that
   straddles a boundary is still retrievable from both sides.
5. A trailing runt chunk is merged back into its predecessor.

Why not fixed-size windows? They cut mid-sentence, which produces embeddings of
fragments and citation snippets that read as broken - and the snippet is what
the user judges the answer by. Why not semantic/embedding-based splitting? It
costs an embedding pass per candidate boundary for a quality gain that does not
show up on notes and articles of this size.

Offsets are relative to the normalized text stored as the item's raw content,
so a citation can address an exact span of what the user sees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import settings

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_BEFORE_NEWLINE = re.compile(r"[ \t]+\n")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(slots=True)
class TextChunk:
    content: str
    char_start: int
    char_end: int


def normalize_text(text: str) -> str:
    """Normalize line endings and collapse runs of blank lines.

    Offsets are computed against the result, which is also what gets stored.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_BEFORE_NEWLINE.sub("\n", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _split_with_offsets(text: str, start: int, end: int, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    """Split a span at ``pattern``, returning [start, end) offsets into ``text``."""
    segment = text[start:end]
    spans: list[tuple[int, int]] = []
    cursor = 0

    for piece in pattern.split(segment):
        position = segment.find(piece, cursor)
        if position == -1:
            continue
        if piece.strip():
            spans.append((start + position, start + position + len(piece)))
        cursor = position + len(piece)

    return spans


def _split_long_span(text: str, span: tuple[int, int], maximum: int) -> list[tuple[int, int]]:
    """Hard-split an oversized span at word boundaries so nothing exceeds ``maximum``."""
    spans: list[tuple[int, int]] = []
    start, end = span

    while end - start > maximum:
        window = text[start : start + maximum]
        break_at = max(window.rfind(" "), window.rfind("\n"))
        stop = start + break_at if break_at > maximum * 0.5 else start + maximum
        spans.append((start, stop))
        start = stop

    if end > start:
        spans.append((start, end))
    return spans


def _to_segments(text: str, target_chars: int) -> list[tuple[int, int]]:
    """Paragraphs, subdivided until every segment fits the target size."""
    segments: list[tuple[int, int]] = []

    for paragraph in _split_with_offsets(text, 0, len(text), _PARAGRAPH_SPLIT):
        if paragraph[1] - paragraph[0] <= target_chars:
            segments.append(paragraph)
            continue

        # Sentence boundary: terminator + whitespace. Deliberately naive - it is
        # a fallback path, and the hard split below catches what it misses.
        for sentence in _split_with_offsets(text, paragraph[0], paragraph[1], _SENTENCE_SPLIT):
            if sentence[1] - sentence[0] <= target_chars:
                segments.append(sentence)
            else:
                segments.extend(_split_long_span(text, sentence, target_chars))

    return segments


def _overlap_start(text: str, start: int, overlap_chars: int) -> int:
    """Walk back up to ``overlap_chars``, snapping to a word boundary."""
    if overlap_chars <= 0 or start == 0:
        return start

    begin = max(0, start - overlap_chars)
    match = re.search(r"\s", text[begin:start])
    return begin if match is None else begin + match.start() + 1


def chunk_text(
    raw_text: str,
    *,
    target_chars: int | None = None,
    overlap_chars: int | None = None,
    min_chars: int | None = None,
) -> tuple[str, list[TextChunk]]:
    """Return the normalized text and its chunks."""
    target = target_chars if target_chars is not None else settings.chunk_target_chars
    overlap = overlap_chars if overlap_chars is not None else settings.chunk_overlap_chars
    minimum = min_chars if min_chars is not None else settings.chunk_min_chars

    text = normalize_text(raw_text)
    if not text:
        return text, []

    spans: list[list[int]] = []
    current: list[int] | None = None

    for segment_start, segment_end in _to_segments(text, target):
        if current is not None and segment_end - current[0] > target:
            spans.append(current)
            current = None
        if current is None:
            current = [segment_start, segment_end]
        else:
            current[1] = segment_end

    if current is not None:
        spans.append(current)

    # Merge a trailing runt: a 20-character last chunk is noise in the index.
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < minimum:
        spans[-2][1] = spans[-1][1]
        spans.pop()

    chunks: list[TextChunk] = []
    for index, (span_start, span_end) in enumerate(spans):
        start = span_start if index == 0 else _overlap_start(text, span_start, overlap)
        content = text[start:span_end].strip()
        if content:
            chunks.append(TextChunk(content=content, char_start=start, char_end=span_end))

    return text, chunks
