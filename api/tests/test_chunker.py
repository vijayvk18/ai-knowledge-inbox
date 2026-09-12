from app.rag.chunker import chunk_text, normalize_text


def test_keeps_a_short_note_as_a_single_chunk():
    _, chunks = chunk_text("A short note about the launch.")
    assert len(chunks) == 1
    assert chunks[0].content == "A short note about the launch."
    assert (chunks[0].char_start, chunks[0].char_end) == (0, 30)


def test_returns_nothing_for_whitespace_only_input():
    assert chunk_text("   \n\n  ")[1] == []


def test_splits_on_paragraph_boundaries_before_character_counts():
    paragraph = lambda label: f"{label}. " + "word " * 50  # noqa: E731
    doc = "\n\n".join([paragraph("One"), paragraph("Two"), paragraph("Three")])

    _, chunks = chunk_text(doc, target_chars=300, overlap_chars=0)

    # Each paragraph is ~270 chars, so each should land in its own chunk.
    assert len(chunks) == 3
    assert chunks[0].content.startswith("One.")
    assert chunks[1].content.startswith("Two.")


def test_every_chunk_after_the_first_overlaps_its_predecessor():
    doc = "\n\n".join(f"Para {i}. " + "word " * 40 for i in range(6))
    text, chunks = chunk_text(doc, target_chars=400, overlap_chars=100)

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert current.char_start < previous.char_end

    # Offsets must address the returned normalized text, not the input.
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end].strip() == chunk.content


def test_hard_splits_text_with_no_usable_boundary():
    _, chunks = chunk_text("x" * 3000, target_chars=500, overlap_chars=0)
    assert len(chunks) >= 6
    assert all(len(chunk.content) <= 500 for chunk in chunks)


def test_merges_a_trailing_runt_chunk():
    doc = "word " * 100 + "\n\nend."
    _, chunks = chunk_text(doc, target_chars=400, overlap_chars=0, min_chars=120)
    assert chunks[-1].content.endswith("end.")
    assert len(chunks[-1].content) > len("end.")


def test_normalize_collapses_blank_lines_and_line_endings():
    assert normalize_text("a\r\n\r\n\r\n\r\nb") == "a\n\nb"
