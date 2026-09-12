"""HTML -> readable text.

The blank lines this produces are load-bearing: the chunker splits on them, so
losing paragraph structure here silently degrades every chunk and every citation
snippet downstream. That coupling is what most of these tests are about.
"""

import pytest

from app.ingest.html_extract import extract_from_plain_text, extract_readable_text
from app.rag.chunker import chunk_text

ARTICLE = """
<html>
  <head>
    <title>Fallback Title</title>
    <meta property="og:title" content="Preferred Title">
  </head>
  <body>
    <nav>Home About Contact</nav>
    <header>Site banner</header>
    <article>
      <h1>Deploy freeze</h1>
      <p>The freeze runs from December 18 through January 5.</p>
      <p>Priya owns the launch checklist.</p>
      <ul><li>Revert the flag</li><li>Page the on-call SRE</li></ul>
    </article>
    <footer>Copyright 2026</footer>
    <script>console.log("tracking")</script>
    <style>body { color: red }</style>
  </body>
</html>
"""


def test_prefers_og_title_over_the_title_tag():
    assert extract_readable_text(ARTICLE).title == "Preferred Title"


def test_falls_back_to_the_title_tag():
    html = "<html><head><title>Just This</title></head><body><p>text</p></body></html>"
    assert extract_readable_text(html).title == "Just This"


def test_falls_back_to_the_first_h1():
    html = "<html><body><h1>Heading Only</h1><p>text</p></body></html>"
    assert extract_readable_text(html).title == "Heading Only"


def test_returns_none_when_there_is_no_title_anywhere():
    assert extract_readable_text("<html><body><p>text</p></body></html>").title is None


@pytest.mark.parametrize(
    "chrome", ["Home About Contact", "Site banner", "Copyright 2026", "tracking", "color: red"]
)
def test_navigation_scripts_and_styles_are_dropped(chrome):
    assert chrome not in extract_readable_text(ARTICLE).text


def test_keeps_the_actual_content():
    text = extract_readable_text(ARTICLE).text

    assert "December 18 through January 5" in text
    assert "Priya owns the launch checklist" in text
    assert "Page the on-call SRE" in text


def test_block_elements_become_blank_lines_the_chunker_can_split_on():
    """The contract with the chunker: paragraphs stay separable."""
    text = extract_readable_text(ARTICLE).text

    assert "\n\n" in text, "no paragraph boundaries survived extraction"
    paragraphs = [block for block in text.split("\n\n") if block.strip()]
    assert len(paragraphs) >= 3


def test_adjacent_blocks_do_not_run_together():
    html = "<body><p>First sentence.</p><p>Second sentence.</p></body>"
    text = extract_readable_text(html).text

    assert "First sentence.Second" not in text
    assert "sentence.Second" not in text


def test_extraction_feeds_chunking_end_to_end():
    extracted = extract_readable_text(ARTICLE)
    _, chunks = chunk_text(extracted.text)

    assert chunks, "extracted article produced no chunks"
    assert "December 18" in " ".join(chunk.content for chunk in chunks)


def test_entities_are_decoded():
    html = "<body><p>Tom &amp; Jerry &lt;3 caf&eacute;</p></body>"
    assert "Tom & Jerry <3 café" in extract_readable_text(html).text


def test_whitespace_is_collapsed_without_losing_structure():
    html = "<body><p>lots     of\n\n   space</p><p>next</p></body>"
    text = extract_readable_text(html).text

    assert "lots of space" in text
    assert "\n\n" in text


def test_a_page_with_no_main_container_still_extracts_from_body():
    html = "<body><div><p>No article tag here, but real content all the same.</p></div></body>"
    assert "real content" in extract_readable_text(html).text


def test_empty_html_yields_empty_text():
    assert extract_readable_text("<html><body></body></html>").text == ""


def test_malformed_html_does_not_raise():
    """Real pages are broken; extraction must degrade, not explode."""
    result = extract_readable_text("<html><body><p>unclosed <div>tags <span>everywhere")
    assert "unclosed" in result.text


def test_plain_text_passes_through_with_a_title_guessed_from_the_url():
    result = extract_from_plain_text("  raw notes here  ", "https://example.com/docs/runbook.txt")

    assert result.text == "raw notes here"
    assert result.title == "runbook.txt"


def test_plain_text_title_falls_back_to_the_hostname():
    assert extract_from_plain_text("body", "https://example.com").title == "example.com"


def test_source_line_wrapping_does_not_invent_paragraphs():
    """Most real HTML wraps long paragraphs across source lines."""
    html = (
        "<body><p>This is one long paragraph that the author\n"
        "        happened to wrap across several source lines\n"
        "        for readability.</p>\n\n"
        "<p>This is genuinely a second paragraph.</p></body>"
    )
    text = extract_readable_text(html).text
    paragraphs = [block for block in text.split("\n\n") if block.strip()]

    assert len(paragraphs) == 2, f"expected 2 paragraphs, got {len(paragraphs)}: {paragraphs!r}"
    assert "wrap across several source lines for readability." in paragraphs[0]


def test_wrapped_html_chunks_as_one_paragraph():
    """The consequence that matters: false boundaries would fragment chunks."""
    body = "<p>" + "\n          ".join(["Sentence number %d about the deploy freeze." % i for i in range(6)]) + "</p>"
    extracted = extract_readable_text(f"<body>{body}</body>")
    _, chunks = chunk_text(extracted.text)

    assert len(chunks) == 1, "a single wrapped paragraph must not split into several chunks"
