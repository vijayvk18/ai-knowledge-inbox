"""HTML -> readable text.

A light readability pass: drop chrome (nav, scripts, footers), prefer the
main/article container when the page has one, then walk the DOM emitting text
with newlines at block boundaries. Block boundaries matter downstream - the
chunker splits on blank lines, so preserving paragraph structure here is what
lets it split on meaning instead of on character counts.

Uses the stdlib html.parser via BeautifulSoup: no lxml build step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

_DROP_SELECTORS = [
    "script", "style", "noscript", "template", "svg", "iframe", "form",
    "nav", "header", "footer", "aside",
    '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]',
    '[aria-hidden="true"]', ".advertisement", ".ad", ".cookie-banner",
]

# Containers that usually hold the article, best guess first.
_MAIN_SELECTORS = ["article", "main", '[role="main"]', "#content", ".post", ".entry-content"]

_BLOCK_TAGS = frozenset(
    """address article blockquote br div dd dl dt figcaption figure footer
    h1 h2 h3 h4 h5 h6 header hr li main ol p pre section table tr ul""".split()
)

_ANY_WHITESPACE = re.compile(r"\s+")
_MULTI_SPACE = re.compile(r"[^\S\n]+")
_SPACED_NEWLINE = re.compile(r" *\n *")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


@dataclass(slots=True)
class ExtractedContent:
    title: str | None
    text: str


def _collect_text(node, out: list[str]) -> None:
    if isinstance(node, NavigableString):
        # Collapse *all* whitespace inside a text node, newlines included. In
        # HTML a source newline inside a paragraph is just a space; keeping it
        # would invent paragraph breaks from the author's line wrapping, and the
        # chunker splits on blank lines - so hand-wrapped markup would silently
        # fragment into false chunks. Block boundaries below are the only thing
        # allowed to introduce a newline.
        out.append(_ANY_WHITESPACE.sub(" ", str(node)))
        return
    if not isinstance(node, Tag):
        return

    is_block = node.name in _BLOCK_TAGS
    if is_block:
        out.append("\n")
    for child in node.children:
        _collect_text(child, out)
    if is_block:
        out.append("\n")


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACED_NEWLINE.sub("\n", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()


def extract_readable_text(html: str) -> ExtractedContent:
    soup = BeautifulSoup(html, "html.parser")

    title = None
    og_title = soup.select_one('meta[property="og:title"]')
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text().strip()

    for selector in _DROP_SELECTORS:
        for element in soup.select(selector):
            element.decompose()

    container = None
    for selector in _MAIN_SELECTORS:
        candidate = soup.select_one(selector)
        if candidate is not None and len(candidate.get_text(strip=True)) > 200:
            container = candidate
            break
    if container is None:
        container = soup.body or soup

    parts: list[str] = []
    _collect_text(container, parts)

    return ExtractedContent(title=title or None, text=_tidy("".join(parts)))


def extract_from_plain_text(body: str, url: str) -> ExtractedContent:
    """Plain-text and JSON responses need no extraction, only a title guess."""
    parsed = urlparse(url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    return ExtractedContent(title=segments[-1] if segments else parsed.hostname, text=body.strip())
