"""Server-side URL fetching.

The server fetching a user-supplied URL is a request-forgery primitive, so this
module is deliberately strict:

* http/https only
* every hop resolved and checked against private/loopback/link-local ranges
* redirects followed manually, so hop 2 cannot escape the hop-1 check
* a wall-clock timeout and a hard byte cap enforced while streaming
* an allowlist of textual content types

Each is a documented failure with its own error code, because "couldn't fetch
that page" is the most common ingestion failure and the user deserves to know
which one it was.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from ..config import settings
from ..errors import UpstreamError, ValidationError
from ..logging_setup import get_logger

logger = get_logger(__name__)

TEXTUAL_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "text/markdown",
    "application/json",
)


@dataclass(slots=True)
class FetchedPage:
    final_url: str
    content_type: str
    body: str
    truncated: bool


def parse_and_validate_url(raw: str) -> str:
    parsed = urlparse(raw.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValidationError(
            "Only http and https URLs can be fetched.", details={"protocol": parsed.scheme or None}
        )
    if not parsed.hostname:
        raise ValidationError("That is not a valid URL.", details={"url": raw})
    return parsed.geturl()


def _is_private(address: str) -> bool:
    """True for anything that is not publicly routable.

    ``is_global`` does the work, as an allowlist: anything not globally routable
    is refused. Two CPython quirks make the obvious alternatives wrong, and both
    are pinned by tests:

    * Enumerating ``is_private or is_loopback or is_reserved or ...`` misses
      100.64.0.0/10 (CGNAT, RFC 6598), which reports False for *every* one of
      those properties. An earlier version of this function let it through.
    * ``is_global`` alone misses multicast - 224.0.0.1 reports ``is_global=True``
      - so that is checked separately.

    IPv4-mapped IPv6 (``::ffff:127.0.0.1``), the classic bypass, is covered.
    An unparseable address is refused rather than trusted: if we cannot tell
    what it is, we do not connect to it.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    return not ip.is_global or ip.is_multicast


async def _assert_reachable_host(url: str) -> None:
    if settings.url_fetch_allow_private:
        return

    hostname = urlparse(url).hostname or ""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UpstreamError(
            f"Could not resolve {hostname}.", code="url_dns_failed", status=400
        ) from exc

    if any(_is_private(info[4][0]) for info in infos):
        raise ValidationError(
            "That URL resolves to a private network address, which is not allowed.",
            details={"host": hostname},
        )


async def _read_capped(response: httpx.Response, max_bytes: int) -> tuple[bytes, bool]:
    """Read the body with a hard ceiling, stopping rather than buffering a huge response."""
    chunks: list[bytes] = []
    total = 0

    async for piece in response.aiter_bytes():
        total += len(piece)
        if total > max_bytes:
            chunks.append(piece[: len(piece) - (total - max_bytes)])
            return b"".join(chunks), True
        chunks.append(piece)

    return b"".join(chunks), False


def _decode(payload: bytes, content_type: str) -> str:
    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


async def fetch_url(raw_url: str) -> FetchedPage:
    url = parse_and_validate_url(raw_url)
    headers = {
        "user-agent": settings.url_fetch_user_agent,
        "accept": "text/html,text/plain;q=0.9,*/*;q=0.5",
    }

    # follow_redirects=False: each hop is validated before it is followed.
    async with httpx.AsyncClient(
        timeout=settings.url_fetch_timeout_seconds, follow_redirects=False, headers=headers
    ) as client:
        for _ in range(settings.url_fetch_max_redirects + 1):
            await _assert_reachable_host(url)

            try:
                request = client.build_request("GET", url)
                response = await client.send(request, stream=True)
            except httpx.TimeoutException as exc:
                raise UpstreamError(
                    f"Fetching the page timed out after {settings.url_fetch_timeout_seconds}s.",
                    code="url_timeout",
                    status=504,
                ) from exc
            except httpx.HTTPError as exc:
                raise UpstreamError(
                    f"Could not connect to {urlparse(url).hostname}.",
                    code="url_unreachable",
                    status=502,
                ) from exc

            try:
                if response.is_redirect and response.headers.get("location"):
                    next_url = urljoin(url, response.headers["location"])
                    logger.debug("following redirect", extra={"from": url, "to": next_url})
                    url = parse_and_validate_url(next_url)
                    continue

                if response.status_code >= 400:
                    raise UpstreamError(
                        f"The page returned HTTP {response.status_code}.",
                        code="url_http_error",
                        # The upstream page 404ing is the user's input problem, not ours.
                        status=404 if response.status_code == 404 else 502,
                        details={"status": response.status_code, "url": url},
                    )

                content_type = response.headers.get("content-type", "")
                if not any(kind in content_type for kind in TEXTUAL_CONTENT_TYPES):
                    label = content_type.split(";")[0] or "an unknown type"
                    raise ValidationError(
                        f'That URL serves "{label}", which cannot be read as text.',
                        details={"supported": list(TEXTUAL_CONTENT_TYPES)},
                    )

                payload, truncated = await _read_capped(response, settings.url_fetch_max_bytes)
                if truncated:
                    logger.warning(
                        "page truncated at byte cap",
                        extra={"url": url, "maxBytes": settings.url_fetch_max_bytes},
                    )

                return FetchedPage(
                    final_url=url,
                    content_type=content_type,
                    body=_decode(payload, content_type),
                    truncated=truncated,
                )
            finally:
                await response.aclose()

    raise UpstreamError(
        f"That URL redirected more than {settings.url_fetch_max_redirects} times.",
        code="url_too_many_redirects",
        status=502,
    )
