"""Server-side URL fetching.

Fetching a user-supplied URL is a request-forgery primitive, so the guards here
are the security surface of the app and deserve direct tests rather than
coverage-by-accident through the API.

The HTTP cases run against a real throwaway server on localhost - mocking the
transport would test the mock, not the redirect/streaming behaviour that
actually matters.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.config import settings
from app.errors import UpstreamError, ValidationError
from app.ingest.url_fetcher import _is_private, fetch_url, parse_and_validate_url


# ---------- SSRF guard ----------


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",        # loopback
        "10.0.0.5",         # RFC1918
        "172.16.4.2",       # RFC1918
        "192.168.1.1",      # RFC1918
        "169.254.169.254",  # link-local: the cloud metadata endpoint
        "100.64.0.1",       # CGNAT
        "0.0.0.0",          # unspecified
        "::1",              # IPv6 loopback
        "fd00::1",          # IPv6 unique-local
        "fe80::1",          # IPv6 link-local
        "::ffff:127.0.0.1", # IPv4-mapped loopback - the classic bypass
        "::ffff:10.0.0.1",  # IPv4-mapped RFC1918
        "192.0.2.1",        # TEST-NET documentation range
        "240.0.0.1",        # reserved
        "255.255.255.255",  # broadcast
        "224.0.0.1",        # multicast - reports is_global=True, needs its own check
        "ff02::1",          # IPv6 multicast
    ],
)
def test_private_addresses_are_rejected(address):
    assert _is_private(address) is True


def test_an_unparseable_address_is_refused_rather_than_trusted():
    """Fail closed: if we cannot classify it, we do not connect to it."""
    assert _is_private("not-an-ip") is True


@pytest.mark.parametrize("address", ["8.8.8.8", "93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"])
def test_public_addresses_are_allowed(address):
    assert _is_private(address) is False


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com"])
def test_non_http_schemes_are_rejected(url):
    with pytest.raises(ValidationError) as excinfo:
        parse_and_validate_url(url)
    assert "http" in str(excinfo.value).lower()


def test_url_without_a_host_is_rejected():
    with pytest.raises(ValidationError):
        parse_and_validate_url("http:///nohost")


async def test_private_hostname_is_refused_before_any_request(monkeypatch):
    """The guard must run on the resolved address, not the literal string."""
    monkeypatch.setattr(settings, "url_fetch_allow_private", False)

    with pytest.raises(ValidationError) as excinfo:
        await fetch_url("http://localhost:9/should-never-connect")
    assert "private" in str(excinfo.value).lower()


# ---------- a real server to fetch from ----------


class _Handler(BaseHTTPRequestHandler):
    routes: dict = {}

    def do_GET(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        status, headers, body = self.routes.get(self.path, (404, {"Content-Type": "text/html"}, b"missing"))
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, *args):  # keep the test output clean
        pass


@pytest.fixture
def server(monkeypatch):
    """A localhost HTTP server, with the SSRF guard relaxed so it can be reached."""
    monkeypatch.setattr(settings, "url_fetch_allow_private", True)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    _Handler.routes = {
        "/page": (200, {"Content-Type": "text/html; charset=utf-8"},
                  b"<html><head><title>Hi</title></head><body><p>Hello there</p></body></html>"),
        "/plain": (200, {"Content-Type": "text/plain"}, b"just text"),
        "/pdf": (200, {"Content-Type": "application/pdf"}, b"%PDF-1.4 binary"),
        "/huge": (200, {"Content-Type": "text/plain"}, b"x" * 50_000),
        "/latin": (200, {"Content-Type": "text/plain; charset=iso-8859-1"}, "café".encode("iso-8859-1")),
        "/gone": (404, {"Content-Type": "text/html"}, b"nope"),
        "/boom": (500, {"Content-Type": "text/html"}, b"server error"),
        "/redirect": (302, {"Location": f"{base}/page"}, b""),
        "/loop": (302, {"Location": f"{base}/loop"}, b""),
        "/relative-redirect": (302, {"Location": "/plain"}, b""),
    }

    yield base
    httpd.shutdown()


async def test_fetches_html_and_reports_the_final_url(server):
    page = await fetch_url(f"{server}/page")

    assert "Hello there" in page.body
    assert "text/html" in page.content_type
    assert page.final_url == f"{server}/page"
    assert page.truncated is False


async def test_follows_a_redirect_and_reports_where_it_landed(server):
    page = await fetch_url(f"{server}/redirect")

    assert page.final_url == f"{server}/page"
    assert "Hello there" in page.body


async def test_follows_a_relative_redirect(server):
    page = await fetch_url(f"{server}/relative-redirect")

    assert page.final_url == f"{server}/plain"
    assert page.body == "just text"


async def test_a_redirect_loop_gives_up_with_a_clear_error(server):
    with pytest.raises(UpstreamError) as excinfo:
        await fetch_url(f"{server}/loop")
    assert excinfo.value.code == "url_too_many_redirects"


async def test_oversized_body_is_truncated_not_buffered(server, monkeypatch):
    monkeypatch.setattr(settings, "url_fetch_max_bytes", 1_000)

    page = await fetch_url(f"{server}/huge")

    assert page.truncated is True
    assert len(page.body) == 1_000, "the cap must be enforced while streaming"


async def test_non_textual_content_type_is_refused(server):
    with pytest.raises(ValidationError) as excinfo:
        await fetch_url(f"{server}/pdf")
    assert "pdf" in str(excinfo.value)


async def test_upstream_404_is_reported_as_the_users_problem(server):
    with pytest.raises(UpstreamError) as excinfo:
        await fetch_url(f"{server}/gone")

    assert excinfo.value.code == "url_http_error"
    # The page 404ing is bad input, not a gateway failure.
    assert excinfo.value.status == 404


async def test_upstream_500_is_reported_as_a_gateway_failure(server):
    with pytest.raises(UpstreamError) as excinfo:
        await fetch_url(f"{server}/boom")
    assert excinfo.value.status == 502


async def test_declared_charset_is_honoured(server):
    page = await fetch_url(f"{server}/latin")
    assert page.body == "café", "a latin-1 page must not be mojibake"


async def test_connection_failure_is_mapped_not_raised_raw(monkeypatch):
    monkeypatch.setattr(settings, "url_fetch_allow_private", True)

    with pytest.raises(UpstreamError) as excinfo:
        # Port 9 (discard) with nothing listening.
        await fetch_url("http://127.0.0.1:9/nothing-here")
    assert excinfo.value.code == "url_unreachable"
