"""Request correlation, wire shapes, and ingestion failure handling.

The through-line: when something goes wrong, can it be traced and can the user
act on it? An item that fails silently, or an error with no request id, is the
worst outcome in an inbox.
"""

import pytest

from app.db import items_repo
from app.db.items_repo import Cursor
from app.ingest.service import get_ingestion_service
from app.presenters import decode_cursor, encode_cursor, to_item_detail, to_item_summary

# ---------- request correlation ----------


async def test_every_response_carries_a_request_id(client):
    response = await client.get("/health")

    assert response.headers["x-request-id"]


async def test_an_inbound_request_id_is_reused_not_replaced(client):
    """A proxy or client-side trace id must survive, or correlation breaks at the edge."""
    response = await client.get("/health", headers={"x-request-id": "trace-from-the-proxy"})

    assert response.headers["x-request-id"] == "trace-from-the-proxy"


async def test_the_request_id_in_an_error_body_matches_the_header(client):
    """The id a user can quote is the id in the logs."""
    response = await client.get("/items/garbage")

    assert response.json()["error"]["requestId"] == response.headers["x-request-id"]


async def test_each_request_gets_a_distinct_id(client):
    first = await client.get("/health")
    second = await client.get("/health")

    assert first.headers["x-request-id"] != second.headers["x-request-id"]


async def test_an_unsupported_method_is_405_in_the_standard_envelope(client):
    response = await client.put("/health")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


# ---------- cursors ----------


def test_a_cursor_round_trips():
    original = Cursor(created_at="2026-03-01T10:00:00+00:00", id="itm_abc")

    decoded = decode_cursor(encode_cursor(original))

    assert decoded.created_at == original.created_at
    assert decoded.id == original.id


def test_an_absent_cursor_is_not_an_error():
    assert decode_cursor(None) is None
    assert decode_cursor("") is None
    assert encode_cursor(None) is None


@pytest.mark.parametrize(
    "tampered",
    [
        "not-base64-at-all!!",
        "eyJqdW5rIjoxfQ",              # valid base64, wrong shape
        "eyJjcmVhdGVkQXQiOjEyM30",     # createdAt present but not a string
    ],
)
def test_a_tampered_cursor_is_rejected_rather_than_half_understood(tampered):
    assert decode_cursor(tampered) is False


def test_cursors_are_opaque():
    """Clients must not be able to read or forge paging state from the token."""
    encoded = encode_cursor(Cursor(created_at="2026-03-01T10:00:00+00:00", id="itm_abc"))

    assert "itm_abc" not in encoded
    assert "2026" not in encoded


# ---------- wire shapes ----------


def test_a_summary_previews_long_content_and_omits_the_body():
    item = items_repo.create(source_type="note", title="Long", raw_content="x" * 1000)

    summary = to_item_summary(item)

    assert len(summary["preview"]) == 243, "240 characters plus an ellipsis"
    assert summary["preview"].endswith("...")
    assert "content" not in summary


def test_short_content_is_previewed_without_an_ellipsis():
    item = items_repo.create(source_type="note", raw_content="short note")

    assert to_item_summary(item)["preview"] == "short note"


def test_detail_carries_the_full_body():
    item = items_repo.create(source_type="note", raw_content="y" * 1000)

    assert to_item_detail(item)["content"] == "y" * 1000


def test_an_item_with_no_content_yet_has_no_preview():
    item = items_repo.create(source_type="url", url="https://example.com/pending")

    assert to_item_summary(item)["preview"] is None


# ---------- ingestion failures ----------


async def test_a_failed_url_is_marked_failed_with_an_actionable_message(client, drain):
    """A dead link must leave a visible, retryable row - never vanish."""
    created = await client.post(
        "/ingest", json={"type": "url", "url": "https://this-host-does-not-resolve.invalid/page"}
    )
    await drain()

    item = (await client.get(f"/items/{created.json()['item']['id']}")).json()["item"]

    assert item["status"] == "failed"
    assert item["error"]["code"]
    assert item["error"]["message"], "a failure the user cannot read is not actionable"
    assert item["chunkCount"] == 0


async def test_a_failed_item_can_be_retried(client, drain):
    created = await client.post(
        "/ingest", json={"type": "url", "url": "https://this-host-does-not-resolve.invalid/page"}
    )
    await drain()
    item_id = created.json()["item"]["id"]

    retried = await client.post(f"/items/{item_id}/retry")

    assert retried.status_code == 202
    await drain()
    # Still failing, because the host is still gone - but the retry ran.
    assert (await client.get(f"/items/{item_id}")).json()["item"]["status"] == "failed"


async def test_queueing_the_same_item_twice_while_in_flight_is_refused(client, drain):
    """The guard is against concurrent duplicates, so the two calls must not yield.

    Both enqueues are synchronous and adjacent: no await between them, so no
    worker can drain the queue in the gap. (Re-queueing an item that has already
    *finished* is allowed - that is exactly what /retry does.)
    """
    item = items_repo.create(source_type="note", raw_content="Idempotency check.")
    service = get_ingestion_service()

    assert service.enqueue(item.id) is True
    assert service.enqueue(item.id) is False, "a duplicate must not be queued twice"

    await drain()
    assert items_repo.find_by_id(item.id).chunk_count == 1


async def test_unfinished_items_are_recovered_after_a_restart(client, drain):
    """The in-memory queue dies with the process; the status column is the recovery point."""
    item = items_repo.create(source_type="note", raw_content="Stranded mid-flight.", status="processing")

    recovered = get_ingestion_service().recover_unfinished()
    await drain()

    assert recovered >= 1
    assert items_repo.find_by_id(item.id).status == "ready"


async def test_health_reports_queue_depth(client):
    body = (await client.get("/health")).json()

    assert body["ingestion"]["concurrency"] >= 1
    assert "queued" in body["ingestion"]
    assert "inFlight" in body["ingestion"]
