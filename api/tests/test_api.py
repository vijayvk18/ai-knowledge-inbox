"""End-to-end API tests over the real ASGI app.

In-memory SQLite, local embeddings, echo answers: no credentials, no network,
and the whole path from POST /ingest through the async workers to POST /query
is exercised for real rather than mocked.
"""

from app.db import items_repo
from app.rag.vector_store import vector_store


async def test_ingest_note_returns_202_pending_then_indexes(client, drain):
    created = await client.post(
        "/ingest",
        json={"type": "note", "content": "The deploy freeze starts on December 18 and lifts on January 5."},
    )

    assert created.status_code == 202
    body = created.json()
    assert body["item"]["status"] == "pending"
    assert body["item"]["chunkCount"] == 0
    assert created.headers["location"] == f"/api/items/{body['item']['id']}"

    await drain()

    fetched = await client.get(f"/items/{body['item']['id']}")
    assert fetched.status_code == 200
    item = fetched.json()["item"]
    assert item["status"] == "ready"
    assert item["chunkCount"] == 1
    assert item["title"].startswith("The deploy freeze")
    assert vector_store.size() > 0


async def test_empty_note_is_rejected_with_field_details(client):
    response = await client.post("/ingest", json={"type": "note", "content": "   "})

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["issues"][0]["field"].endswith("content")
    assert error["requestId"]


async def test_unknown_source_type_is_rejected(client):
    response = await client.post("/ingest", json={"type": "pdf", "url": "x"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"


async def test_non_http_url_is_rejected_before_an_item_is_created(client):
    before = len((await client.get("/items")).json()["items"])

    response = await client.post("/ingest", json={"type": "url", "url": "file:///etc/passwd"})
    assert response.status_code == 400

    after = len((await client.get("/items")).json()["items"])
    assert after == before, "no item row should have been created"


async def test_malformed_json_has_its_own_error_code(client):
    response = await client.post(
        "/ingest", content=b"{oops", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "malformed_json"


async def test_items_list_carries_preview_not_full_content(client, drain):
    await client.post("/ingest", json={"type": "note", "content": "A saved thought about latency budgets."})
    await drain()

    response = await client.get("/items?limit=10")
    assert response.status_code == 200
    body = response.json()

    assert body["counts"]["ready"] >= 1
    assert "preview" in body["items"][0]
    assert "content" not in body["items"][0], "list rows must not carry full content"


async def test_malformed_cursor_is_rejected(client):
    response = await client.get("/items?cursor=not-a-cursor")
    assert response.status_code == 400


async def test_unknown_item_404s_and_malformed_id_400s(client):
    assert (await client.get("/items/itm_missing")).status_code == 404
    assert (await client.get("/items/garbage")).status_code == 400


async def test_query_answers_from_saved_content_with_citations(client, drain):
    await client.post(
        "/ingest",
        json={
            "type": "note",
            "title": "Oncall",
            "content": "The oncall rotation handoff happens every Tuesday at 10am in the platform channel.",
        },
    )
    await drain()

    response = await client.post("/query", json={"question": "When is the oncall rotation handoff?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert len(body["sources"]) >= 1

    top = body["sources"][0]
    assert top["citation"] == 1
    assert top["score"] > 0
    assert "oncall rotation handoff" in top["snippet"]
    assert body["stats"]["embeddingProvider"] == "local"


async def test_query_with_no_match_at_all_is_200_not_an_error(client, drain):
    """Not one shared word: answering would mean answering from nothing."""
    await client.post("/ingest", json={"type": "note", "content": "Notes about the office coffee machine."})
    await drain()

    response = await client.post("/query", json={"question": "xylophone marmalade zeppelin quixotic"})

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is False
    assert body["reason"] == "no_relevant_context"
    assert body["sources"] == []


async def test_query_on_an_empty_inbox_explains_itself(client):
    response = await client.post("/query", json={"question": "anything at all"})

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is False
    assert body["reason"] == "empty_index"


async def test_short_question_is_rejected(client):
    response = await client.post("/query", json={"question": "a"})
    assert response.status_code == 400
    assert response.json()["error"]["details"]["issues"][0]["field"].endswith("question")


async def test_top_k_is_clamped_to_the_configured_maximum(client):
    response = await client.post("/query", json={"question": "anything at all", "topK": 9999})
    assert response.status_code == 400


async def test_delete_removes_item_chunks_and_vectors(client, drain):
    created = await client.post(
        "/ingest", json={"type": "note", "content": "Ephemeral note about the quarterly offsite in Lisbon."}
    )
    await drain()
    item_id = created.json()["item"]["id"]

    chunks_before = vector_store.size()
    deleted = await client.delete(f"/items/{item_id}")

    assert deleted.status_code == 204
    assert (await client.get(f"/items/{item_id}")).status_code == 404
    assert vector_store.size() < chunks_before


async def test_retry_re_indexes_a_finished_item(client, drain):
    created = await client.post("/ingest", json={"type": "note", "content": "A note worth re-indexing."})
    await drain()
    item_id = created.json()["item"]["id"]

    response = await client.post(f"/items/{item_id}/retry")

    assert response.status_code == 202
    assert response.json()["item"]["status"] == "pending"


async def test_retry_on_an_in_flight_item_conflicts(client, drain):
    created = await client.post("/ingest", json={"type": "note", "content": "Currently being processed."})
    await drain()
    item_id = created.json()["item"]["id"]

    # Forced rather than raced: the worker pool drains the queue faster than a
    # test can observe 'pending', so put the row in the state under test directly.
    items_repo.mark_processing(item_id)

    response = await client.post(f"/items/{item_id}/retry")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


async def test_unknown_route_uses_the_standard_error_envelope(client):
    response = await client.get("/does-not-exist")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "route_not_found"
    assert error["requestId"]


async def test_health_reports_live_providers_and_index(client):
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["providers"]["embeddings"]["provider"] == "local"
    assert body["providers"]["answers"]["provider"] == "echo"
    assert body["ingestion"]["concurrency"] >= 1


async def test_search_returns_ranked_passages_without_an_llm(client, drain):
    await client.post(
        "/ingest",
        json={"type": "note", "title": "Hiring", "content": "The hiring loop has four stages ending in a debrief."},
    )
    await drain()

    response = await client.get("/search", params={"q": "hiring loop stages", "topK": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["results"]
    assert body["results"][0]["score"] > 0
    assert "answerProvider" not in body["stats"], "search must not invoke the answer model"


async def test_weak_match_answers_at_low_confidence_instead_of_refusing(client, drain):
    """A small inbox should hand back its best guess, flagged, not a flat refusal.

    "overnight" is the only word in common, which scores below the relevance
    floor but above zero - the band where refusing is unhelpful.
    """
    await client.post(
        "/ingest",
        json={
            "type": "note",
            "content": (
            "Sourdough starter needs feeding every twelve hours with equal parts flour and "
            "water. Keep it at room temperature until it doubles, then move it to the fridge. "
            "Discard half before each feed. Baking day: mix the levain, bulk ferment four "
            "hours, shape, and cold proof overnight in the refrigerator."
            ),
        },
    )
    await drain()

    response = await client.post("/query", json={"question": "overnight shipping rates"})

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert body["lowConfidence"] is True
    assert body["sources"], "best-effort matches should still be cited"


async def test_confident_match_is_not_flagged_low_confidence(client, drain):
    await client.post(
        "/ingest",
        json={"type": "note", "content": "The oncall rotation handoff happens every Tuesday at 10am."},
    )
    await drain()

    response = await client.post("/query", json={"question": "When is the oncall rotation handoff?"})

    body = response.json()
    assert body["answered"] is True
    assert body["lowConfidence"] is False


async def test_search_can_show_scores_below_the_floor(client, drain):
    await client.post(
        "/ingest",
        json={
            "type": "note",
            "content": (
            "Sourdough starter needs feeding every twelve hours with equal parts flour and "
            "water. Keep it at room temperature until it doubles, then move it to the fridge. "
            "Discard half before each feed. Baking day: mix the levain, bulk ferment four "
            "hours, shape, and cold proof overnight in the refrigerator."
            ),
        },
    )
    await drain()

    floored = await client.get("/search", params={"q": "overnight shipping rates"})
    unfloored = await client.get("/search", params={"q": "overnight shipping rates", "minScore": 0})

    assert floored.json()["results"] == []
    assert unfloored.json()["results"], "minScore=0 must expose what sat below the floor"
