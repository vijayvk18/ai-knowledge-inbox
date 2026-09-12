"""Staleness detection and self-healing.

Regression test for a bug that silently broke a live inbox: the local embedder's
tokenization changed while its vector width stayed the same, so every previously
saved item scored exactly 0.0 against every query. Dimensions matched, so no
guard fired, and the user saw "nothing matches that question at all" forever.
"""

from app.db import chunks_repo, items_repo
from app.ingest.service import get_ingestion_service
from app.rag.embeddings import get_embedding_provider
from app.rag.vector_store import vector_store

NOTE = "Deploy freeze runs December 18 through January 5. Priya owns the launch checklist."


async def _save_note(client, drain, content=NOTE):
    created = await client.post("/ingest", json={"type": "note", "content": content})
    await drain()
    return created.json()["item"]["id"]


def _rewrite_embedding_model(item_id: str, model: str) -> None:
    """Simulate vectors written by an older embedding version."""
    connection = chunks_repo.get_connection()
    connection.execute("UPDATE chunks SET embedding_model = ? WHERE item_id = ?", (model, item_id))
    connection.commit()


async def test_stale_vectors_are_detected(client, drain):
    item_id = await _save_note(client, drain)
    provider = get_embedding_provider()

    assert chunks_repo.find_stale_item_ids(provider=provider.name, model=provider.model) == []

    _rewrite_embedding_model(item_id, "hashed-bow-v1-2048")

    assert chunks_repo.find_stale_item_ids(provider=provider.name, model=provider.model) == [item_id]


async def test_stale_vectors_are_not_loaded_into_the_index(client, drain):
    item_id = await _save_note(client, drain)
    provider = get_embedding_provider()
    _rewrite_embedding_model(item_id, "hashed-bow-v1-2048")

    vector_store.hydrate(provider=provider.name, model=provider.model)

    assert vector_store.size() == 0, "incomparable vectors must not be scored against"


async def test_reindex_rebuilds_stale_items_and_restores_search(client, drain):
    item_id = await _save_note(client, drain)
    provider = get_embedding_provider()

    # The exact failure: vectors from an older embedding, index rebuilt from them.
    _rewrite_embedding_model(item_id, "hashed-bow-v1-2048")
    vector_store.hydrate(provider=provider.name, model=provider.model)

    broken = await client.post("/query", json={"question": "when is the deploy freeze"})
    assert broken.json()["answered"] is False, "precondition: search is broken"

    healed = get_ingestion_service().reindex_stale()
    await drain()

    assert healed == 1
    assert chunks_repo.find_stale_item_ids(provider=provider.name, model=provider.model) == []

    recovered = await client.post("/query", json={"question": "when is the deploy freeze"})
    body = recovered.json()
    assert body["answered"] is True
    assert body["sources"][0]["itemId"] == item_id


async def test_reindex_does_not_refetch_a_url(client, drain):
    """Re-embedding uses stored text: no network, so a dead link still re-indexes."""
    item = items_repo.create(
        source_type="url",
        title="Saved page",
        url="https://example.invalid/gone",
        raw_content="Cached article text about deploy freezes and launch checklists.",
        status="ready",
    )
    provider = get_embedding_provider()
    vectors = await provider.embed_documents(["Cached article text about deploy freezes."])
    chunks_repo.replace_for_item(
        item.id,
        [{"content": "Cached article text about deploy freezes.", "char_start": 0, "char_end": 41,
          "embedding": vectors[0]}],
        provider=provider.name,
        model="hashed-bow-v1-2048",
    )

    get_ingestion_service().reindex_stale()
    await drain()

    # example.invalid is unresolvable; reaching the network would have failed the item.
    assert items_repo.find_by_id(item.id).status == "ready"
    assert chunks_repo.find_stale_item_ids(provider=provider.name, model=provider.model) == []
