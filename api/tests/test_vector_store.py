from dataclasses import dataclass

import numpy as np
import pytest

from app.rag.vector_store import VectorStore


@dataclass
class FakeChunk:
    id: str
    chunk_index: int
    content: str
    char_start: int
    char_end: int
    embedding: np.ndarray


def chunk(chunk_id: str, vector: list[float], index: int = 0) -> FakeChunk:
    return FakeChunk(
        id=chunk_id,
        chunk_index=index,
        content=f"content of {chunk_id}",
        char_start=0,
        char_end=10,
        embedding=np.asarray(vector, dtype=np.float32),
    )


@pytest.fixture
def store() -> VectorStore:
    return VectorStore()


def test_ranks_by_cosine_similarity_highest_first(store):
    store.set_item_chunks("itm_a", [chunk("chk_a", [1, 0, 0])])
    store.set_item_chunks("itm_b", [chunk("chk_b", [0.8, 0.6, 0])])
    store.set_item_chunks("itm_c", [chunk("chk_c", [0, 0, 1])])

    hits, _ = store.search(np.asarray([1, 0, 0], dtype=np.float32), top_k=3)

    assert [hit.chunk_id for hit in hits] == ["chk_a", "chk_b", "chk_c"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)


def test_normalizes_on_insert_so_magnitude_does_not_affect_ranking(store):
    store.set_item_chunks("itm_a", [chunk("chk_small", [0.01, 0, 0])])
    store.set_item_chunks("itm_b", [chunk("chk_big", [50, 40, 0])])

    hits, _ = store.search(np.asarray([1, 0, 0], dtype=np.float32), top_k=2)
    assert hits[0].chunk_id == "chk_small"


def test_drops_results_below_the_relevance_floor(store):
    store.set_item_chunks("itm_a", [chunk("chk_a", [1, 0, 0])])
    store.set_item_chunks("itm_b", [chunk("chk_b", [0, 1, 0])])

    hits, _ = store.search(np.asarray([1, 0, 0], dtype=np.float32), top_k=5, min_score=0.5)
    assert [hit.chunk_id for hit in hits] == ["chk_a"]


def test_caps_how_many_chunks_one_item_contributes(store):
    store.set_item_chunks(
        "itm_long",
        [
            chunk("chk_1", [1, 0, 0], 0),
            chunk("chk_2", [0.99, 0.1, 0], 1),
            chunk("chk_3", [0.98, 0.15, 0], 2),
        ],
    )
    store.set_item_chunks("itm_other", [chunk("chk_other", [0.7, 0.7, 0])])

    hits, _ = store.search(np.asarray([1, 0, 0], dtype=np.float32), top_k=4, max_per_item=2)

    assert len([hit for hit in hits if hit.item_id == "itm_long"]) == 2
    assert any(hit.item_id == "itm_other" for hit in hits)


def test_ignores_vectors_whose_dimensions_do_not_match(store):
    # Happens when the embedding provider changes without a re-index.
    store.set_item_chunks("itm_old", [chunk("chk_old", [1, 0])])
    store.set_item_chunks("itm_new", [chunk("chk_new", [1, 0, 0])])

    hits, _ = store.search(np.asarray([1, 0, 0], dtype=np.float32), top_k=5)
    assert [hit.chunk_id for hit in hits] == ["chk_new"]


def test_replaces_item_chunks_rather_than_appending(store):
    store.set_item_chunks("itm_a", [chunk("chk_v1", [1, 0, 0])])
    store.set_item_chunks("itm_a", [chunk("chk_v2", [1, 0, 0])])

    assert store.size() == 1
    hits, _ = store.search(np.asarray([1, 0, 0], dtype=np.float32), top_k=5)
    assert hits[0].chunk_id == "chk_v2"


def test_removes_every_vector_for_a_deleted_item(store):
    store.set_item_chunks("itm_a", [chunk("chk_a", [1, 0, 0])])
    store.remove_item("itm_a")

    assert store.size() == 0
    assert store.item_count() == 0
    hits, _ = store.search(np.asarray([1, 0, 0], dtype=np.float32), top_k=5)
    assert hits == []


def test_returns_nothing_when_the_index_is_empty(store):
    hits, scanned = store.search(np.asarray([1, 0, 0], dtype=np.float32), top_k=5)
    assert hits == []
    assert scanned == 0
