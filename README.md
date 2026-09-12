# AI Knowledge Inbox

Save notes and URLs, then ask questions answered **only** from what you saved — with citations back to the exact chunk each claim came from.

```
React (Vite)  ──HTTP──▶  FastAPI  ──▶  SQLite  (items + chunks + vectors)
                            │
                            ├──▶ asyncio worker pool   fetch → extract → chunk → embed
                            ├──▶ vector index (numpy)  exact cosine, in memory
                            └──▶ Claude                grounded, cited answers
```

FastAPI · Pydantic v2 · SQLite · numpy · React hooks. ~3,700 lines, no framework beyond those.

---

## Quick start

```bash
npm run setup     # creates api/.venv, installs Python + web deps
npm run dev       # API on :8787 (OpenAPI docs at /docs), web on :5173
```

Open <http://localhost:5173>. **Runs with no API keys** — both AI dependencies have offline fallbacks ([below](#running-without-keys)).

For real semantic search and generated answers:

```bash
cp api/.env.example api/.env     # set OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

`api/.env.example` documents every setting with its default and why it exists; a test keeps it in step with the code. Real environment variables override the file, so a deployment cannot be hijacked by a stray `.env`.

```bash
npm test          # 167 tests, 95% line coverage
npm run build     # production build of the frontend
```

Driving Python directly: `cd api && .venv/Scripts/python -m uvicorn app.main:app --reload` (`.venv/bin/python` on macOS/Linux), `python -m pytest`.

---

## API

Base path `/api`. Interactive docs at `/docs`. Every response is a JSON object; every error, without exception:

```json
{ "error": { "code": "validation_error", "message": "…", "requestId": "…", "details": {} } }
```

`code` is stable and machine-readable, `message` is written for a human, and `requestId` — also the `x-request-id` header, reused from an inbound one if a proxy set it — tags every log line for that request.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | Save a note or URL → **202** + `pending` item |
| `GET` | `/items` | Newest-first page + status counts |
| `GET` | `/items/{id}` | One item with its full stored text |
| `POST` | `/items/{id}/retry` | Re-run a failed ingestion → **202** |
| `DELETE` | `/items/{id}` | Remove item, chunks, and vectors → **204** |
| `POST` | `/query` | Ask a question → answer + cited sources |
| `GET` | `/search?q=` | Retrieval only, no LLM — add `minScore=0` to see below the floor |
| `GET` | `/health` | Live providers, index size, queue depth |

**`POST /ingest`** takes a Pydantic discriminated union on `type`, so a note carrying a stray `url` field is rejected rather than silently ignored:

```jsonc
{ "type": "note", "content": "Deploy freeze starts Dec 18.", "title": "Freeze" }  // title optional
{ "type": "url",  "url": "https://example.com/post" }
```

It returns **202, not 201**: the row exists, but nothing has been fetched, chunked, or embedded yet. The client polls `GET /items` for `pending → processing → ready | failed`. A failed item keeps its `{ code, message }` and can be retried.

**`POST /query`** takes `{ "question": "…", "topK": 5 }` and returns:

```jsonc
{
  "answered": true,
  "answer": "Revert the feature flag aurora_enabled and page the on-call SRE [1].",
  "sources": [{
    "citation": 1, "itemId": "itm_…", "chunkId": "chk_…",
    "sourceType": "note", "title": "Aurora launch", "url": null,
    "snippet": "…", "score": 0.62, "charRange": [0, 176],
    "cited": true            // false = retrieved into context, unused by the model
  }],
  "lowConfidence": false,    // true = nothing cleared the relevance floor; best effort
  "stats": { "retrievedChunks": 3, "embeddingMs": 84, "searchMs": 1, "generationMs": 1840, … }
}
```

**An unanswerable question is a 200, not a 404.** An empty inbox and a question nothing matches are ordinary outcomes here, so both return `answered: false` with a `reason` (`empty_index` / `no_relevant_context`) and no sources — rendered as a notice, not an error. Between "confident" and "nothing" sits a third state, `lowConfidence`; see [graded outcomes](#retrieval-outcomes-are-graded-not-binary).

**Status codes:** `200` · `202` async accepted · `204` deleted · `400` validation · `404` unknown item/route · `409` retry while processing · `422` model declined · `429` upstream rate limit · `502` upstream failure · `504` upstream timeout.

---

## Design decisions

### Ingestion is async

Fetching a page, chunking, and embedding takes seconds and fails for reasons the caller cannot fix mid-request. `POST /ingest` validates, writes a `pending` row, enqueues, and returns; a pool of `asyncio` workers started in the app's `lifespan` does the rest.

The `status` column **is** the async contract. The queue is in memory — no broker, because a single-user inbox doesn't need one — so the tradeoff is explicit: **work queued in memory dies with the process**, and rows left `pending`/`processing` are re-queued from SQLite at boot.

### Chunking: structure first, with overlap

Split on blank lines → pack whole paragraphs to ~1000 chars → split oversized paragraphs on sentences, then hard-split at word boundaries → carry 150 chars of overlap → merge a trailing runt.

- **~1000 chars (~250 tokens)** — small enough that a retrieved chunk is mostly signal, large enough to hold one complete thought. Five chunks ≈ 1.2k tokens of context.
- **150-char overlap** — a fact straddling a boundary stays retrievable from both sides, at ~15% more vectors.
- **Not fixed-size windows:** they cut mid-sentence, which embeds fragments and produces citation snippets that read as broken — and the snippet is what the user judges the answer by.
- **Not semantic splitting:** an embedding pass per candidate boundary, for a gain that doesn't show up on notes and articles this short. It would pay off on papers or contracts.
- Offsets are kept against the *stored normalized text*, so a citation addresses an exact span of what the user sees.

### Vector store: SQLite for truth, numpy for search

Vectors are float32 blobs in SQLite (6 KB per 1536-dim vector vs ~30 KB as JSON, and `np.frombuffer` loads them without a parse). At boot they hydrate into one `(n, d)` matrix; a query is a single `matrix @ query` BLAS call plus `np.argpartition`.

**Why brute force:** 10k chunks × 1536 dims is ~60 MB and one matrix-vector product — single-digit milliseconds, *exact* recall, no index build, no tuning, no dependency. HNSW/FAISS would trade exactness and simplicity for a speedup invisible at this size.

Two things matter more than the index structure:
- **A relevance floor** — without it every question retrieves *something* and the model is pushed to answer from noise. Each provider sets its own, since dense and lexical scores aren't on the same scale.
- **A per-item cap** (default 3) — otherwise one long article fills every slot and crowds out other sources.

**Why not pgvector/Chroma/Qdrant:** one file, no daemon. SQLite reads are microseconds, so the sync driver costs the event loop nothing measurable — the slow work is network I/O, which *is* async. A vector DB earns its operational cost at a scale this app is explicitly not at.

### Grounding

The system prompt permits only the numbered context blocks, requires a `[n]` citation per claim, and requires the model to say what's missing rather than fill gaps. The API reports which citations were actually used and the UI dims retrieved-but-uncited passages, so over-retrieval and ungrounded answers are both visible. A policy decline (`stop_reason: "refusal"`) is checked before reading content and surfaces as 422.

### Provider abstraction

Embeddings and answers sit behind small protocols in `api/app/providers/`. Nothing outside those files knows which vendor is live — that's what makes the keyless mode possible and swapping a model a config change.

**Changing the embedding invalidates every stored vector, and the app handles that itself.** Each chunk records the `(provider, model)` that produced it; the local provider's model string carries a version (`hashed-bow-v2-2048`) so a tokenizer change is visible. On boot the index loads only vectors matching the live embedding, and anything else is re-embedded **from the text already in SQLite** — no refetch, so a dead link still re-indexes.

This exists because the naive version bit hard: swapping the local tokenizer kept the vector width at 2048, so the dimension check passed while every stored vector had become meaningless. Saved items scored exactly 0.0 against every query and the app said "nothing matches that question at all" forever. Comparing widths is not comparing spaces.

---

## Running without keys

| | With keys | Without |
|---|---|---|
| Embeddings | OpenAI `text-embedding-3-small` | **local hashed bag-of-words** — lightly stemmed tokens, sub-linear weighting, L2-normalized, 2048 dims |
| Answers | Claude `claude-opus-5` | **echo** — returns the top retrieved passage with a notice |

The local embedder is honest lexical matching, **not semantics**. Suffix stripping means *"budget caps"* finds *"budget cap"*, but nothing bridges vocabulary: *"when is payment due"* will **not** find *"invoice deadline"*, and *"how do I roll back"* will **not** find *"Rollback procedure"* — one word versus two is enough to miss.

**If questions phrased in your own words return little, that is this limitation, not a bug — set `OPENAI_API_KEY` and re-ingest.** `/health` and the UI header always name the live providers, because "why is search bad" is usually answered by that line.

### Retrieval outcomes are graded, not binary

Refusing to answer is a bad response to someone with a five-item inbox, so `/query` distinguishes four states:

| State | Response |
|---|---|
| A chunk cleared the relevance floor | Normal cited answer |
| Nothing cleared it, but something shares wording | Best few passages answered over, `lowConfidence: true`, UI shows a "weak match" banner |
| Not one chunk shares a word | `answered: false`, `reason: no_relevant_context` |
| Nothing saved yet | `answered: false`, `reason: empty_index` |

When a query surprises you, run the doctor:

```bash
npm run doctor -- "the question that returned nothing"
```

It prints the live providers, every saved item with its chunk count, and the score of each chunk against your question — including the ones filtered out by the floor — then names the cause:

```
  relevance floor: 0.1

  ██████████·············· 0.4264  RETRIEVED   Platform sync
  ························ 0.0000  below floor Sourdough

  -> retrieval is working; this question should be answerable.
```

A row of zeros means no shared vocabulary, which is the keyless embedder's hard limit rather than a bug. `GET /search?q=…&minScore=0` is the same data if you prefer curl.

---

## What breaks at scale

| Limit | What breaks | Change |
|---|---|---|
| **~10⁵ chunks** | Linear scan crosses ~100 ms/query; the matrix stops fitting comfortably in one process | pgvector or a dedicated store with HNSW; accept approximate recall |
| **>1 API replica** | Each replica runs its own queue and index copy → double-processed items, stale results | Extract the queue (Redis/arq + worker); make the index a shared service |
| **Concurrent writers** | SQLite serializes writers; WAL helps readers, not write-heavy load | Postgres |
| **Embedding cost / rate limits** | Batch ingest hits 429s; edits re-embed unchanged text | Content-hash chunks and skip unchanged; backoff + dead-letter status; batch API |
| **Large corpus** | Dense top-k returns plausible-but-wrong chunks; exact terms (IDs, names) get missed | Hybrid BM25 + dense with rank fusion, then a cross-encoder re-ranker |
| **Long documents** | A 200-page PDF dominates every query | Per-item cap (already here) + hierarchical summaries / parent-document retrieval |
| **Answer latency** | Users watch a spinner during generation | Stream tokens (SSE) — retrieval and generation are already separate, so sources can render first |
| **CPU-bound embedding** | Local embedding of a huge document blocks the event loop | `run_in_executor` onto a process pool; the OpenAI path is I/O-bound and unaffected |

## Before production

1. **Auth and per-user scoping** — every query is global today; `items` would take a `user_id` and retrieval would filter on it. Out of scope per the brief.
2. **Retries with backoff** on transient upstream failures, plus a dead-letter state distinct from "that page 404'd".
3. **Idempotency keys on `/ingest`** so a double-tapped button doesn't create two items.
4. **Rate limiting** on `/ingest` and `/query` — both spend money on someone else's API.
5. **A real retrieval eval set.** `tests/test_local_embeddings.py` is the seed — it already caught a live ranking bug — but it needs ~50 question→expected-chunk pairs scored with recall@k in CI, run against the OpenAI provider too. Without it, every chunking or model change is a vibe check. Highest-value item on this list.
6. **Tracing** (OpenTelemetry) across fetch → chunk → embed → retrieve → generate, with token and cost attributes. The logs already carry the timings; spans would connect them.
7. **Content-hash dedupe** so re-saving a URL updates in place.

---

## Layout

```
api/app/
  config.py         every tunable (pydantic-settings); nothing else reads the environment
  main.py           app factory + lifespan: providers → index → workers
  middleware.py     request id (ContextVar) + access logging
  errors.py         error types and the handlers that shape every failure response
  schemas.py        Pydantic models — these generate the OpenAPI schema
  presenters.py     row → wire shape (previews, opaque cursors)
  routers/          one file per resource; no business logic
  db/               schema.sql, connection, items_repo, chunks_repo — all SQL lives here
  ingest/           service (submit + worker pool), url_fetcher, html_extract
  rag/              chunker, retriever, answer_service, vector_store, embeddings
  providers/        openai_embeddings, local_embeddings, anthropic_answers, echo_answers
api/tests/          chunker, vector store, retrieval quality, full API path
web/src/
  App.jsx           two columns, one state owner per concern
  hooks/            useItems (list + polling), useAsk (one question at a time)
  components/       AddItemForm, ItemList, AskPanel, AnswerPanel
  lib/api.js        the only module that talks to the backend
```

Dependency direction is one-way: **routers → services → repos/providers**. Routers never touch SQL, repositories never format HTTP, providers never know FastAPI exists. Largest file is 276 lines.

## Security

The server fetching a user-supplied URL is a request-forgery primitive, so `url_fetcher.py` enforces: http/https only · DNS for **every** redirect hop checked against private/loopback/link-local ranges · manual redirect following, so hop 2 can't escape hop 1's check · wall-clock timeout · hard byte cap enforced while streaming · textual content types only. `URL_FETCH_ALLOW_PRIVATE=true` disables the guard for local testing.

Unexpected errors never reach the client: logged with a traceback and request id, returned as a generic 500.

## Testing notes

167 tests, 95% line coverage. What each file is for:

| File | Covers |
|---|---|
| `test_api.py` | Every endpoint end to end through the real ASGI app |
| `test_url_fetcher.py` | The SSRF guard, redirects, byte caps, content types — against a real localhost server |
| `test_html_extract.py` | Extraction, chrome removal, and the paragraph structure the chunker depends on |
| `test_chunker.py` | Boundaries, overlap, offsets, degenerate inputs |
| `test_vector_store.py` | Ranking, floors, per-item caps, dimension mismatches |
| `test_local_embeddings.py` | Retrieval *quality* — the closest thing here to an eval set |
| `test_answer_providers.py` | The Claude request we build, refusals, and error mapping |
| `test_openai_embeddings.py` | Batching, response ordering, and error mapping |
| `test_reindex.py` | Staleness detection and self-healing re-index |
| `test_plumbing.py` | Request-id correlation, cursors, ingestion failure handling |
| `test_env_example.py` | `.env.example` documents every real setting, and only real ones |

The provider tests matter disproportionately: no live call to OpenAI or Anthropic has ever run from this project, so they pin our half of those contracts — the request shape, how the response is read, and how every upstream failure maps onto an HTTP status.


`tests/conftest.py` sets `DB_FILE=:memory:` and forces the local providers **before any app module is imported**, and pops `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` from the environment — without that, a developer with keys exported would have the suite make billed calls and write to the real database. Tests await `queue.join()` instead of sleeping, so the async path is deterministic.

`tests/test_local_embeddings.py` guards retrieval *quality*, not just plumbing. Both cases in it were real bugs, and both only ever affected the keyless fallback — the OpenAI path embeds properly:

- At 512 dimensions, hash collisions let an unrelated page outrank the correct note; the right note scored **negative** against a query containing its own keyword. Fixed by widening to 2048.
- Character 4-grams matched letter sequences rather than words: *"quarterly"* and *"starter"* share `arte` and `rter`, so texts with **no word in common** scored 0.076 and cleared the relevance floor. Replaced with light suffix stemming, which scores **exactly zero** on unrelated text and matches plurals better. The cost is typo tolerance, which n-grams gave and stemming does not — a deliberate trade, since a confident wrong citation is worse than a missed typo.

A compound-word feature (hashing adjacent token pairs, to bridge *"roll back"* ↔ *"rollback"*) was prototyped and **rejected**: it half-fixed one case while reintroducing noise on unrelated text and diluting real matches.

`tests/test_reindex.py` covers the fallout of that second fix: changing the tokenizer left existing vectors the same *width* but a different *space*, so they silently scored zero forever. Stale vectors are now detected by `(provider, model)`, kept out of the index, and re-embedded from stored text on boot.

**Not verified:** no live OpenAI or Anthropic call has been made — no key was available during development. Both providers construct correctly and their error mapping is written, but those two network paths are unproven.
