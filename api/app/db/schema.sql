-- Two tables: the thing the user saved, and the retrievable pieces of it.
-- Chunks are derived data: deleting an item deletes its chunks, and a
-- re-ingest replaces them wholesale.

CREATE TABLE IF NOT EXISTS items (
  id            TEXT PRIMARY KEY,
  source_type   TEXT NOT NULL CHECK (source_type IN ('note', 'url')),
  title         TEXT,
  url           TEXT,
  raw_content   TEXT,
  -- pending -> processing -> ready | failed. Ingestion is asynchronous, so this
  -- column is the contract between POST /ingest and GET /items.
  status        TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
  error_code    TEXT,
  error_message TEXT,
  char_count    INTEGER NOT NULL DEFAULT 0,
  chunk_count   INTEGER NOT NULL DEFAULT 0,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL,
  processed_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_items_created_at ON items (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_items_status ON items (status);

CREATE TABLE IF NOT EXISTS chunks (
  id                 TEXT PRIMARY KEY,
  item_id            TEXT NOT NULL REFERENCES items (id) ON DELETE CASCADE,
  chunk_index        INTEGER NOT NULL,
  content            TEXT NOT NULL,
  char_start         INTEGER NOT NULL,
  char_end           INTEGER NOT NULL,
  -- float32 vector as a raw little-endian blob: compact, and it maps straight
  -- onto a numpy array with no parsing on load.
  embedding          BLOB NOT NULL,
  embedding_provider TEXT NOT NULL,
  embedding_model    TEXT NOT NULL,
  dimensions         INTEGER NOT NULL,
  created_at         TEXT NOT NULL,
  UNIQUE (item_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_item_id ON chunks (item_id);
