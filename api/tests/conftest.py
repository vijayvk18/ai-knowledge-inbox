"""Test configuration.

The environment is set before any app module is imported, because config.py
reads it at import time. Without this the suite would write to the real
database file and, with credentials present in the environment, make billed API
calls instead of using the local providers.
"""

from __future__ import annotations

import os

os.environ["DB_FILE"] = ":memory:"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["ANSWER_PROVIDER"] = "echo"
os.environ["LOG_LEVEL"] = "error"
# Guarantee the fallbacks are used even if the developer's shell exports keys.
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.db.database import close_connection  # noqa: E402
from app.ingest.service import get_ingestion_service, reset_ingestion_service  # noqa: E402
from app.main import create_app  # noqa: E402
from app.rag.vector_store import vector_store  # noqa: E402


@pytest.fixture
async def client():
    """An app instance with its real lifespan run: workers started, index hydrated."""
    reset_ingestion_service()
    vector_store.clear()

    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test/api") as http_client:
        async with app.router.lifespan_context(app):
            yield http_client

    close_connection()


@pytest.fixture
def drain():
    """Ingestion is async by design, so tests wait on the queue rather than sleeping."""

    async def _drain() -> None:
        await get_ingestion_service().wait_until_idle()

    return _drain
