"""FastAPI application assembly and process lifecycle.

Boot order matters. Providers are constructed during startup so a bad
configuration fails immediately with a readable message instead of on the first
request, and the vector index is hydrated from SQLite before the app serves
traffic.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db.database import close_connection, get_connection
from .errors import register_exception_handlers
from .ingest.service import get_ingestion_service
from .logging_setup import configure_logging, get_logger
from .middleware import RequestContextMiddleware
from .rag.answer_service import close_answer_provider, get_answer_provider
from .rag.embeddings import close_embedding_provider, get_embedding_provider
from .rag.vector_store import vector_store
from .routers import health, ingest, items, query

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    get_connection()
    embeddings = get_embedding_provider()
    get_answer_provider()

    # SQLite is the source of truth; the vector index is a derived read model.
    vector_store.hydrate(provider=embeddings.name, model=embeddings.model)

    service = get_ingestion_service()
    await service.start()
    service.recover_unfinished()
    # Items embedded with an older model are re-embedded from their stored text,
    # so changing the embedding self-heals instead of silently matching nothing.
    service.reindex_stale()

    logger.info(
        "api ready",
        extra={"port": settings.port, "env": settings.env, "corsOrigins": settings.cors_origin_list},
    )

    yield

    logger.info("shutting down", extra=service.stats())
    await service.stop()
    await close_embedding_provider()
    await close_answer_provider()
    close_connection()


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Knowledge Inbox",
        description="Save notes and URLs, then ask questions answered only from what you saved.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-request-id"],
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    for router in (health.router, ingest.router, items.router, query.router):
        app.include_router(router, prefix="/api")

    return app


app = create_app()
