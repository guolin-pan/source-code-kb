"""FastAPI application — SCKB Retrieval Service.

Creates a lightweight HTTP service providing retrieval and ingestion capabilities.
External applications (e.g., Copilot SKILLs, other LLM-based apps) can fetch knowledge chunks
via the HTTP API and handle rewrite/rerank/generation on their own.

Startup: sckb serve --port 8765
API docs: http://host:port/docs
"""

from __future__ import annotations

from fastapi import FastAPI

from source_code_kb.server.routes import router, set_config
from source_code_kb.config import AppConfig


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create a FastAPI application instance.

    Args:
        config: Application configuration (optional; uses defaults if not provided)

    Returns:
        A FastAPI application instance with routes configured
    """
    app = FastAPI(
        title="Source Code Knowledge Base — Retrieval Service",
        description=(
            "Source Code Knowledge Base hybrid retrieval and ingestion API service. "
            "Provides vector search, metadata filtering, hierarchical retrieval, and data ingestion."
        ),
        version="0.1.0",
    )

    # Inject the caller-supplied configuration into the routes module's
    # module-level singleton via set_config().  This must happen before any
    # request is served so that route handlers see the correct config.
    # When config is None (e.g., in tests), routes.get_config() will lazily
    # load the default configuration on first access.
    if config:
        set_config(config)

    # Register the API router which carries all /api/v1/* endpoints
    # (search, collections, ingest, etc.).  The router's prefix is defined
    # in routes.py, so no additional prefix is needed here.
    app.include_router(router)

    # A lightweight health-check endpoint at the application root.  This
    # lives outside the versioned API router so load balancers and
    # orchestrators can probe it without needing to know the API version.
    @app.get("/health")
    async def health():
        """Health check endpoint returning service status."""
        return {"status": "ok"}

    return app
