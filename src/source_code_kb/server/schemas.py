"""API data models (Pydantic Schemas) — Request and response structure definitions for the retrieval service.

Uses Pydantic BaseModel to define data structures for all API endpoints.
FastAPI automatically handles request validation and OpenAPI documentation generation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────────


class SearchFilterSchema(BaseModel):
    """Search filter criteria (optional).

    All fields are optional and used to narrow the search scope.
    """

    # Every field is optional (None or empty list) so callers can supply any
    # subset of filters without having to specify the rest.  This mirrors the
    # SearchFilter dataclass in the retrieval layer but uses Pydantic types
    # for automatic request validation.
    domain: str | None = None          # Filter by exact domain
    topic: str | None = None           # Filter by exact topic
    section: str | None = None         # Filter by exact section
    scope: list[str] = Field(default_factory=list)      # Filter by scope
    tags: list[str] = Field(default_factory=list)       # Filter by tags
    min_confidence: float | None = None  # Minimum confidence score
    # Code-specific filters added in v4 to support source-code knowledge bases.
    # These allow narrowing results by file path, symbol name, programming
    # language, or architectural component—metadata that plain-text KBs lack.
    files: list[str] = Field(default_factory=list)       # Filter by file paths
    symbols: list[str] = Field(default_factory=list)     # Filter by symbol names
    language: str | None = None         # Filter by programming language
    component: str | None = None        # Filter by component/subsystem


class EntitiesSchema(BaseModel):
    """Pre-extracted code entities for graph-enhanced retrieval.

    When provided, these entities are passed to the knowledge-graph retriever
    to improve hit rates — especially for natural-language queries that don't
    contain literal symbol names.
    """

    symbols: list[str] = Field(default_factory=list)     # Function/class/macro names
    files: list[str] = Field(default_factory=list)       # Source file paths
    components: list[str] = Field(default_factory=list)  # Component/subsystem names


class SearchRequest(BaseModel):
    """POST /api/v1/search request body."""

    query: str
    collection: str = "default"
    top_k: int = 10
    filter: SearchFilterSchema | None = None
    use_reranker: bool = False          # Apply cross-encoder reranking (default: off)
    rerank_top_n: int | None = None     # Results to keep after reranking (uses config default if None)
    entities: EntitiesSchema | None = None  # Pre-extracted code entities for graph retrieval


class HierarchicalSearchRequest(BaseModel):
    """POST /api/v1/search/hierarchical request body."""

    query: str
    collection: str = "default"
    top_topics: int = 3
    top_k_per_topic: int = 5


# ── Response Models ─────────────────────────────────────────────


class SearchResultSchema(BaseModel):
    """A single search result."""

    content: str                       # Document body content
    metadata: dict[str, Any]           # Document metadata
    score: float                       # Relevance score


class GraphStatsSchema(BaseModel):
    """Graph contribution statistics for a search query."""

    vector_hits: int = 0
    graph_hits_raw: int = 0
    graph_hits_filtered: int = 0
    merged_total: int = 0
    graph_contributed: int = 0
    graph_only: int = 0
    graph_boosted: int = 0
    rank_improvements: list[dict[str, Any]] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """POST /api/v1/search response body."""

    results: list[SearchResultSchema]  # Search results list
    total: int                         # Total number of results
    graph_stats: GraphStatsSchema | None = None  # Graph contribution (present when fusion retriever is used)


class HierarchicalSearchResponse(BaseModel):
    """POST /api/v1/search/hierarchical response body."""

    matched_topics: list[str]          # List of matched topics
    results: list[SearchResultSchema]  # Search results list
    total: int                         # Total number of results


class CollectionInfo(BaseModel):
    """Basic collection information."""

    name: str
    count: int


class CollectionStatsResponse(BaseModel):
    """GET /api/v1/collections/{name}/stats response body."""

    name: str
    exists: bool
    count: int
    sample_metadata_keys: list[str] = Field(default_factory=list)


class TopicsResponse(BaseModel):
    """GET /api/v1/collections/{name}/topics response body."""

    collection: str
    topics: list[str]
    total: int


# ── Ingest Models ───────────────────────────────────────────────


class IngestRecord(BaseModel):
    """A single JSONL record for ingestion."""

    # Core fields — these map 1:1 to the v4 JSONL knowledge-base schema.
    # Pydantic enforces type checking and constraints (e.g., confidence range)
    # at deserialization time, so malformed payloads are rejected before they
    # reach the indexer.
    id: str | None = None
    content: str
    domain: str
    topic: str
    section: str
    scope: list[str]
    tags: list[str]
    confidence: float = Field(ge=0.0, le=1.0)  # Bounded [0,1] by Pydantic validator
    source: str
    updated_at: str
    meta: dict[str, Any] | None = None  # Arbitrary extra metadata (free-form JSON)
    # First-class code metadata fields (v4).  These are stored as top-level
    # metadata in the vector store so they can be filtered on directly,
    # rather than buried inside the free-form `meta` dict.
    files: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    language: str = ""
    component: str = ""
    call_chains: list[str] = Field(default_factory=list)
    api_exports: list[str] = Field(default_factory=list)
    api_imports: list[str] = Field(default_factory=list)
    ipc_mechanism: list[str] = Field(default_factory=list)
    messages_send: list[str] = Field(default_factory=list)
    messages_receive: list[str] = Field(default_factory=list)
    shared_data: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    """POST /api/v1/ingest request body.

    Provide exactly one of ``records`` (inline JSONL objects) or
    ``jsonl_content`` (raw JSONL text, one JSON object per line).
    """

    # Two-mode ingestion design:
    #   Mode 1 — `records`: caller supplies pre-structured Python objects.
    #       Pydantic validates each IngestRecord at deserialization time.
    #   Mode 2 — `jsonl_content`: caller supplies a raw JSONL string (one JSON
    #       object per line).  This is convenient for piping CLI output or
    #       uploading files without client-side parsing.
    # Exactly one of the two must be provided; the route handler enforces this.
    records: list[IngestRecord] | None = None
    jsonl_content: str | None = None
    collection: str = "default"  # Target collection; defaults to "default"


class IngestResponse(BaseModel):
    """POST /api/v1/ingest response body."""

    ingested: int                      # Number of newly ingested documents
    skipped: int                       # Number of duplicates skipped
    errors: list[str] = Field(default_factory=list)  # Validation errors (strict=False)
