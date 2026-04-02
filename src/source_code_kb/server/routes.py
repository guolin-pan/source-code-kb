"""API route definitions — All HTTP endpoints for the retrieval service.

Provides the following API endpoints:
- POST /api/v1/search — Hybrid search (vector + metadata filtering)
- POST /api/v1/search/hierarchical — Hierarchical search (overview first, then details)
- GET  /api/v1/collections — List all collections
- GET  /api/v1/collections/{name}/topics — List topics for a collection
- GET  /api/v1/collections/{name}/stats — Get collection statistics

All endpoints are LLM-independent, using only Embedding + ChromaDB.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from source_code_kb.config import AppConfig, load_config
from source_code_kb.ingest.indexer import (
    create_vectorstore,
    get_collection_stats,
    get_collection_topics,
    ingest_documents,
    list_collections,
)
from source_code_kb.ingest.jsonl_loader import _validate_record, _make_doc_id
from source_code_kb.retrieval.reranker import rerank
from source_code_kb.retrieval.retriever import (
    HybridRetriever,
    SearchFilter,
)
from source_code_kb.server.schemas import (
    CollectionInfo,
    CollectionStatsResponse,
    HierarchicalSearchRequest,
    HierarchicalSearchResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResultSchema,
    TopicsResponse,
)

router = APIRouter(prefix="/api/v1")

# Module-level singleton for the application configuration.  Using a module
# global avoids passing the config through every handler signature and lets
# create_app() inject it once at startup via set_config().
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Return the current config, lazily loading defaults if none was injected."""
    global _config  # noqa: PLW0603
    # Lazy initialisation: if set_config() was never called (e.g., during
    # testing or standalone startup), fall back to the default config.
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: AppConfig) -> None:
    """Replace the module-level config singleton (called once at app startup)."""
    global _config  # noqa: PLW0603
    _config = config


def _get_retriever(collection_name: str) -> HybridRetriever:
    """Create a retriever instance for the specified collection."""
    config = get_config()
    try:
        # Build a fresh vectorstore handle for every request so that newly
        # ingested documents are always visible.  The underlying ChromaDB
        # client is lightweight, so this is inexpensive.
        collection = create_vectorstore(config, collection_name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {e}") from e
    # Wrap the vectorstore in a HybridRetriever which combines vector
    # similarity search with metadata filtering logic.
    return HybridRetriever(collection, config)


# ── Search Endpoints ────────────────────────────────────────────


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Hybrid search: vector similarity + metadata filtering.

    When ``use_reranker`` is true, results are re-scored with the configured
    cross-encoder model before being returned.
    """
    retriever = _get_retriever(req.collection)

    # Convert the Pydantic SearchFilterSchema (API layer) into the internal
    # SearchFilter dataclass (retrieval layer).  This decouples the public
    # API schema from the internal retrieval interface so either can evolve
    # independently.
    search_filter = None
    if req.filter:
        search_filter = SearchFilter(
            domain=req.filter.domain,
            topic=req.filter.topic,
            section=req.filter.section,
            scope=req.filter.scope,
            tags=req.filter.tags,
            min_confidence=req.filter.min_confidence,
            files=req.filter.files,
            symbols=req.filter.symbols,
            language=req.filter.language,
            component=req.filter.component,
        )

    results = retriever.search(
        query=req.query,
        top_k=req.top_k,
        search_filter=search_filter,
    )

    # Optional reranking pass: when enabled, a cross-encoder model re-scores
    # the initial retrieval results for higher relevance precision.  This is
    # only applied when there are results to rerank (empty list is a no-op).
    if req.use_reranker and results:
        config = get_config()
        results = rerank(
            query=req.query,
            results=results,
            config=config,
            top_n=req.rerank_top_n,  # None falls back to config default
        )

    return SearchResponse(
        results=[SearchResultSchema(**r.to_dict()) for r in results],
        total=len(results),
    )


@router.post("/search/hierarchical", response_model=HierarchicalSearchResponse)
async def hierarchical_search(req: HierarchicalSearchRequest):
    """Hierarchical search: match topics via overviews first, then search within topics."""
    retriever = _get_retriever(req.collection)

    matched_topics, results = retriever.hierarchical_search(
        query=req.query,
        top_topics=req.top_topics,
        top_k_per_topic=req.top_k_per_topic,
    )

    return HierarchicalSearchResponse(
        matched_topics=matched_topics,
        results=[SearchResultSchema(**r.to_dict()) for r in results],
        total=len(results),
    )


# ── Table Management Endpoints ──────────────────────────────────


@router.get("/collections", response_model=list[CollectionInfo])
async def collections():
    """List all collections and their document counts."""
    config = get_config()
    names = list_collections(config)
    result = []
    for name in names:
        stats = get_collection_stats(config, name)
        result.append(CollectionInfo(name=name, count=stats.get("count", 0)))
    return result


@router.get("/collections/{name}/topics", response_model=TopicsResponse)
async def topics(name: str):
    """List all topics in the specified collection."""
    config = get_config()
    topic_list = get_collection_topics(config, name)
    return TopicsResponse(collection=name, topics=topic_list, total=len(topic_list))


@router.get("/collections/{name}/stats", response_model=CollectionStatsResponse)
async def stats(name: str):
    """Get statistics for the specified collection."""
    config = get_config()
    data = get_collection_stats(config, name)
    return CollectionStatsResponse(**data)


# ── Ingest Endpoint ─────────────────────────────────────────────


def _parse_jsonl_text(jsonl_text: str) -> tuple[list[dict], list[str]]:
    """Parse raw JSONL text into records, collecting validation errors.

    Returns:
        (valid_records, error_messages)
    """
    import json

    records: list[dict] = []
    errors: list[str] = []

    # Process the raw JSONL text line by line.  Each non-blank line is
    # expected to be a self-contained JSON object.  We track line numbers
    # (1-based) so that error messages are easy for callers to correlate
    # with their input.
    for line_no, line in enumerate(jsonl_text.splitlines(), 1):
        line = line.strip()
        # Skip blank lines silently — they are common in hand-edited JSONL.
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {line_no}: invalid JSON — {e}")
            continue

        # JSONL records must be JSON objects (dicts), not arrays or scalars.
        if not isinstance(record, dict):
            errors.append(f"Line {line_no}: expected JSON object, got {type(record).__name__}")
            continue

        # Delegate to the shared validation function from jsonl_loader which
        # checks required fields, types, and value ranges.
        validation_errors = _validate_record(record, line_no)
        if validation_errors:
            errors.extend(validation_errors)
            continue

        records.append(record)

    return records, errors


def _records_to_documents(records: list[dict]) -> list[Document]:
    """Convert validated record dicts to LangChain Documents."""
    import json as _json

    from langchain_core.documents import Document

    documents: list[Document] = []
    for record in records:
        content = record["content"]
        domain = record.get("domain", "")
        topic = record.get("topic", "")
        section = record.get("section", "")

        # Use the explicit id if provided; otherwise derive a deterministic
        # id from (domain, topic, section) via _make_doc_id so that
        # re-ingesting the same logical record is idempotent.
        doc_id = record.get("id") or _make_doc_id(domain, topic, section)

        # Type coercion for list fields: JSONL producers sometimes emit a
        # bare string instead of a single-element list.  Wrapping the string
        # in a list normalises the value so downstream code can always
        # assume list[str].
        scope = record.get("scope", [])
        if isinstance(scope, str):
            scope = [scope]
        tags = record.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        # Serialise the free-form `meta` dict to a JSON string so it can be
        # stored in ChromaDB metadata (which only accepts scalar values).
        meta = record.get("meta")
        meta_str = _json.dumps(meta, ensure_ascii=False) if meta else ""

        # First-class code metadata fields — same string-to-list coercion as
        # above.  Each field receives identical treatment to be defensive
        # against inconsistent upstream producers.
        files = record.get("files", [])
        if isinstance(files, str):
            files = [files]
        symbols = record.get("symbols", [])
        if isinstance(symbols, str):
            symbols = [symbols]
        language = record.get("language", "")

        # Optional code metadata — architectural and dependency-graph fields.
        component = record.get("component", "")
        call_chains = record.get("call_chains", [])
        if isinstance(call_chains, str):
            call_chains = [call_chains]
        api_exports = record.get("api_exports", [])
        if isinstance(api_exports, str):
            api_exports = [api_exports]
        api_imports = record.get("api_imports", [])
        if isinstance(api_imports, str):
            api_imports = [api_imports]
        ipc_mechanism = record.get("ipc_mechanism", [])
        if isinstance(ipc_mechanism, str):
            ipc_mechanism = [ipc_mechanism]
        messages_send = record.get("messages_send", [])
        if isinstance(messages_send, str):
            messages_send = [messages_send]
        messages_receive = record.get("messages_receive", [])
        if isinstance(messages_receive, str):
            messages_receive = [messages_receive]
        shared_data = record.get("shared_data", [])
        if isinstance(shared_data, str):
            shared_data = [shared_data]

        # Assemble the flat metadata dict that will be stored alongside the
        # document embedding in the vector store.
        metadata = {
            "id": doc_id,
            "domain": domain,
            "topic": topic,
            "section": section,
            "scope": scope,
            "tags": tags,
            "confidence": float(record.get("confidence", 0.5)),
            "source": record.get("source", ""),
            "updated_at": record.get("updated_at", ""),
            "meta": meta_str,
            # First-class code metadata kept at the top level so the
            # retriever can filter on these fields directly.
            "files": files,
            "symbols": symbols,
            "language": language,
            "component": component,
            "call_chains": call_chains,
            "api_exports": api_exports,
            "api_imports": api_imports,
            "ipc_mechanism": ipc_mechanism,
            "messages_send": messages_send,
            "messages_receive": messages_receive,
            "shared_data": shared_data,
        }
        # Build a LangChain Document: page_content holds the text that will
        # be embedded, and metadata holds all structured fields for filtering.
        documents.append(Document(page_content=content, metadata=metadata))

    return documents


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Ingest JSONL data into the knowledge base.

    Accepts either:
    - ``records``: a list of structured JSON objects
    - ``jsonl_content``: raw JSONL text (one JSON object per line)

    All records are validated against the JSONL schema before ingestion.
    """
    # Mutual exclusion: exactly one input mode must be used.  Allowing both
    # simultaneously would create ambiguity about precedence.
    if req.records and req.jsonl_content:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'records' or 'jsonl_content', not both.",
        )
    if not req.records and not req.jsonl_content:
        raise HTTPException(
            status_code=400,
            detail="Provide 'records' (list of objects) or 'jsonl_content' (raw JSONL text).",
        )

    # Accumulate non-fatal validation errors across all records so the
    # response can report every problem at once (rather than failing on the
    # first bad record).
    errors: list[str] = []

    if req.jsonl_content:
        # Mode 2: raw JSONL text — parse line by line with full validation.
        raw_records, errors = _parse_jsonl_text(req.jsonl_content)
    else:
        # Mode 1: structured records — Pydantic has already validated types
        # and constraints, but we still run the schema-level checks from
        # jsonl_loader (e.g., required-field presence, cross-field rules).
        raw_records = []
        for idx, rec in enumerate(req.records, 1):
            rec_dict = rec.model_dump()
            validation_errors = _validate_record(rec_dict, idx)
            if validation_errors:
                errors.extend(validation_errors)
            else:
                raw_records.append(rec_dict)

    # If every record failed validation, return 422 with the collected errors
    # instead of proceeding with an empty ingest.
    if not raw_records:
        raise HTTPException(
            status_code=422,
            detail={"message": "No valid records to ingest.", "errors": errors},
        )

    # Convert plain dicts into LangChain Documents and hand them to the
    # indexer, which handles embedding, deduplication, and storage.
    documents = _records_to_documents(raw_records)
    config = get_config()
    ingested = ingest_documents(config, documents, collection_name=req.collection)

    # `skipped` = documents that the indexer considered duplicates (same id
    # already present in the collection).
    return IngestResponse(
        ingested=ingested,
        skipped=len(documents) - ingested,
        errors=errors,
    )
