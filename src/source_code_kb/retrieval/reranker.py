"""Reranking module — re-scores retrieval results using a cross-encoder model.

Supports two modes:

1. **local** (recommended): Uses sentence-transformers CrossEncoder to run
   a true cross-encoder model locally on CPU.  Produces genuine relevance
   scores and works reliably with models such as BAAI/bge-reranker-v2-m3.

2. **remote** (legacy/fallback): Calls the Ollama /api/embed endpoint with
   concatenated query+document text and computes cosine similarity.  This is
   an approximate bi-encoder approach and yields sub-optimal results because
   the reranker model was not designed for this workflow.

The mode is selected via config.reranker.mode ("local" or "remote").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from source_code_kb.config import AppConfig
    from source_code_kb.retrieval.retriever import SearchResult

console = Console()

# Lazy-loaded singleton — the cross-encoder model is expensive to load
# (several hundred MB), so we instantiate it at most once per process and
# cache it in this module-level variable for all subsequent rerank calls.
_cross_encoder = None


def _get_cross_encoder(model_name: str):
    """Lazy-load and cache the CrossEncoder model."""
    global _cross_encoder  # noqa: PLW0603
    # Only load on first invocation; subsequent calls return the cached instance.
    if _cross_encoder is None:
        # Deferred import: sentence_transformers is heavy, so we avoid
        # importing it at module level to keep startup time fast for users
        # who only use the remote reranker mode.
        from sentence_transformers import CrossEncoder

        console.print(f"[cyan]Loading local cross-encoder:[/cyan] {model_name}")
        # local_files_only=True forces the model to be loaded from the
        # local HuggingFace cache.  This avoids unexpected network calls
        # at runtime; the model must be pre-downloaded during setup.
        _cross_encoder = CrossEncoder(model_name, local_files_only=True)
    return _cross_encoder


def _metadata_list(metadata: dict, key: str) -> list[str]:
    """Extract a list-type metadata field, handling both list and comma-separated string."""
    val = metadata.get(key, [])
    # Metadata fields can arrive in two formats depending on the vector store
    # backend or ingestion path:
    #   1. A comma-separated string (e.g. "foo, bar, baz") — common when the
    #      metadata was serialised for storage that only supports flat strings.
    #   2. A native Python list (e.g. ["foo", "bar"]) — used when the store
    #      preserves structured metadata.
    # Both cases are normalised to a list[str] so callers do not need to care.
    if isinstance(val, str):
        return [s.strip() for s in val.split(",") if s.strip()]
    if isinstance(val, list):
        return [str(v) for v in val if v]
    # Fallback for unexpected types (int, None, etc.) — return empty list.
    return []


def _compute_metadata_boost(query_entities: set[str], metadata: dict) -> float:
    """Compute a score boost based on query entity overlap with result metadata.

    Checks symbols, files, call_chains, api_exports, and api_imports.
    Also checks component name match. Returns a capped boost value.
    """
    if not query_entities:
        return 0.0

    # ── Boost magnitudes per metadata category ──
    # Values are tuned heuristically.  Symbols get the highest boost because
    # an exact symbol match (function/class name, file name) is the strongest
    # signal that a result is relevant.  Call-chain and component matches are
    # slightly weaker signals (they may be transitive), and API import/export
    # overlap is weakest because many chunks may share the same API surface.
    SYMBOL_BOOST = 0.15
    CALL_CHAIN_BOOST = 0.10
    API_BOOST = 0.08
    COMPONENT_BOOST = 0.10
    # Hard cap prevents metadata boost from dominating the cross-encoder's
    # semantic score.  Without a cap, a result that matches several categories
    # could overtake a semantically superior result.
    MAX_BOOST = 0.25

    boost = 0.0

    # ── Category 1: Symbol / file name overlap ──
    # Direct mentions of functions, classes, or filenames in the query that
    # also appear in the result metadata are a strong relevance indicator.
    result_entities = set(
        s.lower() for s in _metadata_list(metadata, "symbols") + _metadata_list(metadata, "files")
    )
    if query_entities & result_entities:
        boost += SYMBOL_BOOST

    # ── Category 2: Call-chain overlap ──
    # Call chains are stored as "A → B → C" or "A -> B -> C".  We split on
    # both arrow variants and check whether any individual symbol in the chain
    # matches a query entity.  This catches cases where the user asks about
    # a function that appears mid-chain rather than at the top level.
    call_chains = _metadata_list(metadata, "call_chains")
    chain_symbols: set[str] = set()
    for chain in call_chains:
        # Normalize arrow variants and split
        for part in chain.replace("→", "->").split("->"):
            part = part.strip().lower()
            if part:
                chain_symbols.add(part)
    if query_entities & chain_symbols:
        boost += CALL_CHAIN_BOOST

    # ── Category 3: API export / import overlap ──
    # A match here means the query mentions a public API symbol that this
    # chunk either exports (defines) or imports (consumes).  The boost is
    # lower than symbol overlap because API annotations may be shared across
    # many chunks in the same module.
    api_entities = set(
        s.lower()
        for s in _metadata_list(metadata, "api_exports") + _metadata_list(metadata, "api_imports")
    )
    if query_entities & api_entities:
        boost += API_BOOST

    # ── Category 4: Component / subsystem name match ──
    # If the user explicitly mentions a component name (e.g., "alarm manager"),
    # chunks belonging to that component should be preferred.
    component = metadata.get("component", "")
    if component and component.lower() in query_entities:
        boost += COMPONENT_BOOST

    # Apply the cap so metadata signals never outweigh semantic relevance.
    return min(boost, MAX_BOOST)


def _rerank_local(
    query: str,
    results: list[SearchResult],
    model_name: str,
    top_n: int,
) -> list[SearchResult]:
    """Rerank using a local CrossEncoder model (true cross-encoding).

    Applies metadata boost: if a result's metadata (symbols, files, call_chains,
    api_exports, api_imports, component) overlaps with code entities from the query,
    an additional score boost is applied.
    """
    from source_code_kb.retrieval.retriever import _extract_code_entities

    encoder = _get_cross_encoder(model_name)

    # Build (query, document) pairs — the cross-encoder jointly encodes both
    # texts through its transformer layers, producing a single relevance score
    # per pair.  This is more accurate than a bi-encoder (separate embeddings
    # + cosine) because the model can attend across query and document tokens.
    pairs = [(query, r.content) for r in results]

    # Score all pairs in a single forward pass (batch prediction).  This is
    # much faster than scoring one pair at a time because it allows the model
    # to parallelise computation on the CPU/GPU.
    scores = encoder.predict(pairs)

    # Extract code-like entities (function names, file paths, etc.) from the
    # query text.  These are used to compute a metadata-based bonus so that
    # results whose metadata explicitly mentions the queried symbol get an
    # additional relevance nudge on top of the cross-encoder score.
    query_symbols, query_files = _extract_code_entities(query)
    query_entities = set(s.lower() for s in query_symbols + query_files)

    # Combine the cross-encoder's semantic score with the metadata boost.
    scored = []
    for result, score in zip(results, scores):
        boost = _compute_metadata_boost(query_entities, result.metadata)
        scored.append((result, float(score) + boost))

    # Sort by combined score in descending order (highest relevance first).
    scored.sort(key=lambda x: x[1], reverse=True)

    # Truncate to the requested top_n and write the final score back into
    # each SearchResult so downstream consumers can inspect it.
    reranked = []
    for result, score in scored[:top_n]:
        result.score = score
        reranked.append(result)
    return reranked


def _rerank_remote(
    query: str,
    results: list[SearchResult],
    model: str,
    base_url: str,
    top_n: int,
) -> list[SearchResult]:
    """Rerank using the Ollama /api/embed endpoint (approximate bi-encoder).

    Also applies metadata boost for consistency with local mode.
    """
    import httpx

    from source_code_kb.retrieval.retriever import _extract_code_entities

    url = f"{base_url.rstrip('/')}/api/embed"

    # ── Approximation strategy ──
    # Ollama does not expose a native cross-encoder scoring endpoint.  As a
    # workaround we concatenate the query and each document into a single
    # string ("query: ... \n document: ...") and obtain an embedding for the
    # concatenated text via /api/embed.  We then compute cosine similarity
    # between the pure-query embedding and each concatenated-pair embedding.
    #
    # This is a *bi-encoder approximation*: it captures some interaction
    # between query and document (because the model sees both in one text),
    # but it lacks the joint cross-attention mechanism of a true cross-encoder.
    # Results are therefore sub-optimal compared to local mode.
    pair_texts = [f"query: {query}\ndocument: {r.content}" for r in results]

    # Embed everything in a single HTTP round-trip: the standalone query
    # embedding is at index 0, followed by one embedding per pair text.
    all_texts = [query] + pair_texts

    resp = httpx.post(
        url,
        json={"model": model, "input": all_texts},
        timeout=120.0,
    )
    resp.raise_for_status()
    embeddings = resp.json()["embeddings"]

    # Separate the query embedding (index 0) from the pair embeddings.
    query_emb = np.array(embeddings[0])
    pair_embs = [np.array(e) for e in embeddings[1:]]

    # Extract code entities for metadata boost (same logic as local mode).
    query_symbols, query_files = _extract_code_entities(query)
    query_entities = set(s.lower() for s in query_symbols + query_files)

    # Compute cosine similarity for each (query_emb, pair_emb) pair and add
    # the metadata boost.  Cosine similarity = dot(a, b) / (||a|| * ||b||).
    scored = []
    for result, pair_emb in zip(results, pair_embs):
        denom = np.linalg.norm(query_emb) * np.linalg.norm(pair_emb)
        # Guard against zero-norm vectors to avoid division by zero.
        sim = float(np.dot(query_emb, pair_emb) / denom) if denom > 0 else 0.0
        boost = _compute_metadata_boost(query_entities, result.metadata)
        scored.append((result, sim + boost))

    # Sort and truncate, same as local mode.
    scored.sort(key=lambda x: x[1], reverse=True)

    reranked = []
    for result, score in scored[:top_n]:
        result.score = score
        reranked.append(result)
    return reranked


def rerank(
    query: str,
    results: list[SearchResult],
    config: AppConfig,
    top_n: int | None = None,
) -> list[SearchResult]:
    """Rerank retrieval results using the configured reranker backend.

    Args:
        query: User query text.
        results: Initial retrieval results.
        config: Application configuration.
        top_n: Number of results to retain (defaults to config.reranker.top_n).

    Returns:
        The reranked list of SearchResult objects.
    """
    if not results:
        return results

    # Use caller-specified top_n, or fall back to the configured default.
    n = top_n or config.reranker.top_n

    try:
        # Dispatch to the appropriate backend based on configuration.
        # "local" uses a true cross-encoder for best accuracy;
        # "remote" falls back to the Ollama bi-encoder approximation.
        if config.reranker.mode == "local":
            return _rerank_local(query, results, config.reranker.model, n)
        else:
            return _rerank_remote(
                query, results, config.reranker.model,
                config.reranker.base_url, n,
            )
    except Exception as e:
        # Graceful degradation: if the reranker fails (model not found,
        # network timeout, OOM, etc.), return the original retrieval order
        # truncated to top_n rather than crashing the entire query pipeline.
        console.print(f"[yellow]Reranker error: {e}. Returning original order.[/yellow]")
        return results[:n]
