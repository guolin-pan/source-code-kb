"""Retriever factory — creates a ``HybridFusionRetriever`` (vector + graph RRF).

The retriever always uses hybrid fusion: vector similarity from ChromaDB
combined with knowledge-graph traversal merged via Reciprocal Rank Fusion.
Both underlying retrievers expose an identical
``.search(query, top_k=..., search_filter=...)`` interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import chromadb

    from source_code_kb.config import AppConfig


def create_retriever(
    collection: chromadb.Collection,
    config: AppConfig,
) -> Any:
    """Build a hybrid fusion retriever (vector + graph RRF).

    Falls back to a plain :class:`HybridRetriever` only when no persisted
    graph exists yet (e.g. first run before any ingestion).
    """
    from source_code_kb.retrieval.retriever import HybridRetriever

    vector_retriever = HybridRetriever(collection, config)

    # Attempt to load the persisted graph.
    from source_code_kb.ingest.indexer import load_graph

    graph = load_graph(config)
    if graph is None or graph.number_of_nodes() == 0:
        return vector_retriever

    # Build a chunk_lookup dict from the ChromaDB collection so the graph
    # retriever can convert graph hits back to SearchResult objects.
    chunk_lookup = _build_chunk_lookup(collection)

    from source_code_kb.graph.retriever import GraphRetriever
    from source_code_kb.retrieval.fusion import HybridFusionRetriever

    graph_retriever = GraphRetriever(graph, chunk_lookup)
    return HybridFusionRetriever(vector_retriever, graph_retriever, config)


def _build_chunk_lookup(collection: "chromadb.Collection") -> dict[str, dict[str, Any]]:
    """Build a mapping ``chunk_id → {content, metadata}`` from ChromaDB."""
    from source_code_kb.retrieval.retriever import _unflatten_metadata

    data = collection.get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    lookup: dict[str, dict[str, Any]] = {}
    for i in range(len(ids)):
        metadata = _unflatten_metadata(metas[i], ids[i])
        lookup[ids[i]] = {"content": docs[i], "metadata": metadata}
    return lookup
