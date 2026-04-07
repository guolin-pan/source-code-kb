"""Hybrid Fusion Retriever — merges vector + graph retrieval via Reciprocal Rank Fusion.

Runs :class:`HybridRetriever` (vector search) and :class:`GraphRetriever`
(knowledge graph traversal) in parallel, then combines results using a
weighted RRF formula.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from source_code_kb.retrieval.retriever import SearchResult

if TYPE_CHECKING:
    from source_code_kb.config import AppConfig
    from source_code_kb.graph.retriever import GraphRetriever
    from source_code_kb.retrieval.retriever import HybridRetriever, SearchFilter


class HybridFusionRetriever:
    """Reciprocal Rank Fusion of vector and graph retrieval results."""

    def __init__(
        self,
        vector_retriever: HybridRetriever,
        graph_retriever: GraphRetriever,
        config: AppConfig,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.config = config

        graph_cfg = getattr(config, "graph", None)
        self._alpha: float = getattr(graph_cfg, "fusion_alpha", 0.6) if graph_cfg else 0.6
        self._rrf_k: int = getattr(graph_cfg, "rrf_k", 60) if graph_cfg else 60
        self._max_hops: int = getattr(graph_cfg, "max_hops", 2) if graph_cfg else 2

        # Per-search statistics — populated after each search() call so that
        # callers (CLI, agent nodes) can display graph contribution hints.
        self.last_search_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API — same signature as HybridRetriever.search so it is a
    #              drop-in replacement.
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        search_filter: "SearchFilter | None" = None,
        entities: dict[str, list[str]] | None = None,
    ) -> list[SearchResult]:
        """Run vector + graph search and merge via weighted RRF.

        Args:
            entities: Pre-extracted code entities from LLM rewrite (symbols,
                      files, components).  Passed to ``GraphRetriever.search``
                      to improve graph hit rates.
        """
        from source_code_kb.retrieval.retriever import SearchFilter as SF  # noqa: F811

        k = top_k or self.config.retrieval.top_k

        # --- Vector search ---
        vector_results = self.vector_retriever.search(query, top_k=k, search_filter=search_filter)

        # --- Graph search (with LLM-extracted entities when available) ---
        graph_results_raw = self.graph_retriever.search(
            query, max_hops=self._max_hops, top_k=k, entities=entities,
        )

        # Apply the same metadata filter to graph results so that filtered
        # queries don't get unfiltered noise injected via the graph path.
        graph_results = graph_results_raw
        if search_filter and graph_results:
            graph_results = _apply_filter(graph_results, search_filter)

        # --- RRF merge ---
        merged = self._rrf_merge(vector_results, graph_results, top_k=k)

        # ── Capture detailed per-search statistics ──
        # These are consumed by CLI, agent nodes, and REST API to visualise
        # the graph's contribution to retrieval quality.

        # Build quick id→rank maps for vector-only ordering.
        def _did(r: SearchResult) -> str:
            return r.metadata.get("id", r.content[:50])

        vector_rank = {_did(r): i + 1 for i, r in enumerate(vector_results)}

        # Per-result breakdown.
        graph_only = 0
        graph_boosted = 0
        rank_improvements: list[dict] = []  # docs whose rank improved due to graph

        for fused_rank, r in enumerate(merged, start=1):
            src = r.metadata.get("retrieval_source", "vector")
            did = _did(r)
            if src == "graph":
                graph_only += 1
                rank_improvements.append({
                    "id": did,
                    "topic": r.metadata.get("topic", "?"),
                    "source": "graph",
                    "vector_rank": None,
                    "fused_rank": fused_rank,
                    "change": "new",
                })
            elif src == "vector+graph":
                graph_boosted += 1
                v_rank = vector_rank.get(did)
                if v_rank is not None and fused_rank < v_rank:
                    rank_improvements.append({
                        "id": did,
                        "topic": r.metadata.get("topic", "?"),
                        "source": "vector+graph",
                        "vector_rank": v_rank,
                        "fused_rank": fused_rank,
                        "change": f"↑{v_rank - fused_rank}",
                    })

        self.last_search_stats = {
            "vector_hits": len(vector_results),
            "graph_hits_raw": len(graph_results_raw),
            "graph_hits_filtered": len(graph_results),
            "merged_total": len(merged),
            "graph_contributed": graph_only + graph_boosted,
            "graph_only": graph_only,
            "graph_boosted": graph_boosted,
            "rank_improvements": rank_improvements,
        }

        return merged

    def hierarchical_search(
        self,
        query: str,
        top_topics: int = 3,
        top_k_per_topic: int = 5,
    ) -> tuple[list[str], list[SearchResult]]:
        """Delegate hierarchical search to the vector retriever."""
        return self.vector_retriever.hierarchical_search(
            query=query,
            top_topics=top_topics,
            top_k_per_topic=top_k_per_topic,
        )

    # ------------------------------------------------------------------
    # RRF implementation
    # ------------------------------------------------------------------

    def _rrf_merge(
        self,
        vector_results: list[SearchResult],
        graph_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Merge two ranked lists using weighted Reciprocal Rank Fusion.

        RRF score for a document *d*:
            score(d) = α / (k + rank_vector(d)) + (1 - α) / (k + rank_graph(d))
        where *k* = ``rrf_k`` (default 60) and *α* = ``fusion_alpha``.

        Scores are normalised to [0, 1] by dividing by the theoretical
        maximum ``1 / (k + 1)`` so that consumers don't need to know the
        RRF internals to interpret relevance.
        """
        alpha = self._alpha
        rrf_k = self._rrf_k
        max_score = 1.0 / (rrf_k + 1)  # theoretical max for one source at rank-1

        # Map: chunk_id → {SearchResult, rrf_score}
        scores: dict[str, float] = {}
        best_result: dict[str, SearchResult] = {}
        vector_ids: set[str] = set()
        graph_ids: set[str] = set()

        def _doc_id(r: SearchResult) -> str:
            return r.metadata.get("id", r.content[:50])

        # Score vector results.
        for rank, r in enumerate(vector_results, start=1):
            did = _doc_id(r)
            scores[did] = scores.get(did, 0.0) + alpha / (rrf_k + rank)
            vector_ids.add(did)
            if did not in best_result or r.score > best_result[did].score:
                best_result[did] = r

        # Score graph results.
        for rank, r in enumerate(graph_results, start=1):
            did = _doc_id(r)
            scores[did] = scores.get(did, 0.0) + (1 - alpha) / (rrf_k + rank)
            graph_ids.add(did)
            if did not in best_result or r.score > best_result[did].score:
                best_result[did] = r

        # Build the merged list sorted by RRF score descending.
        merged: list[SearchResult] = []
        for did, rrf_score in sorted(scores.items(), key=lambda x: -x[1]):
            r = best_result[did]
            in_vector = did in vector_ids
            in_graph = did in graph_ids
            if in_vector and in_graph:
                source = "vector+graph"
            elif in_graph:
                source = "graph"
            else:
                source = "vector"
            metadata = {**r.metadata, "retrieval_source": source}
            merged.append(SearchResult(
                content=r.content,
                metadata=metadata,
                score=round(rrf_score / max_score, 4),
            ))
            if len(merged) >= top_k:
                break

        return merged


# ── Post-filter helper ───────────────────────────────────────────


def _apply_filter(
    results: list[SearchResult], search_filter: "SearchFilter"
) -> list[SearchResult]:
    """Post-filter graph results by the same criteria used for vector search."""
    filtered: list[SearchResult] = []
    for r in results:
        m = r.metadata
        if search_filter.domain and m.get("domain") != search_filter.domain:
            continue
        if search_filter.topic and m.get("topic") != search_filter.topic:
            continue
        if search_filter.section and m.get("section") != search_filter.section:
            continue
        if search_filter.language and m.get("language") != search_filter.language:
            continue
        if search_filter.component and m.get("component") != search_filter.component:
            continue
        if search_filter.min_confidence is not None:
            conf = m.get("confidence", 0.0)
            if isinstance(conf, (int, float)) and conf < search_filter.min_confidence:
                continue
        filtered.append(r)
    return filtered


# ── Formatting helpers ───────────────────────────────────────────


def graph_stats_summary(stats: dict) -> str:
    """Build a compact one-line graph summary for agent-mode _status field.

    Returns a Rich-tag string like ``  |  Graph: 3/10(30%), 1 new, 2 boosted``
    or ``  |  Graph: no hits``.  Returns an empty string when *stats* is empty.
    """
    if not stats:
        return ""
    g_only = stats.get("graph_only", 0)
    g_boost = stats.get("graph_boosted", 0)
    g_total = g_only + g_boost
    merged = stats.get("merged_total", 0)
    if g_total == 0 and stats.get("graph_hits_raw", 0) == 0:
        return "  |  Graph: no hits"
    pct = round(g_total / merged * 100) if merged else 0
    parts = [f"{g_total}/{merged}({pct}%)"]
    if g_only:
        parts.append(f"{g_only} new")
    if g_boost:
        parts.append(f"{g_boost} boosted")
    return f"  |  Graph: [cyan]{', '.join(parts)}[/cyan]"
