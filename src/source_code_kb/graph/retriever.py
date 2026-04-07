"""Graph-based retriever — traverses the knowledge graph to find related chunks.

Given a user query, extracts code entities (symbols, files), locates them in
the graph, and performs a bounded BFS to discover related chunks through
typed relationships (calls, api imports/exports, IPC, shared data, etc.).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

import networkx as nx

from source_code_kb.graph.schema import NodeType
from source_code_kb.retrieval.retriever import SearchResult, extract_code_entities

if TYPE_CHECKING:
    from source_code_kb.config import AppConfig

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieve knowledge-base chunks by traversing the knowledge graph."""

    def __init__(self, graph: nx.DiGraph, chunk_lookup: dict[str, dict[str, Any]]) -> None:
        """
        Args:
            graph: The knowledge graph (as built by ``KnowledgeGraphBuilder``).
            chunk_lookup: Mapping from chunk_id → ``{content, metadata}``
                          so that graph hits can be converted to SearchResult.
        """
        self._graph = graph
        self._chunk_lookup = chunk_lookup
        # Pre-build a name → node_id index for fast entity resolution.
        self._name_index: dict[str, list[str]] = {}
        for node_id, data in graph.nodes(data=True):
            name = data.get("name", "")
            if name:
                self._name_index.setdefault(name, []).append(node_id)
                # Also index lowercase for case-insensitive lookup.
                lower = name.lower()
                if lower != name:
                    self._name_index.setdefault(lower, []).append(node_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        max_hops: int = 2,
        top_k: int = 10,
        entities: dict[str, list[str]] | None = None,
    ) -> list[SearchResult]:
        """Find chunks related to symbols/files mentioned in *query*.

        1. Extract code entities from the query text (or use pre-extracted
           *entities* from the LLM rewrite step).
        2. Resolve entities to graph nodes.
        3. BFS up to *max_hops* to collect reachable ``chunk:*`` nodes.
        4. Score by inverse distance and return up to *top_k* results.

        Args:
            entities: Pre-extracted entities dict with optional keys
                ``"symbols"``, ``"files"``, ``"components"``.  When provided,
                these are used *in addition to* regex-extracted entities from
                the query text, dramatically improving graph hit rates for
                natural-language questions.
        """
        symbols, files = extract_code_entities(query)

        # Merge LLM-extracted entities when available.
        if entities:
            symbols = list(dict.fromkeys(symbols + entities.get("symbols", [])))
            files = list(dict.fromkeys(files + entities.get("files", [])))

        if not symbols and not files:
            # No entities from either source — try component names as symbols.
            if entities and entities.get("components"):
                symbols = entities["components"]
            else:
                return []

        # Resolve entities to graph node IDs.
        seed_nodes: list[str] = []
        for sym in symbols:
            seed_nodes.extend(self._resolve(sym, NodeType.SYMBOL))
        for fpath in files:
            seed_nodes.extend(self._resolve(fpath, NodeType.FILE))

        # Also try resolving component names as COMPONENT nodes.
        if entities and entities.get("components"):
            for comp in entities["components"]:
                seed_nodes.extend(self._resolve(comp, NodeType.COMPONENT))

        if not seed_nodes:
            return []

        # BFS traversal — collect chunk IDs reachable within max_hops.
        chunk_distances = self._bfs_chunks(seed_nodes, max_hops)
        if not chunk_distances:
            return []

        # Convert to SearchResult, scored by inverse hop distance.
        results: list[SearchResult] = []
        for chunk_id, distance in sorted(chunk_distances.items(), key=lambda x: x[1]):
            info = self._chunk_lookup.get(chunk_id)
            if info is None:
                continue
            # Score: 1.0 for distance 0 (direct), decaying with hops.
            score = 1.0 / (1.0 + distance)
            results.append(SearchResult(
                content=info["content"],
                metadata=info["metadata"],
                score=score,
            ))
            if len(results) >= top_k:
                break

        return results

    def trace_call_chain(
        self, start: str, end: str, max_depth: int = 6
    ) -> list[list[str]]:
        """Find all call-chain paths between two symbol names (up to *max_depth*)."""
        start_nodes = self._resolve(start, NodeType.SYMBOL)
        end_nodes = set(self._resolve(end, NodeType.SYMBOL))
        if not start_nodes or not end_nodes:
            return []

        paths: list[list[str]] = []
        for sn in start_nodes:
            for en in end_nodes:
                try:
                    for p in nx.all_simple_paths(self._graph, sn, en, cutoff=max_depth):
                        # Extract human-readable names from node IDs.
                        names = [
                            self._graph.nodes[n].get("name", n) for n in p
                        ]
                        paths.append(names)
                except nx.NetworkXError:
                    pass
        return paths

    def find_dependencies(
        self, symbol: str, direction: str = "both", max_hops: int = 2
    ) -> list[str]:
        """Return symbol names reachable from *symbol* within *max_hops*.

        Args:
            direction: ``"downstream"`` (successors), ``"upstream"``
                       (predecessors), or ``"both"``.
        """
        nodes = self._resolve(symbol, NodeType.SYMBOL)
        if not nodes:
            return []

        found: set[str] = set()
        for node in nodes:
            if direction in ("downstream", "both"):
                found.update(self._bfs_symbols(node, max_hops, reverse=False))
            if direction in ("upstream", "both"):
                found.update(self._bfs_symbols(node, max_hops, reverse=True))

        # Remove the query symbol itself from results.
        found.discard(symbol)
        return sorted(found)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, name: str, expected_type: NodeType) -> list[str]:
        """Resolve a human-readable name to graph node IDs."""
        # Try exact canonical ID first.
        canonical = f"{expected_type.value}:{name}"
        if canonical in self._graph:
            return [canonical]
        # Fall back to name index (case-insensitive).
        candidates = self._name_index.get(name, []) + self._name_index.get(name.lower(), [])
        return list(dict.fromkeys(candidates))  # deduplicate, preserve order

    def _bfs_chunks(
        self, seeds: list[str], max_hops: int
    ) -> dict[str, int]:
        """BFS from *seeds*, collecting chunk node IDs and their distances."""
        visited: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()

        for s in seeds:
            if s not in visited:
                visited[s] = 0
                queue.append((s, 0))

        chunk_distances: dict[str, int] = {}

        while queue:
            node, dist = queue.popleft()

            # If this node is a chunk, record it.
            if node.startswith(f"{NodeType.CHUNK.value}:"):
                # Extract the chunk_id (the part after "chunk:").
                chunk_id = node[len(f"{NodeType.CHUNK.value}:"):]
                if chunk_id not in chunk_distances or dist < chunk_distances[chunk_id]:
                    chunk_distances[chunk_id] = dist

            # Also collect chunk_ids referenced in edge attributes.
            for _, _, edata in self._graph.edges(node, data=True):
                cid = edata.get("chunk_id")
                if cid and (cid not in chunk_distances or dist < chunk_distances[cid]):
                    chunk_distances[cid] = dist

            # Reverse edges too (predecessors).
            for _, _, edata in self._graph.in_edges(node, data=True):
                cid = edata.get("chunk_id")
                if cid and (cid not in chunk_distances or dist < chunk_distances[cid]):
                    chunk_distances[cid] = dist

            if dist >= max_hops:
                continue

            # Expand to successors and predecessors.
            for neighbor in self._graph.successors(node):
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
            for neighbor in self._graph.predecessors(node):
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        return chunk_distances

    def _bfs_symbols(self, start: str, max_hops: int, reverse: bool) -> set[str]:
        """BFS collecting symbol *names* reachable from *start*."""
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        symbols: set[str] = set()

        while queue:
            node, dist = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            ndata = self._graph.nodes.get(node, {})
            if ndata.get("type") == NodeType.SYMBOL.value:
                name = ndata.get("name", "")
                if name:
                    symbols.add(name)

            if dist >= max_hops:
                continue

            neighbors = (
                self._graph.predecessors(node) if reverse
                else self._graph.successors(node)
            )
            for nb in neighbors:
                if nb not in visited:
                    queue.append((nb, dist + 1))

        return symbols
