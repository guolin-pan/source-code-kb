"""Build a NetworkX knowledge graph from JSONL chunk metadata.

Parses relationship fields (call_chains, api_exports, api_imports,
ipc_mechanism, shared_data, symbols, files, component) from each chunk
and creates typed nodes + edges.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from source_code_kb.graph.schema import EdgeType, NodeType

logger = logging.getLogger(__name__)


def _node_id(node_type: NodeType, name: str) -> str:
    """Create a canonical node ID: ``<type>:<name>``."""
    return f"{node_type.value}:{name}"


def _parse_call_chain(chain: str) -> list[str]:
    """Parse a call-chain string like ``"A→B→C"`` into ``['A', 'B', 'C']``."""
    # Support both → (Unicode arrow) and -> (ASCII)
    parts = chain.replace("->", "→").split("→")
    return [p.strip() for p in parts if p.strip()]


class KnowledgeGraphBuilder:
    """Incrementally builds a ``nx.DiGraph`` from knowledge-base chunks."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    # -- public API -------------------------------------------------------

    def build_from_chunks(self, chunks: list[dict[str, Any]]) -> nx.DiGraph:
        """Build (or extend) the graph from a list of raw chunk dicts.

        Each chunk dict is expected to have the same schema as a loaded JSONL
        record (i.e. *after* ``_unflatten_metadata``).

        Returns:
            The accumulated directed graph.
        """
        for chunk in chunks:
            self._process_chunk(chunk)
        logger.info(
            "Graph built: %d nodes, %d edges",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )
        return self._graph

    def get_graph(self) -> nx.DiGraph:
        """Return the current graph (may be empty if ``build_from_chunks`` was not called)."""
        return self._graph

    def clear(self) -> None:
        """Reset the graph to empty."""
        self._graph.clear()

    # -- internal ---------------------------------------------------------

    def _ensure_node(self, node_id: str, node_type: NodeType, **attrs: Any) -> None:
        if node_id not in self._graph:
            self._graph.add_node(node_id, type=node_type.value, **attrs)

    def _ensure_edge(
        self, src: str, dst: str, edge_type: EdgeType, **attrs: Any
    ) -> None:
        # If edge already exists, update attributes instead of adding a duplicate.
        if self._graph.has_edge(src, dst):
            self._graph[src][dst].setdefault("types", set()).add(edge_type.value)
            self._graph[src][dst].update(attrs)
        else:
            self._graph.add_edge(
                src, dst, type=edge_type.value, types={edge_type.value}, **attrs
            )

    def _process_chunk(self, chunk: dict[str, Any]) -> None:
        chunk_id = chunk.get("id", "")
        if not chunk_id:
            return

        component_name: str = chunk.get("component", "")
        source: str = chunk.get("source", "")
        confidence: float = float(chunk.get("confidence", 0.0))

        # --- Chunk node ---
        chunk_node = _node_id(NodeType.CHUNK, chunk_id)
        self._ensure_node(
            chunk_node,
            NodeType.CHUNK,
            domain=chunk.get("domain", ""),
            topic=chunk.get("topic", ""),
            section=chunk.get("section", ""),
            source=source,
            confidence=confidence,
        )

        # --- Component node ---
        comp_node: str | None = None
        if component_name:
            comp_node = _node_id(NodeType.COMPONENT, component_name)
            self._ensure_node(comp_node, NodeType.COMPONENT, name=component_name, source=source)

        # --- Symbols ---
        symbols: list[str] = chunk.get("symbols", []) or []
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]

        for sym in symbols:
            sym_node = _node_id(NodeType.SYMBOL, sym)
            self._ensure_node(sym_node, NodeType.SYMBOL, name=sym, source=source)
            # chunk -> contains_symbol -> symbol
            self._ensure_edge(chunk_node, sym_node, EdgeType.CONTAINS_SYMBOL, chunk_id=chunk_id)
            # symbol -> belongs_to -> component
            if comp_node:
                self._ensure_edge(sym_node, comp_node, EdgeType.BELONGS_TO, chunk_id=chunk_id)

        # --- Files ---
        files: list[str] = chunk.get("files", []) or []
        if isinstance(files, str):
            files = [f.strip() for f in files.split(",") if f.strip()]

        for fpath in files:
            file_node = _node_id(NodeType.FILE, fpath)
            self._ensure_node(file_node, NodeType.FILE, name=fpath, source=source)
            self._ensure_edge(chunk_node, file_node, EdgeType.CONTAINS_FILE, chunk_id=chunk_id)
            if comp_node:
                self._ensure_edge(file_node, comp_node, EdgeType.BELONGS_TO, chunk_id=chunk_id)

        # --- Call chains ---
        call_chains: list[str] = chunk.get("call_chains", []) or []
        if isinstance(call_chains, str):
            call_chains = [c.strip() for c in call_chains.split(",") if c.strip()]

        for chain in call_chains:
            parts = _parse_call_chain(chain)
            for i in range(len(parts) - 1):
                caller_node = _node_id(NodeType.SYMBOL, parts[i])
                callee_node = _node_id(NodeType.SYMBOL, parts[i + 1])
                self._ensure_node(caller_node, NodeType.SYMBOL, name=parts[i], source=source)
                self._ensure_node(callee_node, NodeType.SYMBOL, name=parts[i + 1], source=source)
                self._ensure_edge(
                    caller_node, callee_node, EdgeType.CALLS,
                    chunk_id=chunk_id, confidence=confidence,
                )

        # --- API exports / imports ---
        api_exports: list[str] = chunk.get("api_exports", []) or []
        if isinstance(api_exports, str):
            api_exports = [a.strip() for a in api_exports.split(",") if a.strip()]
        for api_sym in api_exports:
            sym_node = _node_id(NodeType.SYMBOL, api_sym)
            self._ensure_node(sym_node, NodeType.SYMBOL, name=api_sym, source=source)
            if comp_node:
                self._ensure_edge(comp_node, sym_node, EdgeType.EXPORTS_API, chunk_id=chunk_id)

        api_imports: list[str] = chunk.get("api_imports", []) or []
        if isinstance(api_imports, str):
            api_imports = [a.strip() for a in api_imports.split(",") if a.strip()]
        for api_sym in api_imports:
            sym_node = _node_id(NodeType.SYMBOL, api_sym)
            self._ensure_node(sym_node, NodeType.SYMBOL, name=api_sym, source=source)
            if comp_node:
                self._ensure_edge(comp_node, sym_node, EdgeType.IMPORTS_API, chunk_id=chunk_id)

        # --- IPC mechanisms ---
        ipc_mechs: list[str] = chunk.get("ipc_mechanism", []) or []
        if isinstance(ipc_mechs, str):
            ipc_mechs = [m.strip() for m in ipc_mechs.split(",") if m.strip()]
        for mech in ipc_mechs:
            ipc_node = _node_id(NodeType.IPC_CHANNEL, mech)
            self._ensure_node(ipc_node, NodeType.IPC_CHANNEL, name=mech, source=source)
            if comp_node:
                self._ensure_edge(comp_node, ipc_node, EdgeType.IPC_SENDS, chunk_id=chunk_id)

        # --- Messages send / receive (mapped to IPC edges) ---
        msgs_send: list[str] = chunk.get("messages_send", []) or []
        if isinstance(msgs_send, str):
            msgs_send = [m.strip() for m in msgs_send.split(",") if m.strip()]
        for msg in msgs_send:
            ipc_node = _node_id(NodeType.IPC_CHANNEL, msg)
            self._ensure_node(ipc_node, NodeType.IPC_CHANNEL, name=msg, source=source)
            if comp_node:
                self._ensure_edge(comp_node, ipc_node, EdgeType.IPC_SENDS, chunk_id=chunk_id)

        msgs_recv: list[str] = chunk.get("messages_receive", []) or []
        if isinstance(msgs_recv, str):
            msgs_recv = [m.strip() for m in msgs_recv.split(",") if m.strip()]
        for msg in msgs_recv:
            ipc_node = _node_id(NodeType.IPC_CHANNEL, msg)
            self._ensure_node(ipc_node, NodeType.IPC_CHANNEL, name=msg, source=source)
            if comp_node:
                self._ensure_edge(comp_node, ipc_node, EdgeType.IPC_RECEIVES, chunk_id=chunk_id)

        # --- Shared data ---
        shared: list[str] = chunk.get("shared_data", []) or []
        if isinstance(shared, str):
            shared = [s.strip() for s in shared.split(",") if s.strip()]
        for ds in shared:
            ds_node = _node_id(NodeType.DATA_STRUCTURE, ds)
            self._ensure_node(ds_node, NodeType.DATA_STRUCTURE, name=ds, source=source)
            if comp_node:
                self._ensure_edge(comp_node, ds_node, EdgeType.SHARES_DATA, chunk_id=chunk_id)
