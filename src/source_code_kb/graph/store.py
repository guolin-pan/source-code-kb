"""Graph persistence — save/load a NetworkX graph to/from disk."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


class GraphStore:
    """Pickle-based persistence for a ``nx.DiGraph``."""

    def __init__(self, persist_dir: str | Path) -> None:
        self._dir = Path(persist_dir)
        self._path = self._dir / "knowledge_graph.pkl"

    def save(self, graph: nx.DiGraph) -> None:
        """Serialize *graph* to disk."""
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "Graph saved (%d nodes, %d edges) → %s",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            self._path,
        )

    def load(self) -> nx.DiGraph:
        """Deserialize graph from disk.  Returns an empty graph if no file exists."""
        if not self._path.exists():
            logger.warning("No persisted graph at %s — returning empty graph", self._path)
            return nx.DiGraph()
        with open(self._path, "rb") as f:
            graph = pickle.load(f)  # noqa: S301 — trusted local file
        logger.info(
            "Graph loaded (%d nodes, %d edges) ← %s",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            self._path,
        )
        return graph

    def exists(self) -> bool:
        """Return True if a persisted graph file exists."""
        return self._path.exists()
