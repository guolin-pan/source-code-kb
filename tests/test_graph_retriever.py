"""Tests for the graph-based retriever."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from source_code_kb.graph.builder import KnowledgeGraphBuilder
from source_code_kb.graph.retriever import GraphRetriever
from source_code_kb.graph.schema import NodeType

TEST_DATA = Path(__file__).parent / "test_data.jsonl"


def _load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _build_chunk_lookup(chunks: list[dict]) -> dict[str, dict]:
    return {
        c["id"]: {"content": c["content"], "metadata": c}
        for c in chunks
        if c.get("id")
    }


@pytest.fixture()
def graph_retriever() -> GraphRetriever:
    chunks = _load_chunks(TEST_DATA)
    builder = KnowledgeGraphBuilder()
    graph = builder.build_from_chunks(chunks)
    lookup = _build_chunk_lookup(chunks)
    return GraphRetriever(graph, lookup)


class TestGraphRetriever:
    def test_search_returns_results_for_known_symbol(self, graph_retriever: GraphRetriever):
        """Query mentioning start_kernel should return related chunks."""
        results = graph_retriever.search("What does start_kernel do?", max_hops=2, top_k=5)
        assert len(results) > 0
        # start_kernel appears in at least two chunks (Boot Overview, Internals_start_kernel).
        contents = [r.content for r in results]
        assert any("start_kernel" in c for c in contents)

    def test_search_returns_empty_for_unknown(self, graph_retriever: GraphRetriever):
        """Query with no code entities returns empty."""
        results = graph_retriever.search("What is the meaning of life?")
        assert results == []

    def test_search_by_file_path(self, graph_retriever: GraphRetriever):
        """Query mentioning init/main.c should return related chunks."""
        results = graph_retriever.search("Explain init/main.c", max_hops=2, top_k=10)
        assert len(results) > 0

    def test_search_multi_hop(self, graph_retriever: GraphRetriever):
        """A 2-hop search from mm_init should reach the boot process chunks."""
        results = graph_retriever.search(
            "What calls mm_init during boot?",
            max_hops=2,
            top_k=10,
        )
        assert len(results) > 0

    def test_search_top_k_limit(self, graph_retriever: GraphRetriever):
        results = graph_retriever.search("start_kernel boot process", max_hops=3, top_k=2)
        assert len(results) <= 2

    def test_search_scores_are_positive(self, graph_retriever: GraphRetriever):
        results = graph_retriever.search("start_kernel", max_hops=2, top_k=5)
        for r in results:
            assert r.score > 0

    def test_trace_call_chain(self, graph_retriever: GraphRetriever):
        """Trace from go_to_protected_mode to start_kernel should find a path."""
        paths = graph_retriever.trace_call_chain("go_to_protected_mode", "start_kernel")
        assert len(paths) > 0
        # The path should include the intermediate symbols.
        for p in paths:
            assert "go_to_protected_mode" in p
            assert "start_kernel" in p

    def test_trace_call_chain_no_path(self, graph_retriever: GraphRetriever):
        """Non-existent symbols should return empty."""
        paths = graph_retriever.trace_call_chain("nonexistent_func", "start_kernel")
        assert paths == []

    def test_find_dependencies_downstream(self, graph_retriever: GraphRetriever):
        """start_kernel should have downstream dependencies (functions it calls)."""
        deps = graph_retriever.find_dependencies("start_kernel", direction="downstream", max_hops=2)
        # start_kernel calls setup_arch, mm_init, sched_init, etc.
        assert len(deps) > 0

    def test_find_dependencies_upstream(self, graph_retriever: GraphRetriever):
        """mm_init should have upstream callers (start_kernel)."""
        deps = graph_retriever.find_dependencies("mm_init", direction="upstream", max_hops=2)
        assert len(deps) > 0

    def test_find_dependencies_nonexistent(self, graph_retriever: GraphRetriever):
        deps = graph_retriever.find_dependencies("does_not_exist")
        assert deps == []
