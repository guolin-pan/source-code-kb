"""Tests for the knowledge-graph builder."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from source_code_kb.graph.builder import KnowledgeGraphBuilder, _parse_call_chain
from source_code_kb.graph.schema import EdgeType, NodeType

TEST_DATA = Path(__file__).parent / "test_data.jsonl"
MULTI_DOMAIN_DATA = Path(__file__).parent / "multi_domain.jsonl"


def _load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ── Unit tests ───────────────────────────────────────────────────


class TestParseCallChain:
    def test_unicode_arrow(self):
        assert _parse_call_chain("A→B→C") == ["A", "B", "C"]

    def test_ascii_arrow(self):
        assert _parse_call_chain("A->B->C") == ["A", "B", "C"]

    def test_single_symbol(self):
        assert _parse_call_chain("A") == ["A"]

    def test_empty(self):
        assert _parse_call_chain("") == []


class TestKnowledgeGraphBuilder:
    @pytest.fixture()
    def builder(self) -> KnowledgeGraphBuilder:
        return KnowledgeGraphBuilder()

    @pytest.fixture()
    def test_chunks(self) -> list[dict]:
        return _load_chunks(TEST_DATA)

    def test_build_creates_nodes_and_edges(self, builder: KnowledgeGraphBuilder, test_chunks: list[dict]):
        graph = builder.build_from_chunks(test_chunks)
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_chunk_nodes_created(self, builder: KnowledgeGraphBuilder, test_chunks: list[dict]):
        graph = builder.build_from_chunks(test_chunks)
        chunk_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == NodeType.CHUNK.value]
        # Every chunk with an id should produce a chunk node.
        expected = sum(1 for c in test_chunks if c.get("id"))
        assert len(chunk_nodes) == expected

    def test_symbol_nodes_from_call_chains(self, builder: KnowledgeGraphBuilder, test_chunks: list[dict]):
        """call_chains "A→B→C" should create symbol nodes for A, B, and C."""
        graph = builder.build_from_chunks(test_chunks)
        sym_nodes = {n for n, d in graph.nodes(data=True) if d.get("type") == NodeType.SYMBOL.value}
        # All test data contains start_kernel in call_chains or symbols.
        assert f"{NodeType.SYMBOL.value}:start_kernel" in sym_nodes

    def test_calls_edges_from_call_chain(self, builder: KnowledgeGraphBuilder):
        """A single call_chain "A→B→C" should produce CALLS edges A→B and B→C."""
        chunk = {
            "id": "test:chain:1",
            "call_chains": ["funcA→funcB→funcC"],
            "symbols": [],
            "files": [],
            "component": "test-comp",
            "source": "test",
            "confidence": 0.9,
        }
        graph = builder.build_from_chunks([chunk])
        a = f"{NodeType.SYMBOL.value}:funcA"
        b = f"{NodeType.SYMBOL.value}:funcB"
        c = f"{NodeType.SYMBOL.value}:funcC"
        assert graph.has_edge(a, b)
        assert graph.has_edge(b, c)
        assert graph[a][b]["type"] == EdgeType.CALLS.value

    def test_api_exports_creates_edge(self, builder: KnowledgeGraphBuilder):
        chunk = {
            "id": "test:api:1",
            "symbols": [],
            "files": [],
            "call_chains": [],
            "api_exports": ["my_api_func"],
            "component": "my-comp",
            "source": "test",
            "confidence": 0.9,
        }
        graph = builder.build_from_chunks([chunk])
        comp_node = f"{NodeType.COMPONENT.value}:my-comp"
        sym_node = f"{NodeType.SYMBOL.value}:my_api_func"
        assert graph.has_edge(comp_node, sym_node)
        assert graph[comp_node][sym_node]["type"] == EdgeType.EXPORTS_API.value

    def test_api_imports_creates_edge(self, builder: KnowledgeGraphBuilder):
        chunk = {
            "id": "test:api:2",
            "symbols": [],
            "files": [],
            "call_chains": [],
            "api_imports": ["external_func"],
            "component": "my-comp",
            "source": "test",
            "confidence": 0.9,
        }
        graph = builder.build_from_chunks([chunk])
        comp_node = f"{NodeType.COMPONENT.value}:my-comp"
        sym_node = f"{NodeType.SYMBOL.value}:external_func"
        assert graph.has_edge(comp_node, sym_node)
        assert graph[comp_node][sym_node]["type"] == EdgeType.IMPORTS_API.value

    def test_ipc_mechanism_creates_node_and_edge(self, builder: KnowledgeGraphBuilder):
        chunk = {
            "id": "test:ipc:1",
            "symbols": [],
            "files": [],
            "call_chains": [],
            "ipc_mechanism": ["devicetree"],
            "component": "driver-model",
            "source": "test",
            "confidence": 0.9,
        }
        graph = builder.build_from_chunks([chunk])
        ipc_node = f"{NodeType.IPC_CHANNEL.value}:devicetree"
        comp_node = f"{NodeType.COMPONENT.value}:driver-model"
        assert ipc_node in graph
        assert graph.has_edge(comp_node, ipc_node)

    def test_shared_data_creates_node_and_edge(self, builder: KnowledgeGraphBuilder):
        chunk = {
            "id": "test:data:1",
            "symbols": [],
            "files": [],
            "call_chains": [],
            "shared_data": ["buddy_pool"],
            "component": "mm",
            "source": "test",
            "confidence": 0.9,
        }
        graph = builder.build_from_chunks([chunk])
        ds_node = f"{NodeType.DATA_STRUCTURE.value}:buddy_pool"
        comp_node = f"{NodeType.COMPONENT.value}:mm"
        assert ds_node in graph
        assert graph.has_edge(comp_node, ds_node)
        assert graph[comp_node][ds_node]["type"] == EdgeType.SHARES_DATA.value

    def test_file_nodes_and_belongs_to(self, builder: KnowledgeGraphBuilder):
        chunk = {
            "id": "test:file:1",
            "symbols": [],
            "files": ["init/main.c"],
            "call_chains": [],
            "component": "kernel-core",
            "source": "test",
            "confidence": 0.9,
        }
        graph = builder.build_from_chunks([chunk])
        file_node = f"{NodeType.FILE.value}:init/main.c"
        comp_node = f"{NodeType.COMPONENT.value}:kernel-core"
        assert file_node in graph
        assert graph.has_edge(file_node, comp_node)

    def test_incremental_build(self, builder: KnowledgeGraphBuilder):
        """Calling build_from_chunks twice should accumulate, not reset."""
        chunk1 = {
            "id": "test:inc:1",
            "symbols": ["funcX"],
            "files": [],
            "call_chains": [],
            "source": "test",
            "confidence": 0.9,
        }
        chunk2 = {
            "id": "test:inc:2",
            "symbols": ["funcY"],
            "files": [],
            "call_chains": [],
            "source": "test",
            "confidence": 0.9,
        }
        builder.build_from_chunks([chunk1])
        n1 = builder.get_graph().number_of_nodes()
        builder.build_from_chunks([chunk2])
        n2 = builder.get_graph().number_of_nodes()
        assert n2 > n1

    def test_clear(self, builder: KnowledgeGraphBuilder):
        chunk = {
            "id": "test:clear:1",
            "symbols": ["funcZ"],
            "files": [],
            "call_chains": [],
            "source": "test",
            "confidence": 0.9,
        }
        builder.build_from_chunks([chunk])
        builder.clear()
        assert builder.get_graph().number_of_nodes() == 0

    @pytest.mark.skipif(not MULTI_DOMAIN_DATA.exists(), reason="multi_domain.jsonl not available")
    def test_multi_domain_data(self, builder: KnowledgeGraphBuilder):
        chunks = _load_chunks(MULTI_DOMAIN_DATA)
        graph = builder.build_from_chunks(chunks)
        assert graph.number_of_nodes() > 20
        assert graph.number_of_edges() > 20
