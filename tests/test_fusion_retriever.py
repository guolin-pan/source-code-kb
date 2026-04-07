"""Tests for the HybridFusionRetriever (RRF merge logic)."""

from __future__ import annotations

import pytest

from source_code_kb.retrieval.fusion import HybridFusionRetriever
from source_code_kb.retrieval.retriever import SearchResult


class _FakeRetriever:
    """Minimal fake that duck-types as HybridRetriever / GraphRetriever."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def search(self, query: str, **kwargs) -> list[SearchResult]:
        return list(self._results)


class _FakeConfig:
    """Minimal fake for AppConfig with graph and retrieval sections."""

    class _Graph:
        enabled = True
        fusion_alpha = 0.6
        rrf_k = 60
        max_hops = 2

    class _Retrieval:
        top_k = 5

    graph = _Graph()
    retrieval = _Retrieval()


def _result(doc_id: str, score: float, content: str = "") -> SearchResult:
    return SearchResult(
        content=content or f"content of {doc_id}",
        metadata={"id": doc_id},
        score=score,
    )


class TestRRFMerge:
    @pytest.fixture()
    def config(self) -> _FakeConfig:
        return _FakeConfig()

    def test_merge_disjoint(self, config: _FakeConfig):
        """Two disjoint result lists produce a merged list with all items."""
        vec = _FakeRetriever([_result("a", 0.9), _result("b", 0.8)])
        graph = _FakeRetriever([_result("c", 0.7), _result("d", 0.6)])
        fusion = HybridFusionRetriever(vec, graph, config)

        results = fusion.search("test query", top_k=10)
        ids = [r.metadata["id"] for r in results]
        assert set(ids) == {"a", "b", "c", "d"}

    def test_merge_overlapping(self, config: _FakeConfig):
        """Overlapping results should be deduplicated, with boosted scores."""
        vec = _FakeRetriever([_result("a", 0.9), _result("b", 0.8)])
        graph = _FakeRetriever([_result("a", 0.7), _result("c", 0.6)])
        fusion = HybridFusionRetriever(vec, graph, config)

        results = fusion.search("test query", top_k=10)
        ids = [r.metadata["id"] for r in results]
        # "a" appears in both lists → boosted score → should rank first.
        assert ids[0] == "a"
        # Total unique results.
        assert set(ids) == {"a", "b", "c"}

    def test_top_k_respected(self, config: _FakeConfig):
        """Output length should not exceed top_k."""
        vec = _FakeRetriever([_result(f"v{i}", 1.0 - i * 0.1) for i in range(5)])
        graph = _FakeRetriever([_result(f"g{i}", 1.0 - i * 0.1) for i in range(5)])
        fusion = HybridFusionRetriever(vec, graph, config)

        results = fusion.search("test query", top_k=3)
        assert len(results) <= 3

    def test_empty_graph_results(self, config: _FakeConfig):
        """When graph returns nothing, vector results pass through."""
        vec = _FakeRetriever([_result("a", 0.9), _result("b", 0.8)])
        graph = _FakeRetriever([])
        fusion = HybridFusionRetriever(vec, graph, config)

        results = fusion.search("test query", top_k=5)
        ids = [r.metadata["id"] for r in results]
        assert ids == ["a", "b"]

    def test_empty_vector_results(self, config: _FakeConfig):
        """When vector returns nothing, graph results pass through."""
        vec = _FakeRetriever([])
        graph = _FakeRetriever([_result("c", 0.7)])
        fusion = HybridFusionRetriever(vec, graph, config)

        results = fusion.search("test query", top_k=5)
        ids = [r.metadata["id"] for r in results]
        assert ids == ["c"]

    def test_both_empty(self, config: _FakeConfig):
        """When both are empty, result should be empty."""
        vec = _FakeRetriever([])
        graph = _FakeRetriever([])
        fusion = HybridFusionRetriever(vec, graph, config)
        assert fusion.search("test query", top_k=5) == []

    def test_alpha_affects_ranking(self):
        """Higher alpha should favour vector results."""
        # Config with alpha=0.9 (strong vector preference).
        cfg_vec_heavy = _FakeConfig()
        cfg_vec_heavy.graph.fusion_alpha = 0.9

        # "v1" only in vector list, "g1" only in graph list.
        vec = _FakeRetriever([_result("v1", 0.9)])
        graph = _FakeRetriever([_result("g1", 0.9)])

        fusion = HybridFusionRetriever(vec, graph, cfg_vec_heavy)
        results = fusion.search("test query", top_k=2)
        ids = [r.metadata["id"] for r in results]
        # With alpha=0.9, vector result "v1" should rank above "g1".
        assert ids[0] == "v1"

    def test_scores_are_positive(self, config: _FakeConfig):
        vec = _FakeRetriever([_result("a", 0.9)])
        graph = _FakeRetriever([_result("b", 0.7)])
        fusion = HybridFusionRetriever(vec, graph, config)
        for r in fusion.search("test", top_k=5):
            assert r.score > 0

    def test_scores_are_normalized(self, config: _FakeConfig):
        """Scores should be in [0, 1] after normalization."""
        vec = _FakeRetriever([_result("a", 0.9), _result("b", 0.8)])
        graph = _FakeRetriever([_result("a", 0.7), _result("c", 0.6)])
        fusion = HybridFusionRetriever(vec, graph, config)
        results = fusion.search("test", top_k=5)
        for r in results:
            assert 0.0 < r.score <= 1.0, f"score {r.score} out of range"

    def test_retrieval_source_both(self, config: _FakeConfig):
        """Result in both lists gets retrieval_source='vector+graph'."""
        vec = _FakeRetriever([_result("a", 0.9)])
        graph = _FakeRetriever([_result("a", 0.7)])
        fusion = HybridFusionRetriever(vec, graph, config)
        results = fusion.search("test", top_k=5)
        assert results[0].metadata["retrieval_source"] == "vector+graph"

    def test_retrieval_source_vector_only(self, config: _FakeConfig):
        """Result only in vector gets retrieval_source='vector'."""
        vec = _FakeRetriever([_result("a", 0.9)])
        graph = _FakeRetriever([])
        fusion = HybridFusionRetriever(vec, graph, config)
        results = fusion.search("test", top_k=5)
        assert results[0].metadata["retrieval_source"] == "vector"

    def test_retrieval_source_graph_only(self, config: _FakeConfig):
        """Result only in graph gets retrieval_source='graph'."""
        vec = _FakeRetriever([])
        graph = _FakeRetriever([_result("a", 0.7)])
        fusion = HybridFusionRetriever(vec, graph, config)
        results = fusion.search("test", top_k=5)
        assert results[0].metadata["retrieval_source"] == "graph"

    def test_graph_filter_applied(self, config: _FakeConfig):
        """Graph results are filtered by search_filter."""
        from source_code_kb.retrieval.retriever import SearchFilter

        vec_result = SearchResult(
            content="vec content",
            metadata={"id": "v1", "domain": "module-internals", "component": "base"},
            score=0.9,
        )
        graph_match = SearchResult(
            content="graph match",
            metadata={"id": "g1", "domain": "module-internals", "component": "base"},
            score=0.7,
        )
        graph_mismatch = SearchResult(
            content="graph mismatch",
            metadata={"id": "g2", "domain": "module-interface", "component": "services"},
            score=0.8,
        )
        vec = _FakeRetriever([vec_result])
        graph = _FakeRetriever([graph_mismatch, graph_match])
        fusion = HybridFusionRetriever(vec, graph, config)

        sf = SearchFilter(domain="module-internals", component="base")
        results = fusion.search("test", top_k=5, search_filter=sf)
        ids = [r.metadata["id"] for r in results]
        # g2 should be filtered out because domain/component don't match
        assert "g2" not in ids
        assert "v1" in ids
        assert "g1" in ids
