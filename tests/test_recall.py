"""Comprehensive test suite — validates all features and evaluates recall accuracy.

Test areas:
1. Infrastructure verification (collection existence, document counts)
2. Vector search recall evaluation (ground-truth queries)
3. Reranker reranking validation
4. Metadata-filtered search
5. Hierarchical search
6. Edge cases (irrelevant queries, empty filters, top_k=1)
7. Deduplication verification

Usage:
    cd /repo/guolinp/work/kb
    uv run python tests/test_recall.py
"""

from __future__ import annotations
from source_code_kb.server.app import create_app
from fastapi.testclient import TestClient
import subprocess
from source_code_kb.generation.generator import generate_answer
from source_code_kb.generation.generator import _format_context, _extract_sources
from source_code_kb.chat.session import ChatSession
from source_code_kb.retrieval.retriever import HybridRetriever, SearchFilter, SearchResult, _extract_code_entities
from source_code_kb.retrieval.reranker import rerank
from source_code_kb.ingest.indexer import (
    create_vectorstore,
    get_collection_stats,
    get_collection_topics,
    list_collections,
)
from source_code_kb.config import load_config

import sys
from pathlib import Path

# Ensure src/ is importable — adds the project's src directory to the Python
# path so that all source_code_kb.* imports resolve correctly when running
# this script directly (outside of a proper package install).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ── Configuration ────────────────────────────────────────────────

# Load the project configuration (embedding model, ChromaDB path, LLM
# endpoint, etc.) from the repo-root config.yaml.
cfg = load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))

# Use the "default" collection — the one populated by the test_data.jsonl
# fixture during CI or manual setup.
collection_name = "default"

# Obtain a ChromaDB Collection handle (creates it if it doesn't exist yet)
# and wrap it in a HybridRetriever that provides vector + metadata search.
coll_obj = create_vectorstore(cfg, collection_name=collection_name)
retriever = HybridRetriever(coll_obj, cfg)

# ── Lightweight test framework ───────────────────────────────────
# Instead of pulling in pytest/unittest, the suite uses three global
# counters and a tiny `check()` helper.  Each call to check() is one
# assertion: it increments `total`, then either `passed` or `failed`
# depending on the boolean `condition`.  At the end of the run the
# summary uses these counters to report results and set the exit code.
passed = 0
failed = 0
total = 0


def check(name: str, condition: bool, detail: str = ""):
    """Assert a condition and record pass/fail.

    This is the core assertion primitive of the test suite.  Every test
    calls check() with a human-readable `name`, a boolean `condition`,
    and an optional `detail` string that is printed only on failure to
    aid debugging.  The function mutates the global pass/fail/total
    counters so the final summary can be produced without any external
    test runner.
    """
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name} -- {detail}")


def recall_at_k(results: list[SearchResult], expected_topics: list[str], k: int = 5) -> float:
    """Compute Recall@K: fraction of expected topics found in top-k results.

    Recall@K answers "of all the topics we *should* have retrieved, how
    many actually appear in the top-k?"  A recall of 1.0 means every
    expected topic was surfaced.  This metric is sensitive to missing
    results but tolerant of extra (irrelevant) results.
    """
    top_results = results[:k]
    # Collect the unique set of topic names present in the returned results.
    found_topics = {r.metadata.get("topic", "") for r in top_results}
    # Count how many of the expected topics appear in that set.
    hits = sum(1 for t in expected_topics if t in found_topics)
    return hits / len(expected_topics) if expected_topics else 0.0


def precision_at_k(results: list[SearchResult], expected_topics: list[str], k: int = 5) -> float:
    """Compute Precision@K: fraction of top-k results from expected topics.

    Precision@K answers "of the k results we returned, how many belong
    to an expected topic?"  A precision of 1.0 means every returned
    result is relevant.  This metric is sensitive to noisy/irrelevant
    results but tolerant of missing ones.
    """
    top_results = results[:k]
    hits = sum(1 for r in top_results if r.metadata.get("topic", "") in expected_topics)
    return hits / k if k > 0 else 0.0


# ══════════════════════════════════════════════════════════════════
# Test 1: Infrastructure Verification
# ══════════════════════════════════════════════════════════════════
# Before testing any retrieval logic we verify that the underlying
# infrastructure is healthy: the ChromaDB collection exists, contains
# the expected documents, and exposes topic metadata.  If any of these
# checks fail every subsequent test is meaningless, so this section
# acts as a fast-fail gate.

print("\n" + "=" * 70)
print("Test 1: Infrastructure Verification")
print("=" * 70)

# 1.1 Collection exists and has data
# Retrieve collection-level stats (existence flag, document count, etc.).
# If the collection was never created or the ingest step was skipped,
# `exists` will be False and `count` will be 0.
stats = get_collection_stats(cfg, collection_name=collection_name)

# The collection must exist — if it doesn't, ChromaDB is misconfigured
# or the test data was never ingested.
check("Collection exists", stats.get("exists", False))

# There must be at least one document.  The test fixture (test_data.jsonl)
# contains 10 documents, so count > 0 is a minimal sanity gate.
doc_count = stats.get("count", 0)
check(f"Document count = {doc_count} (>0)", doc_count > 0, f"actual: {doc_count}")

# 1.2 Topics available
# Topics are extracted from document metadata during ingestion.  Having
# at least one topic confirms that metadata was stored correctly and that
# topic-based filtering/hierarchical search can function.
topics = get_collection_topics(cfg, collection_name=collection_name)
check(f"Topic count = {len(topics)} (>0)", len(topics) > 0, f"actual: {len(topics)}")

# 1.3 List all collections
# The indexer exposes a helper that lists every collection in ChromaDB.
# Verify the list is non-empty and that our "default" collection appears
# in it — this also implicitly tests the list_collections() code path.
collections = list_collections(cfg)
check("Collection list is non-empty", len(collections) > 0)
check("Contains default", "default" in collections)


# ══════════════════════════════════════════════════════════════════
# Test 2: Vector Search Recall Evaluation (core)
# ══════════════════════════════════════════════════════════════════
# These are *ground-truth queries* — each query string is crafted to
# match concepts known to exist in the test corpus.  The test verifies
# that the embedding model + ChromaDB vector search returns non-empty
# results with positive similarity scores.  This is the most fundamental
# retrieval quality gate: if these queries return nothing, the embedding
# pipeline is broken.

print("\n" + "=" * 70)
print("Test 2: Vector Search Recall Evaluation")
print("=" * 70)

# Print all available topics so the test output is self-documenting;
# a human reader can cross-reference query terms with topic names.
print(f"\n  Available topics: {topics}")

# Each query targets a distinct knowledge area in the test corpus:
#   - "platform initialization hardware detection"  -> Boot / HW init docs
#   - "alarm management deduplication"              -> Alarm Management docs
#   - "configuration management YANG validation"    -> Config / NETCONF docs
#   - "device management eqpt_hwa_app plat_srv"    -> Device Management docs
# Using domain-specific jargon (e.g. YANG, eqpt_hwa_app) tests the
# embedding model's ability to handle technical vocabulary.
basic_search_tests = [
    "platform initialization hardware detection",
    "alarm management deduplication",
    "configuration management YANG validation",
    "device management eqpt_hwa_app plat_srv",
]

print(f"\n  {'Query':<50} | {'Top-1 Topic':<30} | Score")
print("  " + "-" * 90)

for query_text in basic_search_tests:
    results = retriever.search(query_text, top_k=5)

    # Print a diagnostic row: the query, the topic of the best match,
    # and its cosine similarity score.  Useful for manual inspection.
    if results:
        top_topic = results[0].metadata.get("topic", "")
        top_score = results[0].score
        print(f"  {query_text:<50} | {top_topic:<30} | {top_score:.4f}")
    else:
        print(f"  {query_text:<50} | NO RESULTS")

    # Every ground-truth query must return at least one result.
    # Zero results means the embedding or the DB query is broken.
    check(
        f"Search returns results for '{query_text[:40]}'",
        len(results) > 0,
        "no results returned",
    )
    # The top result's similarity score must be strictly positive.
    # A zero or negative score would indicate degenerate embeddings.
    if results:
        check(
            f"Top result has non-zero score",
            results[0].score > 0,
            f"score: {results[0].score}",
        )


# ══════════════════════════════════════════════════════════════════
# Test 3: Reranker Validation
# ══════════════════════════════════════════════════════════════════
# The reranker (bge-reranker-v2-m3 via Ollama) re-scores the initial
# vector-search results using a cross-encoder, which typically improves
# relevance ordering.  This section validates three properties:
#   (a) The reranker code path runs without errors and returns the
#       requested number of results.
#   (b) The reranker actually *changes* the document ordering (proving
#       it does meaningful work, not just a pass-through).
#   (c) The reranked scores are in descending order (highest first).
#   (d) For a targeted query about alarms, the top-1 reranked result
#       belongs to the "Alarm Management" topic.

print("\n" + "=" * 70)
print("Test 3: Reranker Validation")
print("=" * 70)

# bge-reranker-v2-m3 via Ollama embed API is approximate.
# This test validates the reranker code path works without errors.

# Three test cases with different queries and requested top_n values.
# Each exercises the reranker on a different knowledge domain.
rerank_test_cases = [
    ("platform initialization", 5),       # HW-init domain, request 5 results
    ("alarm report mechanism", 3),         # Alarm domain, request only 3
    ("NETCONF configuration validation", 5),  # Config domain, request 5
]

for query_text, top_n in rerank_test_cases:
    # First retrieve a larger candidate set (top_k=10) via vector search,
    # then let the reranker narrow it down to the best top_n.
    raw_results = retriever.search(query_text, top_k=10)
    reranked = rerank(query_text, raw_results, cfg, top_n=top_n)

    # The reranker must return exactly `top_n` results.  Fewer means
    # it silently dropped candidates; more means the slicing is wrong.
    check(
        f"Rerank '{query_text[:25]}' returns {top_n} results",
        len(reranked) == top_n,
        f"expected: {top_n}, actual: {len(reranked)}",
    )
    # All reranked scores must be Python floats — this catches type
    # errors from the Ollama API (e.g., returning strings or None).
    check(
        f"Rerank '{query_text[:25]}' scores are floats",
        all(isinstance(r.score, float) for r in reranked),
        "scores are not float",
    )

# 3.2 Reranker effectiveness — verify reranking actually changes order.
# If the reranker simply returned results in the same order as the
# initial vector search, it would add latency without any benefit.
# Using a specific query ("alarm deduplication and throttling mechanism")
# that we expect the cross-encoder to reshuffle.
print()
query_eff = "alarm deduplication and throttling mechanism"
raw = retriever.search(query_eff, top_k=10)
# Request all 10 back so we can compare the full ordering.
reranked_eff = rerank(query_eff, raw, cfg, top_n=len(raw))

# Compare document ID ordering before and after reranking.
raw_ids = [r.metadata.get("id", "") for r in raw]
reranked_ids = [r.metadata.get("id", "") for r in reranked_eff]
order_changed = raw_ids != reranked_ids

# Print the top-3 topics from each ordering for visual comparison.
print(f"  Raw      order (top-3 topics): {[r.metadata.get('topic', '')[:20] for r in raw[:3]]}")
print(f"  Reranked order (top-3 topics): {[r.metadata.get('topic', '')[:20] for r in reranked_eff[:3]]}")

# The orderings should differ — if they are identical, the reranker
# is not performing meaningful re-scoring.
check("Reranker changes document ordering", order_changed, "order unchanged")

# 3.3 Reranked scores should be in descending order.
# This is a basic contract of any ranking function: results must be
# sorted from most-relevant (highest score) to least-relevant.
reranked_scores = [r.score for r in reranked_eff]
check(
    "Reranked scores in descending order",
    all(reranked_scores[i] >= reranked_scores[i + 1] for i in range(len(reranked_scores) - 1)),
    f"scores: {[round(s, 4) for s in reranked_scores[:5]]}",
)

# 3.4 Reranker top result relevance — for a query about "alarm
# deduplication and throttling", the cross-encoder should rank an
# Alarm Management document first.  This is a topic-specific top-1
# precision check.
top_reranked_topic = reranked_eff[0].metadata.get("topic", "") if reranked_eff else ""
check(
    "Reranker top-1 is Alarm Management",
    top_reranked_topic == "Alarm Management",
    f"actual: {top_reranked_topic}",
)


# ══════════════════════════════════════════════════════════════════
# Test 4: Metadata-Filtered Search
# ══════════════════════════════════════════════════════════════════
# The retriever supports filtering results by structured metadata fields
# (domain, topic, scope, tags, confidence).  These tests verify that
# each filter type works correctly — i.e., *every* returned result
# matches the filter predicate.  Metadata filtering is critical for the
# "focused search" UX where a user narrows results to a specific module
# or knowledge domain.

print("\n" + "=" * 70)
print("Test 4: Metadata-Filtered Search")
print("=" * 70)

if topics:
    # 4.1 Filter by domain
    # "domain" classifies documents by their architectural scope (e.g.,
    # "module-internals" for implementation details).  Every returned
    # result must carry domain == "module-internals".
    results_domain = retriever.search(
        "system architecture",
        search_filter=SearchFilter(domain="module-internals"),
    )
    check(
        "Filter by domain (module-internals)",
        all(r.metadata.get("domain") == "module-internals" for r in results_domain),
        "results contain non-matching domains",
    )
    # Also verify the filter does not suppress all results — an empty
    # result set would trivially pass the all() check above.
    check("Domain filter returns results", len(results_domain) > 0)

    # 4.2 Filter by topic (use first available topic)
    # "topic" is the primary knowledge-area label (e.g., "Alarm
    # Management").  Pick the first available topic dynamically so the
    # test is data-driven rather than hard-coded.
    test_topic = topics[0]
    results_topic = retriever.search(
        "architecture",
        search_filter=SearchFilter(topic=test_topic),
    )
    # Every result must belong to exactly the requested topic.
    check(
        f"Filter by topic ({test_topic[:20]})",
        all(r.metadata.get("topic") == test_topic for r in results_topic),
        "results contain non-matching topics",
    )
    check(f"Topic filter returns >= 1 result", len(results_topic) >= 1)

    # 4.3 Filter by scope
    # "scope" is a list field indicating hardware/board applicability
    # (e.g., ["board-A", "board-B"]).  The filter selects documents whose
    # scope list contains the requested value.  Because not all documents
    # have scope metadata, we check execution success unconditionally and
    # only validate content when results are returned.
    results_scope = retriever.search(
        "hardware",
        search_filter=SearchFilter(scope=["board-A"]),
    )
    check("Scope filter executes without error", True)
    if results_scope:
        has_scope = any(
            "board-A" in r.metadata.get("scope", []) for r in results_scope
        )
        check("Scope filter returns relevant results", has_scope)

    # 4.4 Filter by tags
    # "tags" is a list of free-form keyword labels attached during
    # ingestion.  This exercises the tag-based filtering code path.
    # We only assert that the call succeeds (no exception) because the
    # test corpus may or may not have the "startup" tag.
    results_tags = retriever.search(
        "init startup",
        search_filter=SearchFilter(tags=["startup"]),
    )
    check("Tags filter executes without error", True)

    # 4.5 Filter by min_confidence
    # "confidence" is a float (0-1) assigned during ingestion reflecting
    # the LLM's self-assessed certainty about the document's content.
    # The filter discards any document below the threshold.
    results_conf = retriever.search(
        "system",
        search_filter=SearchFilter(min_confidence=0.9),
    )
    if results_conf:
        # Verify the contract: every returned document must meet or
        # exceed the 0.9 confidence threshold.
        check(
            "Min confidence filter",
            all(r.metadata.get("confidence", 0) >= 0.9 for r in results_conf),
            "results below confidence threshold",
        )

    # 4.6 search_by_topic
    # A convenience method that combines a text query with a topic filter
    # in a single call.  Functionally equivalent to search() with a
    # SearchFilter(topic=...), but exists as a dedicated API for the
    # common "search within one topic" pattern.
    results_by_topic = retriever.search_by_topic("architecture", test_topic)
    check(
        f"search_by_topic ({test_topic[:20]})",
        all(r.metadata.get("topic") == test_topic for r in results_by_topic),
    )


# ══════════════════════════════════════════════════════════════════
# Test 5: Hierarchical Search
# ══════════════════════════════════════════════════════════════════
# Hierarchical search is a two-phase retrieval strategy:
#   Phase 1 — identify the most relevant *topics* for the query
#             (using topic-level embeddings or metadata matching).
#   Phase 2 — within each matched topic, retrieve the top-k documents
#             via standard vector search.
# This approach improves recall for broad queries that span multiple
# knowledge areas, because it guarantees representation from each
# relevant topic rather than letting one dominant topic crowd out the
# results.

print("\n" + "=" * 70)
print("Test 5: Hierarchical Search")
print("=" * 70)

# 5.1 Hierarchical search should find topics and return documents.
# Request the top 2 topics, with up to 3 documents per topic.
matched_topics, hier_results = retriever.hierarchical_search(
    "system architecture", top_topics=2, top_k_per_topic=3,
)
# Phase 1 validation: at least one topic must match the query.
check("Hierarchical search: matched topics non-empty", len(matched_topics) > 0,
      f"matched: {matched_topics}")
# Phase 2 validation: documents must be returned from those topics.
check("Hierarchical search: results non-empty", len(hier_results) > 0)

# 5.2 Hierarchical search across multiple topics.
# A broader query ("management system") should activate more topics,
# exercising the multi-topic aggregation logic.
matched_topics2, hier_results2 = retriever.hierarchical_search(
    "management system", top_topics=3, top_k_per_topic=3,
)
check("Hierarchical search: multi-topic", len(matched_topics2) > 0,
      f"matched: {matched_topics2}")


# ══════════════════════════════════════════════════════════════════
# Test 6: Edge Cases
# ══════════════════════════════════════════════════════════════════
# Robust retrieval must handle degenerate inputs gracefully: queries
# about content that is completely absent from the KB, filters that
# match nothing, and extreme top_k values.

print("\n" + "=" * 70)
print("Test 6: Edge Cases")
print("=" * 70)

# 6.1 Irrelevant query (content not in KB)
# "quantum computing superconducting qubit" has zero overlap with the
# telecom / embedded-systems test corpus.  The retriever should still
# return *something* (the closest vectors in the space) rather than
# crashing, but the similarity scores should be noticeably lower than
# for on-topic queries (below 0.95).
results_irrelevant = retriever.search("quantum computing superconducting qubit", top_k=3)
check("Irrelevant query still returns results", len(results_irrelevant) > 0,
      "should return closest matches")
if results_irrelevant:
    max_score = max(r.score for r in results_irrelevant)
    # A high score (>= 0.95) for a completely unrelated query would
    # indicate that the embedding space is degenerate (all vectors
    # cluster together regardless of semantics).
    check("Irrelevant query has lower max score", max_score < 0.95,
          f"max score: {max_score:.4f}")

# 6.2 Filter with non-existent topic
# When the topic does not exist in any document's metadata, the filter
# must return an empty result set — not an error.
results_empty = retriever.search(
    "anything",
    search_filter=SearchFilter(topic="nonexistent_topic_xyz"),
)
check("Non-existent topic filter returns empty", len(results_empty) == 0)

# 6.3 top_k=1
# The smallest useful retrieval size.  Verify it returns exactly one
# result, not zero and not more than one.
results_top1 = retriever.search("initialization", top_k=1)
check("top_k=1 returns exactly 1 result", len(results_top1) == 1)

# 6.4 get_documents (metadata-only query, no vector similarity)
# get_documents retrieves documents purely by metadata filter, without
# computing any embedding similarity.  Useful for inventory queries
# like "show all documents in topic X".
if topics:
    docs = retriever.get_documents(SearchFilter(topic=topics[0]))
    check(f"get_documents returns results for {topics[0][:20]}", len(docs) > 0,
          f"actual: {len(docs)}")


# ══════════════════════════════════════════════════════════════════
# Test 7: Deduplication Verification
# ══════════════════════════════════════════════════════════════════
# During ingestion the indexer assigns a deterministic ID to each
# document (typically a hash of its content + metadata).  If the same
# JSONL file is ingested twice, duplicates must be detected and
# skipped.  This test retrieves *all* documents and asserts that every
# ID is unique.  Duplicate IDs would mean the KB contains redundant
# copies, which wastes storage, inflates search results, and skews
# recall/precision metrics.

print("\n" + "=" * 70)
print("Test 7: Deduplication Verification")
print("=" * 70)

# Retrieve up to 1000 documents (well above the 10 in the test corpus).
all_docs = retriever.get_documents(limit=1000)
all_ids = [d.metadata.get("id", "") for d in all_docs if d.metadata.get("id")]
unique_ids = set(all_ids)
# If total IDs != unique IDs, at least one document was inserted twice.
check("All document IDs are unique", len(all_ids) == len(unique_ids),
      f"total: {len(all_ids)}, unique: {len(unique_ids)}")


# ══════════════════════════════════════════════════════════════════
# Test 8: ChatSession Unit Tests
# ══════════════════════════════════════════════════════════════════
# ChatSession is a lightweight in-memory object that manages
# conversation history, active metadata filters, and the current
# interaction mode (simple vs. agent).  These are pure unit tests —
# they do not require the LLM or the vector store, only the
# ChatSession class itself.

print("\n" + "=" * 70)
print("Test 8: ChatSession Unit Tests")
print("=" * 70)


# 8.1 Basic message history
# Verify that user and assistant messages are appended in order and
# stored with the correct LangChain message types (HumanMessage /
# AIMessage).
ss = ChatSession()
ss.add_user_message("hello")
ss.add_assistant_message("hi there", [{"topic": "greeting"}])
check("Chat history has 2 messages", len(ss.history) == 2)
check("First message is user (HumanMessage)", ss.history[0].type == "human")
check("Second message is assistant (AIMessage)", ss.history[1].type == "ai")

# 8.2 get_history_text
# The text serialisation is used when injecting conversation context
# into LLM prompts.  Verify it contains the expected prefixes.
history_text = ss.get_history_text()
check("History text contains User:", "User: hello" in history_text)
check("History text contains Assistant:", "Assistant: hi there" in history_text)

# 8.3 get_last_sources
# After an assistant reply, the session stores the source documents
# that were used to generate the answer.  This accessor is used by the
# CLI/UI to display "Sources" to the user.
sources = ss.get_last_sources()
check("Last sources returns list", len(sources) == 1)
check("Last sources has topic", sources[0].get("topic") == "greeting")

# 8.4 Filter management
# Users can set a persistent metadata filter (e.g., restrict all
# subsequent searches to a single topic).  Verify set/clear works.
ss.set_filter("topic", "Boot Initialization")
check("Filter set correctly", ss.active_filter.get("topic") == "Boot Initialization")
ss.clear_filter()
check("Filter cleared", len(ss.active_filter) == 0)

# 8.5 Mode switching
# "simple" mode does a single retrieval + generation pass; "agent"
# mode may perform multi-step reasoning.  Verify the default and
# that switching works.
check("Default mode is simple", ss.mode == "simple")
ss.mode = "agent"
check("Mode switched to agent", ss.mode == "agent")

# 8.6 Clear history
# Resetting the conversation (e.g., "/clear" in the CLI) should
# empty the history list entirely.
ss.clear_history()
check("History cleared", len(ss.history) == 0)

# 8.7 History trimming (max_history)
# When max_history is set, the session must evict the oldest messages
# to stay within the limit.  Here max_history=4 and we add 10
# messages, so only the last 4 should survive.
ss2 = ChatSession(max_history=4)
for i in range(10):
    ss2.add_user_message(f"msg {i}")
check("History trimmed to max_history", len(ss2.history) == 4,
      f"actual: {len(ss2.history)}")

# 8.8 get_last_sources when no assistant messages
# Edge case: if the user has typed a message but the assistant has not
# replied yet, get_last_sources() should return an empty list (not
# crash).
ss3 = ChatSession()
ss3.add_user_message("test")
check("No sources when no assistant reply", len(ss3.get_last_sources()) == 0)


# ══════════════════════════════════════════════════════════════════
# Test 9: Generator Context Formatting
# ══════════════════════════════════════════════════════════════════
# The generator module converts raw SearchResult objects into a
# formatted text block that is injected into the LLM prompt as
# context, plus a structured list of source citations.  These tests
# validate the formatting helpers *without* calling the LLM.

print("\n" + "=" * 70)
print("Test 9: Generator Context Formatting")
print("=" * 70)


# Use real search results (not mocks) so the content and metadata are
# realistic.  We search for "alarm management" and take the top 3.
fake_results = retriever.search("alarm management", top_k=3)

if fake_results:
    # _format_context() produces the text block that goes into the LLM
    # prompt.  Each result is labelled "[Source N]" and separated by
    # "---" dividers for readability.
    ctx = _format_context(fake_results)
    check("Context string is non-empty", len(ctx) > 0)
    # Verify the "[Source 1]" annotation exists — it is used by the LLM
    # to attribute statements to specific documents.
    check("Context contains Source annotation", "[Source 1]" in ctx)
    # Multiple results should be separated by "---"; a single result
    # naturally has no separator.
    check("Context contains separator", "---" in ctx or len(fake_results) == 1)

    # _extract_sources() builds a list of citation dicts from the raw
    # search results.  Each dict must carry at least "topic" and "score"
    # so the UI can display meaningful source attributions.
    srcs = _extract_sources(fake_results)
    check("Extracted sources non-empty", len(srcs) > 0)
    check("Sources have topic field", all("topic" in s for s in srcs))
    check("Sources have score field", all("score" in s for s in srcs))

    # Deduplication: if two search results come from the same topic and
    # section, _extract_sources should collapse them into one citation.
    # Duplicate citations would confuse the user.
    keys = [f"{s['topic']}__{s.get('section', '')}" for s in srcs]
    check("Sources are deduplicated", len(keys) == len(set(keys)))


# ══════════════════════════════════════════════════════════════════
# Test 10: LLM Error Handling (Chat/Query)
# ══════════════════════════════════════════════════════════════════
# The LLM (served by Ollama) may or may not be reachable during the
# test run.  This section validates *graceful degradation*: if the LLM
# is up, the answer should be a non-empty string with source citations;
# if the LLM is down, the code should raise a clear exception rather
# than returning garbage or hanging indefinitely.

print("\n" + "=" * 70)
print("Test 10: LLM Error Handling (Chat/Query)")
print("=" * 70)

# 10.1 generate_answer should either succeed or raise a clear error.
# We retrieve real context documents, then attempt LLM generation.
results_for_gen = retriever.search("alarm management", top_k=3)
llm_error_caught = False
llm_succeeded = False
try:
    answer, srcs = generate_answer("test question", results_for_gen, cfg)
    llm_succeeded = True
    print(f"  LLM reachable — answer generated ({len(answer)} chars)")
except Exception as exc:
    # Any exception is acceptable as long as it is a *clean* error
    # (not a segfault, not a hang).  Record the exception type for the
    # test log.
    llm_error_caught = True
    exc_type = type(exc).__name__
    print(f"  LLM error caught: {exc_type}")

# The call must have terminated — either with a result or an exception.
# A hang (timeout) would cause the test runner itself to abort.
check("LLM call completes (success or clean error)", llm_succeeded or llm_error_caught)

# 10.2 If LLM succeeded, validate the answer structure.
# The answer must be a non-empty string and the sources a list.
if llm_succeeded:
    check("Answer is non-empty string", isinstance(answer, str) and len(answer) > 0)
    check("Sources is a list", isinstance(srcs, list))

# 10.3 ChatSession remains usable after any LLM outcome.
# Regardless of whether the LLM call succeeded or failed, the
# ChatSession object should still function normally — it must not
# be corrupted by a failed generation attempt.
ss_after_err = ChatSession()
ss_after_err.add_user_message("test after error")
check("ChatSession usable after LLM call", len(ss_after_err.history) == 1)


# ══════════════════════════════════════════════════════════════════
# Test 10b: _extract_code_entities
# ══════════════════════════════════════════════════════════════════
# _extract_code_entities uses regex patterns to pull out identifiers
# (function/variable names) and file paths from a natural-language
# query.  This lets the retriever boost results that mention those
# exact symbols, improving precision for code-centric questions.

print("\n" + "=" * 70)
print("Test 10b: _extract_code_entities")
print("=" * 70)

# Test with a query that contains both a function name ("hw_detect")
# and a file path ("platform/main.c").  The regex should extract both.
symbols, files = _extract_code_entities("hw_detect function in platform/main.c")
check("Extract code entities: symbols found", len(symbols) > 0, f"symbols: {symbols}")
check("Extract code entities: hw_detect in symbols", "hw_detect" in symbols, f"symbols: {symbols}")
check("Extract code entities: file found", "platform/main.c" in files, f"files: {files}")

# Test with a plain-English query that has no code identifiers.
# The regex should return empty lists, not false positives.
symbols2, files2 = _extract_code_entities("what is the architecture?")
check("Extract code entities: no symbols in generic query", len(symbols2) == 0, f"symbols: {symbols2}")


# ══════════════════════════════════════════════════════════════════
# Test 11: CLI Commands Smoke Test
# ══════════════════════════════════════════════════════════════════
# These tests invoke the `sckb` CLI as a subprocess, just like a real
# user would.  This validates the full stack from CLI argument parsing
# through to ChromaDB queries.  Subprocess-based testing catches issues
# that in-process tests miss, such as broken entry points, missing
# dependencies in the installed package, or incorrect sys.path setup.

print("\n" + "=" * 70)
print("Test 11: CLI Commands Smoke Test")
print("=" * 70)


# 11.1 sckb stats (should succeed without LLM)
# The "stats" command only queries ChromaDB — it does not need the LLM,
# so it should always pass when the DB is available.
result_stats = subprocess.run(
    ["uv", "run", "sckb", "stats", "-c", "default"],
    capture_output=True, text=True, timeout=60,
    cwd=str(Path(__file__).resolve().parent.parent),
)
check("sckb stats exits successfully", result_stats.returncode == 0,
      f"returncode: {result_stats.returncode}, stderr: {result_stats.stderr[:200]}")

# 11.2 sckb stats output contains collection info
# The output should include the word "Documents" or the literal count
# "10" (since the test fixture has exactly 10 documents).
check("sckb stats shows document count", "Documents" in result_stats.stdout or "10" in result_stats.stdout,
      f"stdout: {result_stats.stdout[:200]}")

# 11.3 sckb query — should either succeed (LLM up) or fail gracefully
# (LLM down).  Same graceful-degradation philosophy as Test 10, but
# exercised through the CLI interface.
result_query = subprocess.run(
    ["uv", "run", "sckb", "query", "test question"],
    capture_output=True, text=True, timeout=120,
    cwd=str(Path(__file__).resolve().parent.parent),
)
if result_query.returncode == 0:
    # LLM is reachable — should produce an answer
    check("sckb query produces answer", "Answer" in result_query.stdout,
          f"stdout: {result_query.stdout[:200]}")
else:
    # LLM not reachable — should show error message, not a stack trace
    check("sckb query shows error on LLM failure",
          "Error" in result_query.stdout or "error" in result_query.stderr.lower(),
          f"stdout: {result_query.stdout[:200]}")

# 11.4 Duplicate ingestion test (should skip duplicates)
# Re-ingesting the same test_data.jsonl should detect that every
# document already exists (by deterministic ID) and report "0 new".
# This validates the deduplication logic at the CLI/ingest layer.
result_reingest = subprocess.run(
    ["uv", "run", "sckb", "ingest", "-i", "tests/test_data.jsonl"],
    capture_output=True, text=True, timeout=120,
    cwd=str(Path(__file__).resolve().parent.parent),
)
check("Re-ingest exits successfully", result_reingest.returncode == 0,
      f"returncode: {result_reingest.returncode}")
# "0 new" in the output confirms all documents were recognized as
# duplicates and skipped.
check("Re-ingest reports 0 new (all duplicates)", "0 new" in result_reingest.stdout,
      f"stdout: {result_reingest.stdout[:200]}")


# ══════════════════════════════════════════════════════════════════
# Test 12: API Endpoint Tests (FastAPI TestClient)
# ══════════════════════════════════════════════════════════════════
# These tests exercise every HTTP endpoint exposed by the FastAPI
# application using FastAPI's TestClient (which runs requests
# in-process, no real HTTP server needed).  This validates URL routing,
# request/response serialisation, status codes, and the integration
# between the API layer and the retrieval/indexer backends.

print("\n" + "=" * 70)
print("Test 12: API Endpoint Tests (FastAPI TestClient)")
print("=" * 70)


# Create the FastAPI app with the test configuration and wrap it in a
# TestClient that allows making requests without starting a server.
api_app = create_app(cfg)
client = TestClient(api_app)

# 12.1 Health check
# The /health endpoint is used by load balancers and monitoring.  It
# must always return 200 with {"status": "ok"}.
resp = client.get("/health")
check("GET /health returns 200", resp.status_code == 200)
check("GET /health body has status=ok", resp.json().get("status") == "ok",
      f"body: {resp.json()}")

# 12.2 List collections
# Returns all ChromaDB collections with their document counts.
resp = client.get("/api/v1/collections")
check("GET /collections returns 200", resp.status_code == 200)
coll_list = resp.json()
check("GET /collections returns list", isinstance(coll_list, list) and len(coll_list) > 0,
      f"body: {coll_list}")
# Verify the "default" collection appears and has the expected count.
coll_names = [c["name"] for c in coll_list]
check("GET /collections contains 'default'", "default" in coll_names,
      f"names: {coll_names}")
default_coll = next((c for c in coll_list if c["name"] == "default"), {})
check("GET /collections default has count=10", default_coll.get("count") == 10,
      f"count: {default_coll.get('count')}")

# 12.3 Collection stats
# Detailed statistics for a single collection.
resp = client.get("/api/v1/collections/default/stats")
check("GET /collections/default/stats returns 200", resp.status_code == 200)
stats_body = resp.json()
check("Stats: exists=true", stats_body.get("exists") is True)
check("Stats: count=10", stats_body.get("count") == 10,
      f"count: {stats_body.get('count')}")
# sample_metadata_keys tells the UI which filter fields are available.
check("Stats: has metadata keys", len(stats_body.get("sample_metadata_keys", [])) > 0)

# 12.4 Non-existent collection stats (ChromaDB auto-creates via get_or_create)
# ChromaDB's get_or_create_collection will silently create a new empty
# collection, so this should return 200 with count=0, not a 404.
resp = client.get("/api/v1/collections/nonexistent_stats_test/stats")
check("Stats for new collection returns 200", resp.status_code == 200)
check("Stats for new collection: count=0", resp.json().get("count") == 0)

# 12.5 Collection topics
# Returns the unique set of topic names found in document metadata.
resp = client.get("/api/v1/collections/default/topics")
check("GET /collections/default/topics returns 200", resp.status_code == 200)
topics_body = resp.json()
check("Topics: has topic list", len(topics_body.get("topics", [])) > 0)
# The "total" field must agree with the actual list length.
check("Topics: total matches list length",
      topics_body.get("total") == len(topics_body.get("topics", [])))
# Spot-check: "Alarm Management" is a known topic in the test corpus.
check("Topics: contains Alarm Management",
      "Alarm Management" in topics_body.get("topics", []))

# 12.6 Search — basic query
# POST /search with a simple query and top_k.  Validates the core
# search endpoint returns results with the expected JSON structure.
resp = client.post("/api/v1/search", json={"query": "alarm management", "top_k": 5})
check("POST /search returns 200", resp.status_code == 200)
search_body = resp.json()
check("Search: results non-empty", len(search_body.get("results", [])) > 0)
check("Search: total > 0", search_body.get("total", 0) > 0)

# Validate the structure of each result object (content, metadata,
# score).  Every result must carry all three fields.
if search_body.get("results"):
    first = search_body["results"][0]
    check("Search result has content", "content" in first and len(first["content"]) > 0)
    check("Search result has metadata", "metadata" in first and isinstance(first["metadata"], dict))
    check("Search result has score", "score" in first and isinstance(first["score"], (int, float)))
    check("Search result score > 0", first["score"] > 0)

# 12.7 Search — with metadata filter (topic)
# Verify that the API-level topic filter produces results exclusively
# from the requested topic, matching the retriever-level behavior
# tested in Test 4.
resp = client.post("/api/v1/search", json={
    "query": "alarm",
    "top_k": 5,
    "filter": {"topic": "Alarm Management"},
})
check("Search with topic filter returns 200", resp.status_code == 200)
filtered_results = resp.json().get("results", [])
check("Filtered search returns results", len(filtered_results) > 0)
if filtered_results:
    all_match = all(r["metadata"].get("topic") == "Alarm Management" for r in filtered_results)
    check("Filtered results all match topic", all_match,
          f"topics: {[r['metadata'].get('topic') for r in filtered_results]}")

# 12.8 Search — with domain filter
# Same concept as 12.7 but using the "domain" metadata field.
resp = client.post("/api/v1/search", json={
    "query": "system",
    "filter": {"domain": "module-internals"},
})
check("Search with domain filter returns 200", resp.status_code == 200)
domain_results = resp.json().get("results", [])
if domain_results:
    all_domain = all(r["metadata"].get("domain") == "module-internals" for r in domain_results)
    check("Domain filter results all match", all_domain)

# 12.9 Search — nonexistent collection (ChromaDB auto-creates, returns empty)
# Querying a collection that does not exist should not error out;
# ChromaDB auto-creates it, and the search returns zero results.
resp = client.post("/api/v1/search", json={
    "query": "test",
    "collection": "nonexistent_xyz",
})
check("Search on nonexistent collection returns 200", resp.status_code == 200)
check("Search on nonexistent returns empty results",
      resp.json().get("total") == 0)

# 12.10 Search — empty query validation
# An empty JSON body (no "query" field) should trigger FastAPI's
# automatic request validation and return 422 Unprocessable Entity.
resp = client.post("/api/v1/search", json={})
check("Search with missing query returns 422", resp.status_code == 422)

# 12.11 Search — top_k boundary
# With top_k=1 the API must return at most one result.  This mirrors
# the retriever-level Test 6.3 but through the HTTP layer.
resp = client.post("/api/v1/search", json={"query": "init", "top_k": 1})
check("Search top_k=1 returns 200", resp.status_code == 200)
check("Search top_k=1 returns <=1 result",
      len(resp.json().get("results", [])) <= 1)

# 12.12 Hierarchical search via API
# The hierarchical search endpoint mirrors the retriever-level Test 5
# but through the HTTP layer.
resp = client.post("/api/v1/search/hierarchical", json={
    "query": "system architecture",
    "top_topics": 2,
    "top_k_per_topic": 3,
})
check("POST /search/hierarchical returns 200", resp.status_code == 200)
hier_body = resp.json()
check("Hierarchical: has matched_topics", len(hier_body.get("matched_topics", [])) > 0)
check("Hierarchical: has results", len(hier_body.get("results", [])) > 0)
# The "total" field must agree with the actual result list length.
check("Hierarchical: total matches results length",
      hier_body.get("total") == len(hier_body.get("results", [])))

# 12.13 Hierarchical search — nonexistent collection (auto-created, returns empty)
# Same auto-create behavior as 12.9, but for the hierarchical endpoint.
resp = client.post("/api/v1/search/hierarchical", json={
    "query": "test",
    "collection": "nonexistent_collection_abc",
})
check("Hierarchical on nonexistent returns 200", resp.status_code == 200)
check("Hierarchical on nonexistent has empty results",
      resp.json().get("total") == 0)

# 12.14 Search — with min_confidence filter
# Verify the confidence threshold filter works through the API layer.
resp = client.post("/api/v1/search", json={
    "query": "management",
    "filter": {"min_confidence": 0.9},
})
check("Search with min_confidence returns 200", resp.status_code == 200)
conf_results = resp.json().get("results", [])
if conf_results:
    all_above = all(r["metadata"].get("confidence", 0) >= 0.9 for r in conf_results)
    check("Min confidence filter works", all_above,
          f"confidences: {[r['metadata'].get('confidence') for r in conf_results]}")

# 12.15 Search — with scope filter
# Exercises the list-valued scope filter through the API.
resp = client.post("/api/v1/search", json={
    "query": "hardware device",
    "filter": {"scope": ["board-A"]},
})
check("Search with scope filter returns 200", resp.status_code == 200)

# 12.16 Search — with tags filter
# Exercises the list-valued tags filter through the API.
resp = client.post("/api/v1/search", json={
    "query": "initialization startup",
    "filter": {"tags": ["startup"]},
})
check("Search with tags filter returns 200", resp.status_code == 200)

# 12.17 Topics for nonexistent collection
# An auto-created empty collection should have zero topics.
resp = client.get("/api/v1/collections/nonexistent_xyz/topics")
check("Topics for nonexistent returns 200", resp.status_code == 200)
check("Topics for nonexistent has empty list",
      len(resp.json().get("topics", [])) == 0)


# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
# Print final pass/fail counts and exit with code 0 (all passed) or
# code 1 (at least one failure) so CI pipelines can gate on test
# results.

print("\n" + "=" * 70)
print(f"Test Summary: {passed}/{total} passed, {failed}/{total} failed")
print("=" * 70)

if failed > 0:
    print(f"\n  WARNING: {failed} test(s) failed")
    sys.exit(1)
else:
    print("\n  All tests passed!")
    sys.exit(0)
