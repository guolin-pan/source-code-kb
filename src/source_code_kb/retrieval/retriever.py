"""Hybrid retriever — vector search + metadata filtering + symbol exact matching.

A ChromaDB-based retrieval engine:
1. Vector search: semantic search based on embedding similarity
2. Metadata filtering: ChromaDB where-clause filtering by domain/topic/section/confidence/language/component
3. Post-filtering: scope/tags filtering (ChromaDB doesn't support list types)
4. Symbol/file exact matching: extracts code entities from query and boosts matching results

Scope, tags, files, symbols, and other list fields are stored as comma-separated strings in ChromaDB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

if TYPE_CHECKING:
    import chromadb

    from source_code_kb.config import AppConfig


# ── Data structure definitions ───────────────────────────────────────────


@dataclass
class SearchFilter:
    """Structured metadata filter.

    All fields are optional; unset fields are excluded from filtering.
    """

    domain: str | None = None
    topic: str | None = None
    section: str | None = None
    scope: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    min_confidence: float | None = None
    # New code-specific filters
    files: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    language: str | None = None
    component: str | None = None


@dataclass
class SearchResult:
    """A single search result."""

    content: str
    metadata: dict[str, Any]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": round(self.score, 4),
        }


# ── Code entity extraction ─────────────────────────────────────────


# Patterns for identifying code entities in user queries.
# _FUNC_RE matches snake_case identifiers like "get_user_name" or "parse_config".
# It requires at least one underscore so plain English words ("hello") are not matched.
_FUNC_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")  # snake_case
# _CAMEL_RE matches CamelCase identifiers like "HttpClient" or "UserManager".
# It requires at least two uppercase-led segments to avoid matching single capitalized words.
_CAMEL_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")  # CamelCase
# _FILE_RE matches file paths with common source code extensions.
# Allows directory separators (/) and dots in the path prefix (e.g. "src/utils/parser.py").
_FILE_RE = re.compile(r"\b([\w/.-]+\.(?:c|h|cpp|hpp|py|go|java|rs|js|ts))\b")


def _extract_code_entities(query: str) -> tuple[list[str], list[str]]:
    """Extract likely function names and file paths from a query string.

    Returns:
        (symbols, files) — lists of extracted symbol names and file paths.
    """
    symbols: list[str] = []
    files: list[str] = []

    # Extract file paths first — these are identified by their source-code extensions
    # (e.g. ".c", ".py"). Collected before symbols so we can exclude them from the
    # symbol list below (a path like "my_module.py" also matches _FUNC_RE).
    for m in _FILE_RE.finditer(query):
        files.append(m.group(1))

    # Extract snake_case identifiers (likely function/variable names).
    for m in _FUNC_RE.finditer(query):
        name = m.group(1)
        # Min-length filter (>= 4 chars) avoids matching short, ambiguous tokens
        # like "is_a" that are more likely natural language than code identifiers.
        # Also skip anything already captured as a file path — "my_utils.py" would
        # otherwise be double-counted as both a file and a symbol.
        if len(name) >= 4 and name not in files:
            symbols.append(name)

    # CamelCase identifiers are almost always class or type names; no length filter
    # needed because the regex already requires at least two capitalized segments,
    # making false positives rare.
    for m in _CAMEL_RE.finditer(query):
        symbols.append(m.group(1))

    return symbols, files


# ── ChromaDB filter construction ──────────────────────────────────


def _build_chroma_filter(search_filter: SearchFilter | None = None) -> dict | None:
    """Convert a structured filter into a ChromaDB where clause.

    Handles domain, topic, section, and min_confidence.
    Scope and tags require post-filtering (stored as comma-separated strings).

    Args:
        search_filter: Metadata filter conditions.

    Returns:
        A ChromaDB where dict, or None when there are no conditions.
    """
    if not search_filter:
        return None

    # Only scalar-valued fields (domain, topic, section, confidence, language,
    # component) can be filtered natively by ChromaDB's where clause.  List-type
    # fields like scope and tags are stored as comma-separated strings, so
    # ChromaDB cannot perform element-wise matching on them.  Those fields are
    # handled separately in _post_filter_scope_tags after the query returns.
    conditions: list[dict] = []

    if search_filter.domain:
        conditions.append({"domain": {"$eq": search_filter.domain}})
    if search_filter.topic:
        conditions.append({"topic": {"$eq": search_filter.topic}})
    if search_filter.section:
        conditions.append({"section": {"$eq": search_filter.section}})
    if search_filter.min_confidence is not None:
        conditions.append({"confidence": {"$gte": search_filter.min_confidence}})
    if search_filter.language:
        conditions.append({"language": {"$eq": search_filter.language}})
    if search_filter.component:
        conditions.append({"component": {"$eq": search_filter.component}})

    if not conditions:
        return None
    # ChromaDB requires a bare condition dict when there is only one clause;
    # wrapping a single condition in {"$and": [...]} would still work but is
    # unnecessarily verbose.  When two or more conditions exist, they are
    # combined with $and so all must be satisfied simultaneously.
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _post_filter_scope_tags(
    results: list[SearchResult],
    search_filter: SearchFilter | None,
) -> list[SearchResult]:
    """Post-filter results by scope and tags (comma-separated string matching).

    Args:
        results: Pre-filtered search results.
        search_filter: Filter containing scope/tags to match.

    Returns:
        Filtered results where scope/tags match.
    """
    if not search_filter:
        return results
    if not search_filter.scope and not search_filter.tags:
        return results

    filtered = []
    for r in results:
        # Scope and tags are stored as comma-separated strings in ChromaDB
        # (e.g. "kernel,driver,mm") because ChromaDB does not support list-typed
        # metadata values.  Here we parse them back into Python sets so we can
        # do proper element-wise comparison.  The isinstance check handles the
        # case where _unflatten_metadata has already converted them to lists.
        result_scope = set(
            s.strip() for s in r.metadata.get("scope", "").split(",") if s.strip()
        ) if isinstance(r.metadata.get("scope"), str) else set(r.metadata.get("scope", []))

        result_tags = set(
            t.strip() for t in r.metadata.get("tags", "").split(",") if t.strip()
        ) if isinstance(r.metadata.get("tags"), str) else set(r.metadata.get("tags", []))

        # "Any overlap" matching: a result is kept if at least one of its scope
        # values appears in the requested scopes.  This is intentionally lenient
        # — requiring an exact / full match would be too restrictive because a
        # document about "kernel,driver" is still relevant when the user asks
        # for scope=["driver"].
        if search_filter.scope:
            if not result_scope.intersection(search_filter.scope):
                continue

        # Same "any overlap" logic for tags.
        if search_filter.tags:
            if not result_tags.intersection(search_filter.tags):
                continue

        filtered.append(r)

    return filtered


def _unflatten_metadata(metadata: dict, doc_id: str) -> dict:
    """Restore list fields from comma-separated strings.

    Args:
        metadata: Raw ChromaDB metadata dict.
        doc_id: Document ID (stored separately in ChromaDB).

    Returns:
        Metadata dict with scope/tags as lists.
    """
    # _unflatten_metadata reverses the flattening that was applied when ingesting
    # documents into ChromaDB.  During ingestion, Python lists were joined into
    # comma-separated strings because ChromaDB only supports scalar metadata
    # values.  Here we split them back into proper Python lists.
    m = dict(metadata)
    # ChromaDB stores document IDs separately from metadata; re-attach it so
    # downstream consumers have a single unified metadata dict.
    m["id"] = doc_id

    # Restore the two core taxonomy fields (scope, tags) from CSV strings to lists.
    scope_raw = m.get("scope", "")
    m["scope"] = [s.strip() for s in scope_raw.split(",") if s.strip()] if scope_raw else []

    tags_raw = m.get("tags", "")
    m["tags"] = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

    # Restore all code-analysis metadata list fields.  These cover source files,
    # symbol names, function call chains, API boundaries, IPC mechanisms, and
    # shared data references — every field that was originally a list during
    # knowledge-base generation.
    for key in ("files", "symbols", "call_chains", "api_exports", "api_imports",
                "ipc_mechanism", "messages_send", "messages_receive", "shared_data"):
        raw = m.get(key, "")
        # Guard against values that are already lists (e.g. if unflatten is
        # called on already-processed metadata).
        if isinstance(raw, str):
            m[key] = [v.strip() for v in raw.split(",") if v.strip()] if raw else []

    # Confidence was stored as a string in some code paths; ensure it is always
    # a float for consistent downstream comparisons and sorting.
    m["confidence"] = float(m.get("confidence", 0.0))

    return m


# ── Hybrid retriever ─────────────────────────────────────────────────


class HybridRetriever:
    """Hybrid retriever — combines vector similarity with metadata filtering."""

    def __init__(self, collection: chromadb.Collection, config: AppConfig):
        self.collection = collection
        self.config = config
        self._embed_fn = OllamaEmbeddings(
            model=config.embedding.model,
            base_url=config.embedding.base_url,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        search_filter: SearchFilter | None = None,
    ) -> list[SearchResult]:
        """Perform hybrid search: vector similarity + metadata filtering + symbol matching.

        When code entities (function names, file paths) are detected in the query,
        an additional metadata $contains query is issued and results are merged,
        with exact matches prioritized.

        Args:
            query: User query text.
            top_k: Number of results to return.
            search_filter: Metadata filter conditions.

        Returns:
            A list of SearchResult objects sorted by relevance.
        """
        k = top_k or self.config.retrieval.top_k

        # When scope or tags filtering is requested, we cannot rely on ChromaDB
        # to do it (those fields are comma-separated strings, not native lists).
        # Instead, we over-fetch by 3x and then discard non-matching results in
        # Python.  The 3x multiplier is a heuristic that balances recall against
        # the cost of embedding comparison — fetching too few would risk
        # returning fewer than k results after post-filtering.
        needs_post_filter = (
            search_filter
            and (search_filter.scope or search_filter.tags)
        )
        fetch_k = k * 3 if needs_post_filter else k

        # Embed the user's natural-language query into the same vector space as
        # the stored document chunks so we can compute cosine/L2 similarity.
        query_vector = self._embed_fn.embed_query(query)

        # Build the ChromaDB where clause from scalar metadata fields only.
        # Scope/tags are excluded here and handled in the post-filter step.
        where = _build_chroma_filter(search_filter)

        # Assemble the ChromaDB query.  "distances" is requested so we can
        # convert raw L2 distance into a 0-1 similarity score below.
        query_params: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": fetch_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        results = self.collection.query(**query_params)

        # ChromaDB returns parallel lists wrapped in an outer list (one entry
        # per query).  We always send a single query, so unwrap index [0].
        search_results: list[SearchResult] = []
        ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for i in range(len(ids)):
            metadata = _unflatten_metadata(metadatas[i], ids[i])
            # ChromaDB returns L2 (Euclidean) distances, where 0 = identical.
            # Convert to a similarity score in (0, 1] using the formula
            #   score = 1 / (1 + distance)
            # This maps distance=0 to score=1.0 and distance=inf to score->0,
            # giving an intuitive "higher is better" ranking.
            distance = distances[i]
            score = 1.0 / (1.0 + distance)
            search_results.append(SearchResult(
                content=documents[i],
                metadata=metadata,
                score=score,
            ))

        # Apply the Python-side post-filter for scope/tags that could not be
        # handled by ChromaDB natively.
        if needs_post_filter:
            search_results = _post_filter_scope_tags(search_results, search_filter)

        # --- Symbol / file exact-match boosting ---
        # In addition to semantic (vector) similarity, we look for exact code
        # entity mentions in the metadata.  If the user's query contains a
        # function name like "parse_config", we want documents that explicitly
        # reference that symbol to appear at the top, even if the embedding
        # similarity is not the highest.
        extracted_symbols, extracted_files = _extract_code_entities(query)
        # Cap the number of entities to avoid issuing too many individual
        # ChromaDB $contains queries, which are O(n) scans.  Five per category
        # is a pragmatic upper bound that covers realistic queries.
        extracted_symbols = extracted_symbols[:5]
        extracted_files = extracted_files[:5]
        if extracted_symbols or extracted_files:
            # Track IDs already present in vector results to avoid duplicates.
            seen_ids = {r.metadata.get("id") for r in search_results}
            exact_matches: list[SearchResult] = []

            # Issue $contains metadata queries across multiple metadata fields.
            # A symbol like "init_device" might appear in the "symbols" field
            # (defined there), in "call_chains" (called by other code), or in
            # "api_exports" / "api_imports" (part of an API boundary).  We check
            # all relevant fields to maximize recall.
            for entity_list, metadata_key in [
                (extracted_symbols, "symbols"),
                (extracted_files, "files"),
                (extracted_symbols, "call_chains"),
                (extracted_symbols, "api_exports"),
                (extracted_symbols, "api_imports"),
            ]:
                for entity in entity_list:
                    try:
                        meta_results = self.collection.get(
                            where={metadata_key: {"$contains": entity}},
                            include=["documents", "metadatas"],
                            limit=k,
                        )
                        meta_ids = meta_results.get("ids") or []
                        meta_docs = meta_results.get("documents") or []
                        meta_metas = meta_results.get("metadatas") or []
                        for j in range(len(meta_ids)):
                            if meta_ids[j] not in seen_ids:
                                seen_ids.add(meta_ids[j])
                                m = _unflatten_metadata(meta_metas[j], meta_ids[j])
                                # Assign score=1.0 (the maximum) to exact
                                # metadata matches.  This guarantees they rank
                                # above all vector-similarity results (whose
                                # scores are strictly < 1.0 unless distance is
                                # exactly zero, which is near-impossible for
                                # real embeddings).
                                exact_matches.append(SearchResult(
                                    content=meta_docs[j],
                                    metadata=m,
                                    score=1.0,
                                ))
                    except Exception:
                        pass  # ChromaDB $contains may not work on all field types

            # Merge strategy: prepend exact matches before vector results so
            # that symbol/file hits always appear first in the final ranking.
            # Vector results retain their original similarity-based ordering
            # as a tiebreaker for the remaining slots.
            if exact_matches:
                search_results = exact_matches + search_results

        # Truncate to the originally requested k after all merging/boosting.
        return search_results[:k]

    def search_by_topic(
        self,
        query: str,
        topic: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Search within a specific topic."""
        return self.search(
            query,
            top_k=top_k,
            search_filter=SearchFilter(topic=topic),
        )

    def hierarchical_search(
        self,
        query: str,
        top_topics: int = 3,
        top_k_per_topic: int = 5,
    ) -> tuple[list[str], list[SearchResult]]:
        """Hierarchical search: first locate relevant topics, then search within them."""
        # --- Phase 1: Topic discovery ---
        # Search only "Overview" section chunks.  These are high-level summaries
        # generated during ingestion, one per topic.  By matching the query
        # against overviews first, we identify which topics are most relevant
        # without wading through all the fine-grained detail chunks.
        # We fetch 2x the desired number of topics because duplicate topics may
        # appear (multiple overview chunks per topic) and we deduplicate below.
        overview_filter = SearchFilter(section="Overview")
        overview_results = self.search(
            query,
            top_k=top_topics * 2,
            search_filter=overview_filter,
        )

        # Deduplicate topics while preserving the relevance order returned by
        # the vector search — the first occurrence of each topic is the most
        # semantically similar to the query.
        seen_topics: set[str] = set()
        matched_topics: list[str] = []
        for r in overview_results:
            topic = r.metadata.get("topic", "")
            if topic and topic not in seen_topics:
                seen_topics.add(topic)
                matched_topics.append(topic)
                if len(matched_topics) >= top_topics:
                    break

        # Fallback: if no topics were identified (e.g. the knowledge base has
        # no Overview sections), fall back to an unscoped global search so the
        # caller still gets useful results.
        if not matched_topics:
            return [], self.search(query, top_k=top_k_per_topic)

        # --- Phase 2: Topic-specific deep search ---
        # Now that we know which topics matter, run a focused search within each
        # one.  This narrows the vector space and returns more precise, detailed
        # chunks than a global search would.
        all_results: list[SearchResult] = []
        for topic in matched_topics:
            topic_results = self.search_by_topic(query, topic, top_k=top_k_per_topic)
            all_results.extend(topic_results)

        # Re-sort the combined results across all topics by score so the caller
        # gets a single globally-ranked list.
        all_results.sort(key=lambda r: r.score, reverse=True)
        return matched_topics, all_results

    def get_documents(
        self,
        search_filter: SearchFilter | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """Retrieve documents by metadata filter (no vector similarity)."""
        # This method uses ChromaDB's .get() instead of .query().  Unlike
        # .query(), .get() does not require an embedding vector and does not
        # rank by similarity — it simply returns all documents whose metadata
        # matches the where clause (up to the limit).  This is useful for
        # enumerating or exporting chunks by metadata criteria (e.g. "give me
        # all documents for topic=X") without needing a natural-language query.
        where = _build_chroma_filter(search_filter)

        # We only need documents and metadatas; distances/embeddings are not
        # available from .get() and not needed for metadata-only retrieval.
        get_params: dict[str, Any] = {
            "include": ["documents", "metadatas"],
            "limit": limit,
        }
        if where:
            get_params["where"] = where

        data = self.collection.get(**get_params)

        docs = []
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        for i in range(len(ids)):
            meta = _unflatten_metadata(metadatas[i], ids[i])
            docs.append(Document(
                page_content=documents[i],
                metadata=meta,
            ))

        # Scope/tags post-filtering is needed here too, identical in rationale
        # to the post-filter in search(): ChromaDB cannot natively filter on
        # comma-separated list fields.
        if search_filter and (search_filter.scope or search_filter.tags):
            filtered = []
            for doc in docs:
                keep = True
                if search_filter.scope:
                    doc_scope = set(doc.metadata.get("scope", []))
                    if not doc_scope.intersection(search_filter.scope):
                        keep = False
                if search_filter.tags and keep:
                    doc_tags = set(doc.metadata.get("tags", []))
                    if not doc_tags.intersection(search_filter.tags):
                        keep = False
                if keep:
                    filtered.append(doc)
            docs = filtered

        return docs
