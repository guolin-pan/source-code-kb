"""Answer generation module — Generates RAG answers based on search results and LLM.

Core flow:
1. Format retrieved chunks as a context string (with source annotations)
2. Inject context into the prompt template
3. Call LLM to generate the final answer
4. Extract the list of cited sources

Supports single-turn Q&A, multi-turn conversation (with injected history), and streaming output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from source_code_kb.generation.prompts import RAG_PROMPT, RAG_WITH_HISTORY_PROMPT

if TYPE_CHECKING:
    from source_code_kb.config import AppConfig
    from source_code_kb.retrieval.retriever import SearchResult


def evaluate_chunks(
    question: str,
    results: list[SearchResult],
    config: AppConfig,
) -> tuple[str, list[SearchResult]]:
    """Evaluate relevance of retrieved chunks and filter out irrelevant ones (R3).

    Uses LLM to classify chunks. If "partial", performs chunk-level filtering.

    Args:
        question: User question
        results: List of retrieved chunks
        config: Application configuration

    Returns:
        Tuple of (verdict, filtered_chunks) where verdict is "relevant"/"partial"/"insufficient"
        and filtered_chunks contains only relevant chunks.
    """
    # Phase 0: Short-circuit — no chunks means nothing to evaluate.
    if not results:
        return "insufficient", []

    # Lazy import to avoid circular dependency at module level.
    from source_code_kb.generation.prompts import EVALUATE_CHUNKS_PROMPT, FILTER_CHUNKS_PROMPT

    llm = create_llm(config)

    # ── Phase 1: Overall relevance verdict ──────────────────────────
    # Ask the LLM to read a *summary* of the retrieved chunks and return a
    # single-word verdict: "relevant", "partial", or "insufficient".
    # This is part of the R3 feature (chunk-level relevance evaluation) which
    # prevents the RAG pipeline from feeding clearly irrelevant context to the
    # answer-generation step.
    #
    # Build a compact summary of chunks for the overall evaluation prompt.
    parts = []
    # Limit to the first 10 chunks to keep the evaluation prompt within a
    # reasonable token budget; the full set is still used later if needed.
    for i, r in enumerate(results[:10], 1):
        # Extract key metadata fields for the summary line.
        topic = r.metadata.get("topic", "")
        section = r.metadata.get("section", "")
        component = r.metadata.get("component", "")
        symbols = r.metadata.get("symbols", [])
        files = r.metadata.get("files", [])
        # Cap list fields to avoid blowing up the summary: 5 symbols, 3 files.
        symbols_str = ",".join(symbols[:5]) if isinstance(symbols, list) else str(symbols)
        files_str = ",".join(files[:3]) if isinstance(files, list) else str(files)
        # Truncate content to 200 chars — just enough for the LLM to judge
        # relevance without consuming excessive prompt tokens.
        snippet = r.content[:200]
        # Build a one-line summary: always include topic/section, conditionally
        # add component, symbols, and files when they are present.
        meta_parts = [f"topic={topic}", f"section={section}"]
        if component:
            meta_parts.append(f"component={component}")
        if symbols_str:
            meta_parts.append(f"symbols=[{symbols_str}]")
        if files_str:
            meta_parts.append(f"files=[{files_str}]")
        parts.append(f"[{i}] {', '.join(meta_parts)}: {snippet}")
    chunks_summary = "\n".join(parts)

    # Invoke the EVALUATE_CHUNKS_PROMPT via a LangChain LCEL chain to get
    # the overall verdict as a single word.
    chain = EVALUATE_CHUNKS_PROMPT | llm | StrOutputParser()
    verdict = chain.invoke({"question": question, "chunks_summary": chunks_summary}).strip().lower()

    # Guard against unexpected LLM output — if the verdict is not one of the
    # three expected values, default to "relevant" so we err on the side of
    # providing context rather than discarding it.
    if verdict not in ("relevant", "partial", "insufficient"):
        verdict = "relevant"

    # ── Phase 1 verdict routing ──────────────────────────────────────
    # "relevant"     -> return all chunks as-is, no filtering needed.
    # "insufficient" -> none of the chunks help, return empty list.
    # "partial"      -> fall through to Phase 2 for per-chunk filtering.

    # If relevant, return all chunks
    if verdict == "relevant":
        return verdict, results

    # If insufficient, return empty
    if verdict == "insufficient":
        return verdict, []

    # ── Phase 2: Per-chunk filtering (only reached for "partial") ───
    # When the overall verdict is "partial", some chunks are useful and some
    # are noise.  We now build a more detailed representation of every chunk
    # (not limited to 10) and ask the LLM to label each one individually.
    # The detailed representation includes a longer content snippet (300 chars)
    # and additional metadata (files, symbols) to improve classification accuracy.

    # If partial, filter chunks
    detail_parts = []
    for i, r in enumerate(results, 1):
        # Build per-chunk detail blocks with metadata header + content preview.
        topic = r.metadata.get("topic", "")
        section = r.metadata.get("section", "")
        component = r.metadata.get("component", "")
        symbols = r.metadata.get("symbols", [])
        files = r.metadata.get("files", [])
        symbols_str = ",".join(symbols[:5]) if isinstance(symbols, list) else str(symbols)
        files_str = ",".join(files[:3]) if isinstance(files, list) else str(files)
        header = f"[{i}] topic={topic}, section={section}"
        if component:
            header += f", component={component}"
        # Append file/symbol info as extra lines below the header.
        extra = []
        if files_str:
            extra.append(f"files: {files_str}")
        if symbols_str:
            extra.append(f"symbols: {symbols_str}")
        extra_line = "\n".join(extra)
        # Include a 300-char content snippet (longer than Phase 1's 200 chars)
        # to give the LLM more signal for per-chunk relevance judgement.
        if extra_line:
            detail_parts.append(f"{header}\n{extra_line}\n{r.content[:300]}")
        else:
            detail_parts.append(f"{header}\n{r.content[:300]}")
    chunks_detail = "\n\n".join(detail_parts)

    # Ask the LLM to classify each chunk individually.
    filter_chain = FILTER_CHUNKS_PROMPT | llm | StrOutputParser()
    filter_result = filter_chain.invoke({"question": question, "chunks_detail": chunks_detail})

    # Parse the filter results.  The expected format from the LLM is one line
    # per chunk: "idx:label" where idx is the 1-based chunk number and label
    # is either "relevant" or "irrelevant".  For example:
    #   1:relevant
    #   2:irrelevant
    #   3:relevant
    filtered = []
    for line in filter_result.strip().split("\n"):
        if ":" in line:
            try:
                idx_str, label = line.split(":", 1)
                # Convert 1-based LLM output to 0-based Python index.
                idx = int(idx_str.strip()) - 1
                if label.strip().lower() == "relevant" and 0 <= idx < len(results):
                    filtered.append(results[idx])
            except (ValueError, IndexError):
                # Skip malformed lines — the LLM may occasionally produce
                # unexpected formatting.
                continue

    # Safety fallback: if parsing failed or every chunk was marked irrelevant,
    # return the original unfiltered set.  This prevents the downstream answer
    # generator from receiving an empty context, which would always produce an
    # "insufficient data" response even though some chunks may actually help.
    if not filtered:
        filtered = results

    return verdict, filtered


def classify_follow_up(question: str, history: str, config: AppConfig) -> bool:
    """Use LLM to determine if the question is a follow-up on the same topic.

    Args:
        question: Current user question
        history: Recent conversation history text
        config: Application configuration

    Returns:
        True if the question is a same-topic follow-up, False if it's a new topic.
    """
    # R5 feature: Follow-up detection.
    # In multi-turn conversation, this function decides whether the new question
    # is about the *same* topic (so we can reuse the previously retrieved chunks
    # and save a retrieval round-trip) or a *new* topic (requiring a fresh
    # vector search).  Lazy import to keep module-level imports lightweight.
    from source_code_kb.generation.prompts import FOLLOW_UP_PROMPT

    llm = create_llm(config)
    chain = FOLLOW_UP_PROMPT | llm | StrOutputParser()
    verdict = chain.invoke({"question": question, "history": history}).strip().lower()

    # "same" -> True: the caller should reuse cached chunks from the previous
    # turn instead of performing a new retrieval, reducing latency and cost.
    # Any other value (including "new") -> False: trigger fresh retrieval.
    return verdict == "same"


def _format_metadata_header(i: int, meta: dict, compact: bool = False) -> str:
    """Build a structured metadata header for a context source.

    Args:
        i: 1-based source index.
        meta: Metadata dict (lists already unflattened).
        compact: If True, emit only the most valuable fields to save tokens.
    """
    # Start the header with a numbered source tag so the LLM can reference it
    # (e.g., "[Source 3]") when citing evidence in its answer.
    lines: list[str] = [f"[Source {i}]"]

    # Line 1 — structural locator fields.  These four fields (domain, topic,
    # section, source) are *always* emitted when present because they identify
    # where this chunk lives in the knowledge base hierarchy.  They are shown
    # on a single pipe-separated line for readability.
    locator = []
    for key in ("domain", "topic", "section", "source"):
        val = meta.get(key)
        if val:
            locator.append(f"{key}: {val}")
    if locator:
        lines.append(" | ".join(locator))

    # Line 2 — scalar code metadata (component, language, confidence).
    # These are shown unconditionally in both compact and full mode because
    # they are small and provide important context for the LLM's reasoning.
    # Confidence is only shown when positive (> 0) to avoid noise.
    scalars = []
    for key in ("component", "language"):
        val = meta.get(key)
        if val:
            scalars.append(f"{key}: {val}")
    conf = meta.get("confidence")
    if conf and float(conf) > 0:
        scalars.append(f"confidence: {conf}")
    if scalars:
        lines.append(" | ".join(scalars))

    # List fields — variable-length metadata arrays.
    # When compact mode is active (triggered by >8 results in _format_context),
    # only "files" and "symbols" are shown — the two most valuable fields for
    # the LLM to cite precise code locations.  This keeps the combined prompt
    # within token limits when many chunks are present.
    # In full mode, all list fields are shown, including call chains, API
    # boundaries, IPC mechanisms, messages, shared data, and tags.
    if compact:
        list_fields = ("files", "symbols")
    else:
        list_fields = (
            "files", "symbols", "call_chains",
            "api_exports", "api_imports",
            "ipc_mechanism", "messages_send", "messages_receive",
            "shared_data", "tags",
        )

    # Emit each list field on its own line, joining elements with commas.
    # Handles both list-typed and string-typed metadata values.
    for key in list_fields:
        val = meta.get(key)
        if val:
            if isinstance(val, list):
                joined = ", ".join(str(v) for v in val if v)
            else:
                joined = str(val)
            if joined:
                lines.append(f"{key}: {joined}")

    return "\n".join(lines)


def _format_context(results: list[SearchResult]) -> str:
    """Format search results as a context string for prompt injection.

    Each chunk is prefixed with a structured metadata header containing all
    available metadata fields, enabling the LLM to cite files, symbols,
    call chains, and other code-specific information in its answer.
    """
    # When there are many results (>8), switch to compact metadata headers to
    # reduce the total token count injected into the prompt.  This avoids
    # exceeding the LLM's context window while still providing enough metadata
    # for the model to cite sources accurately.
    compact = len(results) > 8
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        # Each chunk is formatted as a structured metadata header (built by
        # _format_metadata_header) followed by a blank line and the chunk's
        # text content.  The header gives the LLM structured fields (files,
        # symbols, call chains, etc.) it can reference when generating answers.
        header = _format_metadata_header(i, r.metadata, compact=compact)
        parts.append(f"{header}\n\n{r.content}")

    # Separate chunks with a "---" horizontal rule so the LLM can clearly
    # distinguish boundaries between different context sources.  The blank
    # lines around the separator improve readability in the raw prompt.
    return "\n\n---\n\n".join(parts)


def _extract_sources(results: list[SearchResult]) -> list[dict]:
    """Extract a deduplicated list of source references from search results.

    Deduplicates by topic + section combination. Includes code metadata for richer citations.
    """
    # Deduplicate sources by a composite key of "topic__section".  Multiple
    # chunks may originate from the same knowledge base article section (e.g.,
    # overlapping windows or multi-angle retrieval hitting the same content).
    # We only want to report each unique section once in the source list.
    sources = []
    seen = set()
    for r in results:
        topic = r.metadata.get("topic", "")
        section = r.metadata.get("section", "")
        # Use double-underscore as a separator that is unlikely to appear in
        # either topic or section names, avoiding false dedup collisions.
        key = f"{topic}__{section}"
        if key not in seen:
            seen.add(key)
            # Build the base source dict with structural fields and the
            # retrieval similarity score.
            src = {
                "domain": r.metadata.get("domain", ""),
                "topic": topic,
                "section": section,
                "source": r.metadata.get("source", ""),
                "score": r.score,
            }
            # Conditionally include code metadata fields (component, files,
            # symbols) when they are present, so callers can display richer
            # citations (e.g., listing the exact files and function names
            # that support a claim).
            for field in ("component", "files", "symbols"):
                val = r.metadata.get(field)
                if val:
                    src[field] = val
            sources.append(src)
    return sources


def create_llm(config: AppConfig) -> ChatOpenAI:
    """Create an LLM instance for answer generation.

    Uses a temperature slightly above 0 (0.1) to maintain accuracy
    while keeping answers more natural and fluent.

    Args:
        config: Application configuration

    Returns:
        A ChatOpenAI instance
    """
    # Temperature is set to 0.1 (near-deterministic) rather than the strict 0.0
    # used by the query_rewriter module.  The slight randomness produces more
    # natural, fluent prose in the final answer while keeping hallucination risk
    # very low.  Query rewriting uses 0.0 because it needs fully deterministic
    # keyword extraction, whereas answer generation benefits from minimal
    # stylistic variation.
    return ChatOpenAI(
        base_url=config.llm.base_url,
        model=config.llm.model,
        api_key=config.llm.api_key,
        temperature=0.1,
    )


def generate_answer(
    question: str,
    results: list[SearchResult],
    config: AppConfig,
    history: str = "",
) -> tuple[str, list[dict]]:
    """Generate a RAG answer based on search results.

    Selects different prompt templates depending on whether conversation history is present.
    Returns the cited source list alongside the generated answer.

    Args:
        question: User question
        results: List of retrieved chunks
        config: Application configuration
        history: Conversation history text (used for multi-turn conversations)

    Returns:
        A tuple of (answer text, source list)
    """
    llm = create_llm(config)
    # Format the raw search results into a context string with structured
    # metadata headers, and extract a deduplicated source list for citation.
    context = _format_context(results)
    sources = _extract_sources(results)

    # Select prompt template based on whether conversation history is present.
    # When history is provided, we use RAG_WITH_HISTORY_PROMPT which includes
    # a "Conversation history" section *before* the context so the LLM can
    # resolve pronouns and contextual references from prior turns.
    # When there is no history, we use the simpler RAG_PROMPT to avoid
    # injecting an empty history block that could confuse the model.
    if history:
        chain = RAG_WITH_HISTORY_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({
            "question": question,
            "context": context,
            "history": history,
        })
    else:
        chain = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"question": question, "context": context})

    return answer, sources


def generate_answer_stream(
    question: str,
    results: list[SearchResult],
    config: AppConfig,
    history: str = "",
) -> Generator[str, None, None]:
    """Generate a RAG answer in streaming mode.

    Returns tokens incrementally, suitable for scenarios requiring real-time display of generation progress.

    Args:
        question: User question
        results: List of retrieved chunks
        config: Application configuration
        history: Conversation history text

    Yields:
        Incremental fragments of the answer text
    """
    llm = create_llm(config)
    context = _format_context(results)

    # Build the LCEL (LangChain Expression Language) chain and input dict,
    # choosing the appropriate prompt template depending on history presence.
    if history:
        # Standard multi-turn mode — includes conversation history.
        chain = RAG_WITH_HISTORY_PROMPT | llm | StrOutputParser()
        input_data = {"question": question, "context": context, "history": history}
    else:
        # Single-turn mode — no history block in the prompt.
        chain = RAG_PROMPT | llm | StrOutputParser()
        input_data = {"question": question, "context": context}

    # Use LangChain's chain.stream() to yield tokens incrementally.
    # chain.stream() returns a generator of string fragments; each fragment
    # is a small piece of the LLM's response (often a single token or a few
    # tokens).  "yield from" delegates directly to that generator, so the
    # caller receives each fragment as soon as it arrives from the LLM —
    # enabling real-time display of the answer as it is being generated.
    yield from chain.stream(input_data)
