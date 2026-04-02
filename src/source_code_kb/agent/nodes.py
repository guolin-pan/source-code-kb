"""LangGraph Agent node functions — Each function corresponds to a processing node in the workflow graph.

Node descriptions:
- classify_query: Classify query type (simple/complex/compare)
- rewrite: Rewrite query into a form better suited for retrieval
- retrieve: Execute vector search
- rerank: Re-rank search results
- generate: Generate the final answer based on search results
- decompose: Decompose complex questions into sub-questions
- sub_retrieve: Retrieve and answer each sub-question individually
- synthesize: Synthesize sub-question answers into a final answer
- compare: Perform comparative analysis

Each node function is created via a factory function (make_xxx_node),
capturing config and retriever dependencies through closures.
"""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from source_code_kb.agent.state import AgentState
from source_code_kb.config import AppConfig
from source_code_kb.generation.generator import _extract_sources, _format_context, evaluate_chunks, generate_answer
from source_code_kb.generation.prompts import (
    CLASSIFY_QUERY_PROMPT,
    COMPARE_PROMPT,
    DECOMPOSE_PROMPT,
    SYNTHESIZE_PROMPT,
)
from source_code_kb.retrieval.query_rewriter import create_llm, generate_multi_angle_queries, rewrite_query
from source_code_kb.retrieval.reranker import rerank
from source_code_kb.retrieval.retriever import HybridRetriever


# --- Factory function pattern ---
# Every make_*_node function is a factory that returns a closure.  The closure
# captures ``config`` (and optionally ``retriever``, ``on_token``) so that the
# inner function conforms to the LangGraph node signature:
#     (state: AgentState) -> dict[str, Any]
# This avoids global state and makes nodes independently testable.

def make_classify_node(config: AppConfig):
    """Create the query classification node.

    Uses LLM to classify user queries into simple/complex/compare types,
    determining which processing path to follow.
    """

    def classify_query(state: AgentState) -> dict[str, Any]:
        # Build a simple LangChain chain: prompt -> LLM -> parse to string.
        llm = create_llm(config)
        chain = CLASSIFY_QUERY_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"query": state["question"]}).strip().lower()

        # Normalization fallback: if the LLM returns anything other than one
        # of the three expected labels (e.g., a full sentence or a typo),
        # default to "simple" so the pipeline always proceeds safely.
        if result not in ("simple", "complex", "compare"):
            result = "simple"

        # Human-readable path description shown in the streaming status bar.
        path_labels = {"simple": "Simple → rewrite → retrieve → generate",
                       "complex": "Complex → decompose → sub-retrieve → synthesize",
                       "compare": "Compare → retrieve → comparative analysis"}
        return {
            "query_type": result,
            "_status": f"Query type: [cyan]{result}[/cyan]  |  Path: {path_labels[result]}",
        }

    # Return the inner function -- LangGraph will call it with the state dict.
    return classify_query


def make_rewrite_node(config: AppConfig):
    """Create the query rewrite node.

    Uses multi-angle query generation (R2) to produce 3-5 English queries
    from different angles for improved retrieval coverage.
    """

    def rewrite(state: AgentState) -> dict[str, Any]:
        # R2 strategy: generate multiple query variants from different angles
        # (e.g., synonym-based, concept-focused, example-seeking).  This
        # increases recall because a single phrasing may miss relevant
        # documents that use different terminology.
        queries = generate_multi_angle_queries(state["question"], config)

        # If multi-angle generation fails (e.g., LLM error or empty output),
        # fall back to the original question so retrieval can still proceed.
        if not queries:
            queries = [state["question"]]

        # Preview the first 5 queries in the status for user visibility.
        query_preview = "  |  ".join(queries[:5])
        return {
            "rewritten_queries": queries,
            "_status": f"Generated [cyan]{len(queries)}[/cyan] multi-angle queries:\n         {query_preview}",
        }

    return rewrite


def make_retrieve_node(config: AppConfig, retriever: HybridRetriever):
    """Create the retrieval node.

    Executes vector search for all rewritten queries, merging and deduplicating results.
    Falls back to the original question if no rewritten queries exist.
    """

    def retrieve(state: AgentState) -> dict[str, Any]:
        # Fall back to the original question when rewritten_queries is empty
        # (e.g., on the first pass before rewrite runs, or if rewrite failed).
        queries = state.get("rewritten_queries") or [state["question"]]
        all_chunks: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # Execute a vector search for each query variant, then merge results.
        # Deduplication by doc_id prevents the same chunk from appearing
        # multiple times when different query phrasings retrieve overlapping
        # documents.  When no explicit "id" exists in metadata, the first 50
        # characters of content serve as a rough dedup key.
        for query in queries:
            results = retriever.search(query)
            for r in results:
                doc_id = r.metadata.get("id", r.content[:50])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_chunks.append(r.to_dict())

        # Build a topic frequency summary for the status display.  This gives
        # the user a quick sense of what domains the retrieved chunks cover
        # (e.g., "auth(4), config(3), logging(2)").
        topics = {}
        for c in all_chunks:
            t = c.get("metadata", {}).get("topic", "?")
            topics[t] = topics.get(t, 0) + 1
        # Show only the top-5 topics, sorted by frequency descending.
        topic_summary = ", ".join(f"{t}({n})" for t, n in sorted(topics.items(), key=lambda x: -x[1])[:5])

        return {
            "retrieved_chunks": all_chunks,
            "_status": f"Retrieved [cyan]{len(all_chunks)}[/cyan] chunks from {len(queries)} queries  |  Topics: {topic_summary}",
        }

    return retrieve


def make_rerank_node(config: AppConfig):
    """Create the re-ranking node.

    Re-scores and reorders search results using a Reranker model.
    Skips this step if the reranker is disabled in configuration.
    """

    def rerank_node(state: AgentState) -> dict[str, Any]:
        # Early exit: when the reranker is disabled in config, skip entirely.
        # Returning _skip_display tells the streaming UI not to show this node.
        if not config.retrieval.use_reranker:
            return {"_skip_display": True}

        # Lazy import to avoid loading the SearchResult class (and potentially
        # its heavy dependencies) when reranking is disabled.
        from source_code_kb.retrieval.retriever import SearchResult

        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return {"_status": "No chunks to rerank — skipped"}

        # The state stores chunks as plain dicts for JSON serializability.
        # Convert them back to SearchResult objects because the rerank()
        # function expects typed objects with .content / .metadata / .score
        # attributes.
        results = [
            SearchResult(
                content=c["content"],
                metadata=c["metadata"],
                score=c["score"],
            )
            for c in chunks
        ]

        # Re-score and re-order using a cross-encoder or similar reranker model.
        reranked = rerank(state["question"], results, config)

        # Convert back to dicts for state storage.
        return {
            "retrieved_chunks": [r.to_dict() for r in reranked],
            "_status": f"Reranked [cyan]{len(chunks)}[/cyan] → top [cyan]{len(reranked)}[/cyan] chunks",
        }

    return rerank_node


def make_evaluate_node(config: AppConfig):
    """Create the chunk evaluation node (R3).

    Evaluates relevance of retrieved chunks and filters out irrelevant ones.
    If insufficient after filtering and iteration < 3, triggers retry via conditional edge.
    """

    def evaluate(state: AgentState) -> dict[str, Any]:
        from source_code_kb.retrieval.retriever import SearchResult

        chunks = state.get("retrieved_chunks", [])
        iteration = state.get("iteration", 0)

        # Edge case: no chunks at all means we must retry (if iterations remain).
        if not chunks:
            return {
                "chunk_verdict": "insufficient",
                "iteration": iteration + 1,
                "_status": f"Round {iteration + 1}: [red]insufficient[/red] — no chunks",
            }

        # Convert dict chunks back to SearchResult objects for the evaluation
        # function, which needs typed objects to inspect content and metadata.
        results = [
            SearchResult(content=c["content"], metadata=c["metadata"], score=c["score"])
            for c in chunks
        ]

        # R3 evaluation: an LLM judges whether the retrieved chunks are
        # "relevant", "partial", or "insufficient" for answering the question.
        # It also returns a filtered subset of chunks deemed relevant.
        verdict, filtered_results = evaluate_chunks(state["question"], results, config)

        # Replace the full chunk list with only the chunks that passed the
        # relevance filter, discarding noise.
        filtered_chunks = [
            {"content": r.content, "metadata": r.metadata, "score": r.score}
            for r in filtered_results
        ]

        # Color-code the verdict for the streaming status display.
        color = {"relevant": "green", "partial": "yellow", "insufficient": "red"}.get(verdict, "dim")

        # If the verdict is "insufficient" and we haven't exhausted the retry
        # budget (max 3 iterations), signal that the graph's conditional edge
        # should route back to the rewrite node for another attempt with
        # different query formulations.
        action = ""
        if verdict == "insufficient" and iteration + 1 < 3:
            action = " → retrying with new queries"

        status = f"Round {iteration + 1}: [{color}]{verdict}[/{color}] ({len(chunks)} → {len(filtered_chunks)} kept){action}"

        return {
            "retrieved_chunks": filtered_chunks,  # Replace with filtered
            # If all chunks were filtered out, override verdict to "insufficient"
            # so the retry logic triggers even if the LLM said "partial".
            "chunk_verdict": verdict if filtered_chunks else "insufficient",
            # Increment the iteration counter so the conditional edge in the
            # graph can enforce the max-3-retries limit.
            "iteration": iteration + 1,
            "_status": status,
        }

    return evaluate


def make_generate_node(config: AppConfig, on_token=None):
    """Create the answer generation node.

    Uses the retrieved (or re-ranked) results as context and calls LLM to generate the final answer.
    When *on_token* is provided, streams tokens through the callback for real-time display.
    """

    def generate(state: AgentState) -> dict[str, Any]:
        from source_code_kb.retrieval.retriever import SearchResult

        # Reconstruct SearchResult objects from the serialized dict form stored
        # in the state so that generation helpers can work with typed objects.
        chunks = state.get("retrieved_chunks", [])
        results = [
            SearchResult(content=c["content"], metadata=c["metadata"], score=c["score"])
            for c in chunks
        ]

        # Streaming path: when an on_token callback is provided (typically by
        # the CLI or web UI), use the streaming generator variant of answer
        # generation.  Each token is pushed to the callback *as it arrives*
        # from the LLM, enabling real-time typewriter-style display, while
        # also being collected into answer_parts so we can store the complete
        # answer in the state after the stream finishes.
        if on_token:
            from source_code_kb.generation.generator import _extract_sources, generate_answer_stream

            sources = _extract_sources(results)
            answer_parts: list[str] = []
            for token in generate_answer_stream(
                state["question"], results, config
            ):
                on_token(token)
                answer_parts.append(token)
            return {"answer": "".join(answer_parts), "sources": sources}

        # Non-streaming path: invoke the LLM in one shot and return the
        # complete answer.  Used in batch / programmatic mode.
        answer, sources = generate_answer(state["question"], results, config)
        return {"answer": answer, "sources": sources}

    return generate


def make_decompose_node(config: AppConfig):
    """Create the question decomposition node.

    Splits a complex question into 2-4 independently retrievable and answerable sub-questions.
    Used in the complex query type processing path.
    """

    def decompose(state: AgentState) -> dict[str, Any]:
        # Build a chain that asks the LLM to break the complex question into
        # smaller, self-contained sub-questions that can each be answered
        # independently through their own retrieve-and-generate mini-pipeline.
        llm = create_llm(config)
        chain = DECOMPOSE_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"query": state["question"]})

        # Parse the LLM output: each line is expected to be one sub-question.
        # Empty lines and whitespace-only lines are filtered out.
        sub_questions = [q.strip() for q in result.strip().split("\n") if q.strip()]

        # Format the sub-questions as a bulleted list for the status display.
        sq_list = "\n         ".join(f"• {q}" for q in sub_questions)
        return {
            "sub_questions": sub_questions,
            "_status": f"Decomposed into [cyan]{len(sub_questions)}[/cyan] sub-questions:\n         {sq_list}",
        }

    return decompose


def make_sub_retrieve_node(config: AppConfig, retriever: HybridRetriever):
    """Create the sub-question retrieval and answering node.

    For each sub-question, performs: retrieval → re-ranking → answer generation.
    Collects answers and sources from all sub-questions for the subsequent synthesis node.
    """

    def sub_retrieve(state: AgentState) -> dict[str, Any]:
        sub_questions = state.get("sub_questions", [])
        sub_answers: list[dict[str, Any]] = []

        # Run a full mini-pipeline (retrieve → optional rerank → generate)
        # for each sub-question independently.  This is the "complex" path's
        # way of handling multi-faceted questions: answer each facet on its
        # own, then synthesize the results in the next node.
        for sq in sub_questions:
            # Retrieval: search the vector store for this sub-question.
            results = retriever.search(sq)

            # Optional re-ranking: apply the cross-encoder reranker if enabled.
            if config.retrieval.use_reranker:
                results = rerank(sq, results, config)

            # Generate an answer for this individual sub-question.
            answer, sources = generate_answer(sq, results, config)
            sub_answers.append({
                "question": sq,
                "answer": answer,
                "sources": sources,
            })

        # Aggregate all source references from every sub-question into a
        # single flat list.  This is stored in retrieved_chunks so that the
        # final output can report all sources that contributed to the answer.
        all_chunks = []
        for sa in sub_answers:
            for src in sa["sources"]:
                all_chunks.append(src)

        total_sources = sum(len(sa["sources"]) for sa in sub_answers)
        return {
            "sub_answers": sub_answers,
            "retrieved_chunks": all_chunks,
            "_status": f"Answered [cyan]{len(sub_answers)}[/cyan] sub-questions  |  {total_sources} source chunks collected",
        }

    return sub_retrieve


def make_synthesize_node(config: AppConfig, on_token=None):
    """Create the synthesis node.

    Synthesizes multiple sub-question answers into a single, coherent final answer.
    When *on_token* is provided, streams tokens through the callback for real-time display.
    """

    def synthesize(state: AgentState) -> dict[str, Any]:
        # Build a synthesis chain that takes all sub-question answers and
        # combines them into one coherent response to the original question.
        llm = create_llm(config)
        chain = SYNTHESIZE_PROMPT | llm | StrOutputParser()

        # Format all sub-question/answer pairs into a single text block that
        # the LLM can reason over holistically.
        sub_text = ""
        all_sources: list[dict[str, Any]] = []
        for sa in state.get("sub_answers", []):
            sub_text += f"Sub-question: {sa['question']}\nAnswer: {sa['answer']}\n\n"
            # Collect sources from every sub-answer so the final response can
            # cite all provenance information.
            all_sources.extend(sa.get("sources", []))

        input_data = {"question": state["question"], "sub_answers": sub_text}

        # Streaming path: push tokens through the callback as they arrive,
        # while accumulating them for the complete answer string.
        if on_token:
            answer_parts: list[str] = []
            for token in chain.stream(input_data):
                on_token(token)
                answer_parts.append(token)
            answer = "".join(answer_parts)
        else:
            # Non-streaming path: invoke in one shot.
            answer = chain.invoke(input_data)

        return {"answer": answer, "sources": all_sources}

    return synthesize


def make_compare_node(config: AppConfig, retriever: HybridRetriever, on_token=None):
    """Create the comparative analysis node.

    Retrieves more candidate documents (top_k=15), re-ranks them (top_n=10),
    and generates a structured similarities/differences analysis using a comparison prompt.
    When *on_token* is provided, streams tokens through the callback for real-time display.
    """

    def compare(state: AgentState) -> dict[str, Any]:
        # Comparison queries need a wider retrieval window than simple queries
        # because they typically involve two or more entities/concepts, and we
        # need sufficient context about *each* one.  top_k=15 (vs. the default,
        # typically 5-10) ensures both sides of the comparison are well-represented.
        results = retriever.search(state["question"], top_k=15)

        # Re-rank with a higher top_n=10 cutoff to retain more diverse results,
        # ensuring both compared entities survive the reranking filter.
        if config.retrieval.use_reranker:
            results = rerank(state["question"], results, config, top_n=10)

        context = _format_context(results)
        sources = _extract_sources(results)

        # Use the comparison-specific prompt (COMPARE_PROMPT) instead of the
        # standard generation prompt.  This prompt instructs the LLM to produce
        # a structured similarities/differences analysis rather than a plain
        # narrative answer.
        llm = create_llm(config)
        chain = COMPARE_PROMPT | llm | StrOutputParser()
        input_data = {"question": state["question"], "context": context}

        # Streaming path: push tokens through the callback for real-time display.
        if on_token:
            answer_parts: list[str] = []
            for token in chain.stream(input_data):
                on_token(token)
                answer_parts.append(token)
            answer = "".join(answer_parts)
        else:
            # Non-streaming path: invoke in one shot.
            answer = chain.invoke(input_data)

        return {
            "answer": answer,
            "sources": sources,
            # Store retrieved chunks in the state for potential downstream use
            # (e.g., logging or inspection).
            "retrieved_chunks": [r.to_dict() for r in results],
        }

    return compare
