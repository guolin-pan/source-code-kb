"""LangGraph Agent state definition.

Defines the state data structure passed between nodes in the Agent workflow.
Uses TypedDict to declare field types; LangGraph automatically manages state updates and propagation.
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# LangGraph uses TypedDict as the canonical way to define the state schema.
# The state object is the single data structure that flows between every node
# in the graph: each node receives the full state as input and returns a
# partial dict of updates that LangGraph merges back into the state
# automatically.  This design decouples nodes from each other -- they
# communicate exclusively through this shared state rather than direct calls.
class AgentState(TypedDict):
    """RAG Agent state data structure.

    Passed between nodes in the LangGraph workflow; each node can read and update the state.

    Attributes:
        question: The user's original question
        query_type: Query type classification result ("simple" / "complex" / "compare")
        rewritten_queries: List of LLM-rewritten queries
        retrieved_chunks: List of retrieved knowledge chunks (dict format)
        sub_questions: List of sub-questions decomposed from complex queries
        sub_answers: List of sub-question answers (each containing question, answer, sources)
        answer: The final generated answer text
        sources: List of reference sources
        iteration: Current iteration count (for iterative retrieval scenarios)
        chunk_verdict: Relevance verdict from chunk evaluation (R3)
        _status: Human-readable status message from the last node (for streaming display)
        messages: LangGraph message list (for checkpointer conversation persistence)
    """

    # The original user question -- immutable across the entire pipeline.
    question: str

    # Set by the classify node; determines which processing path the graph
    # takes ("simple", "complex", or "compare").
    query_type: str

    # Multi-angle rewritten queries produced by the rewrite node (R2 strategy).
    # Multiple query variants improve recall by searching from different angles.
    rewritten_queries: list[str]

    # Code entities extracted by the LLM during the rewrite step (symbols,
    # files, components).  Passed to the graph retriever to improve graph
    # hit rates for natural-language questions.
    extracted_entities: dict[str, list[str]]

    # Retrieved document chunks stored as plain dicts (serializable).
    # Nodes convert these to/from SearchResult objects as needed.
    retrieved_chunks: list[dict[str, Any]]

    # Sub-questions produced by the decompose node for the "complex" path.
    sub_questions: list[str]

    # Each entry holds {"question": ..., "answer": ..., "sources": ...} for
    # one sub-question, populated by the sub_retrieve node.
    sub_answers: list[dict[str, Any]]

    # The final answer text produced by generate / synthesize / compare nodes.
    answer: str

    # Provenance information -- list of source metadata dicts attached to the answer.
    sources: list[dict[str, Any]]

    # Tracks how many retrieve-evaluate cycles have been attempted.  The R3
    # (Relevance-based Retrieval Retry) loop increments this counter each round
    # and gives up after 3 iterations to prevent infinite loops.
    iteration: int

    # The R3 evaluation verdict ("relevant", "partial", or "insufficient").
    # The conditional edge after the evaluate node inspects this value to
    # decide whether to proceed to generation or loop back for another
    # retrieval attempt.
    chunk_verdict: str

    # A Rich-markup status string updated by every node.  The streaming UI
    # reads this field after each node completes to show live progress to the
    # user (e.g., "Retrieved 12 chunks ...").  The leading underscore signals
    # that this is a display-only field, not part of the RAG data model.
    _status: str

    # The Annotated[list, add_messages] type tells LangGraph to use the
    # add_messages *reducer* (accumulator) instead of simple replacement.
    # When a node returns {"messages": [new_msg]}, LangGraph appends
    # new_msg to the existing list rather than overwriting it.  This is
    # essential for checkpointer-based conversation persistence where the
    # full message history must be preserved across turns.
    messages: Annotated[list, add_messages]
