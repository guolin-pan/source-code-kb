"""LangGraph Agent — Multi-step reasoning workflow graph.

Defines a LangGraph-based RAG Agent workflow supporting three query processing paths:

1. Simple query (simple):
   START → classify → rewrite → retrieve → rerank → generate → END

2. Complex query (complex):
   START → classify → decompose → sub_retrieve → synthesize → END

3. Comparison query (compare):
   START → classify → compare → END

The Agent automatically selects the path based on query classification, with no manual intervention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generator

from langgraph.graph import END, START, StateGraph

from source_code_kb.agent.nodes import (
    make_classify_node,
    make_compare_node,
    make_decompose_node,
    make_evaluate_node,
    make_generate_node,
    make_rerank_node,
    make_retrieve_node,
    make_rewrite_node,
    make_sub_retrieve_node,
    make_synthesize_node,
)
from source_code_kb.agent.state import AgentState

if TYPE_CHECKING:
    from source_code_kb.config import AppConfig


def _route_by_query_type(state: AgentState) -> str:
    """Route to the corresponding processing subgraph based on query type.

    Args:
        state: Current Agent state

    Returns:
        Target node name
    """
    # Read the classification result set by the classify_query node.
    # Default to "simple" if query_type was never set, ensuring the pipeline
    # always has a valid path to follow.
    qt = state.get("query_type", "simple")

    # Conditional routing: the returned string must match one of the keys in
    # the routing map passed to add_conditional_edges().  Each key maps to a
    # different processing subgraph:
    #   "complex"  -> decompose node  (multi-step reasoning path)
    #   "compare"  -> compare node    (side-by-side analysis path)
    #   otherwise  -> rewrite node    (simple single-pass RAG path)
    if qt == "complex":
        return "decompose"
    if qt == "compare":
        return "compare"
    return "rewrite"  # simple queries go through the rewrite path


def build_agent_graph(
    config: AppConfig,
    retriever: Any,
    on_token: callable | None = None,
) -> StateGraph:
    """Build the LangGraph Agent workflow graph.

    Graph structure:
        START → classify_query
          → [simple]  → rewrite → retrieve → rerank → evaluate → generate → END
          → [complex] → decompose → sub_retrieve → synthesize → END
          → [compare] → compare → END

    Args:
        config: Application configuration
        retriever: Hybrid retriever instance
        on_token: Optional callback for real-time token streaming from generation nodes.
                  Signature: ``on_token(token: str) -> None``

    Returns:
        An uncompiled StateGraph instance
    """
    # --- Step 1: Instantiate the graph with AgentState as the schema ---
    # StateGraph manages state propagation: it passes the full AgentState to
    # each node and merges the partial dict each node returns back into the
    # state (using reducers where defined, e.g., add_messages).
    graph = StateGraph(AgentState)

    # --- Step 2: Register all nodes ---
    # Each node is created via a factory function that captures config /
    # retriever / on_token through closures, producing a callable with the
    # signature (AgentState) -> dict.
    graph.add_node("classify_query", make_classify_node(config))
    graph.add_node("rewrite", make_rewrite_node(config))
    graph.add_node("retrieve", make_retrieve_node(config, retriever))
    graph.add_node("rerank", make_rerank_node(config))
    graph.add_node("evaluate", make_evaluate_node(config))
    graph.add_node("generate", make_generate_node(config, on_token=on_token))
    graph.add_node("decompose", make_decompose_node(config))
    graph.add_node("sub_retrieve", make_sub_retrieve_node(config, retriever))
    graph.add_node("synthesize", make_synthesize_node(config, on_token=on_token))
    graph.add_node("compare", make_compare_node(config, retriever, on_token=on_token))

    # --- Step 3: Wire edges ---
    # Entry edge: every invocation starts at the classification node.
    graph.add_edge(START, "classify_query")

    # Conditional routing after classification: dispatch to one of the three
    # processing paths.  The routing map keys must match the strings returned
    # by _route_by_query_type; the values are the target node names.
    graph.add_conditional_edges(
        "classify_query",
        _route_by_query_type,
        {
            "rewrite": "rewrite",
            "decompose": "decompose",
            "compare": "compare",
        },
    )

    # --- Simple query path ---
    # rewrite → retrieve → rerank → evaluate → (generate | retry rewrite)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "evaluate")

    # R3 retry mechanism: after evaluation, a conditional edge decides whether
    # to proceed to generation or loop back to rewrite for another retrieval
    # attempt.  The decision is based on chunk_verdict and the iteration counter.
    def _route_after_evaluate(state: AgentState) -> str:
        verdict = state.get("chunk_verdict", "relevant")
        iteration = state.get("iteration", 0)
        # If chunks are insufficient and we haven't exceeded the retry budget
        # (max 3 rounds), loop back to rewrite to try different query phrasings.
        # Otherwise, proceed to generate with whatever chunks we have.
        if verdict == "insufficient" and iteration < 3:
            return "rewrite"  # retry
        return "generate"

    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {
            "rewrite": "rewrite",
            "generate": "generate",
        },
    )
    # Terminal edge for the simple path.
    graph.add_edge("generate", END)

    # --- Complex query path ---
    # decompose → sub_retrieve → synthesize → END
    # Each sub-question gets its own mini retrieve-generate pipeline inside
    # the sub_retrieve node; synthesize then merges all sub-answers.
    graph.add_edge("decompose", "sub_retrieve")
    graph.add_edge("sub_retrieve", "synthesize")
    graph.add_edge("synthesize", END)

    # --- Comparison query path ---
    # compare → END  (single node handles retrieval + generation internally)
    graph.add_edge("compare", END)

    return graph


def create_agent(config: AppConfig, retriever: Any, on_token=None):
    """Create a compiled LangGraph Agent.

    Args:
        config: Application configuration
        retriever: Hybrid retriever instance
        on_token: Optional callback for real-time token streaming

    Returns:
        A compiled, directly callable Agent instance
    """
    # compile() finalizes the graph definition and returns a runnable object.
    # Compilation validates the graph structure (no dangling edges, all nodes
    # reachable, etc.) and produces an optimized executor that supports both
    # .invoke() for batch execution and .stream() for incremental execution.
    graph = build_agent_graph(config, retriever, on_token=on_token)
    return graph.compile()


def run_agent(
    question: str,
    config: AppConfig,
    retriever: Any,
) -> dict:
    """Run the Agent to process a question and return the result.

    Automatically completes the full pipeline: query classification → path selection → retrieval → answer generation.

    Args:
        question: User question
        config: Application configuration
        retriever: Hybrid retriever instance

    Returns:
        A result dict containing question, query_type, answer, and sources
    """
    agent = create_agent(config, retriever)

    # Initialize every field in AgentState explicitly.  All fields must be
    # present because LangGraph validates the state schema at invocation time
    # and nodes may read any field (e.g., evaluate reads "iteration").
    # Using empty/zero defaults ensures that early nodes (which only set a few
    # fields) don't encounter KeyError when downstream nodes access unrelated
    # state fields.
    initial_state: AgentState = {
        "question": question,
        "query_type": "",              # Set by classify_query
        "rewritten_queries": [],       # Set by rewrite
        "retrieved_chunks": [],        # Set by retrieve / sub_retrieve / compare
        "sub_questions": [],           # Set by decompose (complex path only)
        "sub_answers": [],             # Set by sub_retrieve (complex path only)
        "answer": "",                  # Set by generate / synthesize / compare
        "sources": [],                 # Set by generate / synthesize / compare
        "iteration": 0,               # Incremented by evaluate (R3 retry counter)
        "chunk_verdict": "",           # Set by evaluate (R3 verdict)
        "_status": "",                 # Updated by every node for UI display
        "messages": [],                # Accumulator for conversation persistence
    }

    # invoke() runs the full graph synchronously from START to END and returns
    # the final state.  We extract only the fields the caller needs.
    result = agent.invoke(initial_state)
    return {
        "question": question,
        "query_type": result.get("query_type", "simple"),
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }


def run_agent_stream(
    question: str,
    config: AppConfig,
    retriever: Any,
    on_token: callable | None = None,
) -> Generator[tuple[str, dict], None, None]:
    """Run the Agent in streaming mode, yielding node-level progress.

    Yields ``(node_name, node_output)`` tuples as each node completes.
    When *on_token* is provided, LLM-generating nodes (generate, synthesize,
    compare) also push individual tokens through the callback in real-time
    **during** execution, before the node-completion event is yielded.

    Args:
        question: User question
        config: Application configuration
        retriever: Hybrid retriever instance
        on_token: Optional callback invoked with each LLM token as it is
                  generated.  Signature: ``on_token(token: str) -> None``

    Yields:
        Tuples of (node_name, node_output_dict)
    """
    # Pass on_token so that generation nodes (generate, synthesize, compare)
    # can push individual tokens to the caller in real-time while the node is
    # still executing -- before the node-completion event is yielded below.
    agent = create_agent(config, retriever, on_token=on_token)

    # Same full initialization as run_agent -- see comments there.
    initial_state: AgentState = {
        "question": question,
        "query_type": "",
        "rewritten_queries": [],
        "retrieved_chunks": [],
        "sub_questions": [],
        "sub_answers": [],
        "answer": "",
        "sources": [],
        "iteration": 0,
        "chunk_verdict": "",
        "_status": "",
        "messages": [],
    }

    # agent.stream() yields one event dict per completed node.  Each event has
    # the shape {node_name: node_output_dict}, where node_output_dict is the
    # partial state update returned by that node.  We unpack these into
    # (node_name, node_output) tuples for a cleaner caller interface.
    for event in agent.stream(initial_state):
        # LangGraph stream yields {node_name: node_output} dicts
        for node_name, node_output in event.items():
            yield node_name, node_output
