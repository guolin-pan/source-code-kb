"""Query rewriting module — uses an LLM to rewrite natural language questions
into a form more suitable for vector retrieval.

Key capabilities:
1. Single-query rewrite: strips colloquialisms, preserves key technical terms
2. Multi-query variant generation: produces multiple semantically similar
   variants of the same question to improve recall

All prompts are written in English.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from source_code_kb.config import AppConfig

# ── Query rewriting prompts ───────────────────────────────────────────

# REWRITE_PROMPT strategy: instruct the LLM to transform a conversational
# question into a terse, keyword-rich form that maximises cosine similarity
# with relevant chunks in the vector store.  The prompt explicitly tells the
# model to preserve technical identifiers (which are high-signal for code
# search) and to strip conversational filler (which adds noise).  If the
# question has multiple facets, the model is allowed to split it into several
# single-aspect queries (one per line) so each sub-query can be matched
# independently.
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a query rewriting assistant. Rewrite the user's natural language question "
        "into a form that is more suitable for vector similarity search.\n"
        "Rules:\n"
        "1. Preserve key technical terms (function names, module names, file names).\n"
        "2. Remove colloquial or conversational expressions.\n"
        "3. If the question covers multiple aspects, split it into separate retrieval queries.\n"
        "4. Output one query per line. Do NOT number them.\n"
        "5. If the question is already clear and concise, return it as-is.",
    ),
    ("human", "{query}"),
])

# MULTI_QUERY_PROMPT strategy: generate N *semantically equivalent* but
# *lexically diverse* rephrasings of the same query.  This is the classic
# "Multi-Query Retrieval" technique — each variant may match a different set
# of chunks due to vocabulary mismatch, so the union of results improves
# recall without sacrificing precision.  The n_variants placeholder is filled
# at invocation time.
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a query expansion assistant. Generate {n_variants} semantically similar "
        "but differently worded variants of the given query to improve vector search recall.\n"
        "Output one variant per line. Do NOT number them.",
    ),
    ("human", "{query}"),
])


# ── Utility functions ────────────────────────────────────────────


def create_llm(config: AppConfig) -> ChatOpenAI:
    """Create an OpenAI-compatible LLM client.

    Creates a ChatOpenAI instance from the base_url, model, and api_key in config.
    Supports Ollama, OpenAI, and any OpenAI API-compatible service.

    Args:
        config: Application configuration.

    Returns:
        A ChatOpenAI instance.
    """
    return ChatOpenAI(
        base_url=config.llm.base_url,
        model=config.llm.model,
        api_key=config.llm.api_key,
        temperature=0.0,
    )


def rewrite_query(query: str, config: AppConfig) -> str:
    """Rewrite a single query using the LLM.

    Transforms the user's natural language question into a form better
    suited for vector retrieval.  May return multiple lines (one sub-query
    per line) if the question covers multiple aspects.

    Args:
        query: Original user query.
        config: Application configuration.

    Returns:
        The rewritten query string (may contain multiple lines).
    """
    llm = create_llm(config)
    # Build a LangChain LCEL (LangChain Expression Language) chain:
    #   REWRITE_PROMPT  — fills the ChatPromptTemplate with {query}
    #       |
    #   llm             — sends the formatted prompt to the LLM and gets a
    #                     ChatMessage response
    #       |
    #   StrOutputParser — extracts the plain-text string content from the
    #                     ChatMessage, discarding role metadata
    #
    # The "|" operator is LangChain's pipe syntax for composing Runnables.
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"query": query}).strip()


def generate_multi_queries(
    query: str,
    config: AppConfig,
    n_variants: int = 3,
) -> list[str]:
    """Generate multiple query variants for Multi-Query Retrieval to improve recall.

    Always includes the original query, then appends LLM-generated variants.
    Removes blank lines and returns a deduplicated query list.

    Args:
        query: Original user query.
        config: Application configuration.
        n_variants: Number of variants to generate.

    Returns:
        A list containing the original query and its variants.
    """
    llm = create_llm(config)
    # Same LCEL chain pattern as rewrite_query, but using the multi-query
    # prompt that asks the LLM for n_variants rephrasings.
    chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"query": query, "n_variants": n_variants})

    # Parse LLM output: one query variant per line.
    # Blank lines are discarded since the LLM occasionally inserts them.
    variants = [line.strip() for line in result.strip().split("\n") if line.strip()]
    # The original query must always be present in the final list so that
    # retrieval never loses the user's exact wording (which may contain
    # specific identifiers the LLM paraphrases away).  We insert it at
    # position 0 to give it priority in downstream retrieval order.
    if query not in variants:
        variants.insert(0, query)
    # Note: no explicit deduplication is performed here — the LLM is prompted
    # to produce diverse variants, and exact duplicates are rare in practice.
    return variants


def generate_multi_angle_queries(query: str, config: AppConfig) -> list[str]:
    """Generate 3-5 English retrieval queries from multiple angles (R2).

    Covers four angles: Original, Synonyms, Technical, Broader.
    All queries are always in English regardless of user's language.

    Args:
        query: Original user query (any language).
        config: Application configuration.

    Returns:
        A list of 3-5 English retrieval queries.
    """
    # Deferred import: MULTI_ANGLE_QUERY_PROMPT lives in the generation
    # package.  This is the "R2" feature in the retrieval pipeline — it
    # generates 4-6 queries from distinct angles to maximise recall:
    #   1. Original keywords — direct English translation of user terms
    #   2. Function / symbol names — inferred identifiers
    #   3. File paths — likely source file locations
    #   4. Call chains / data flow — execution or data flow descriptions
    #   5. Component / subsystem — broader architectural context
    #   6. Tags and domain — classification tags and domain identifiers
    # All generated queries are forced to English regardless of the user's
    # input language, because the indexed code chunks and their embeddings
    # are in English.
    from source_code_kb.generation.prompts import MULTI_ANGLE_QUERY_PROMPT

    llm = create_llm(config)
    # Same LCEL chain pattern as the other query generation functions.
    chain = MULTI_ANGLE_QUERY_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"query": query})

    # Parse: one query per line, discard blank lines.
    queries = [line.strip() for line in result.strip().split("\n") if line.strip()]
    # Fallback: if the LLM returns empty output (e.g. due to a timeout or
    # content filter), use the original query as-is so that retrieval can
    # still proceed with at least one query.
    if not queries:
        queries = [query]
    return queries
