"""RAG prompt template module.

Defines all LLM interaction prompt templates used in the system, including:
- RAG Q&A prompts (single-turn / multi-turn conversation)
- Query classification prompt
- Question decomposition prompt
- Answer synthesis prompt
- Comparative analysis prompt

All prompts are written in English so the LLM can better follow instructions.
"""

from langchain_core.prompts import ChatPromptTemplate

# ── RAG Q&A Prompts ─────────────────────────────────────────────

# System prompt: instructs the LLM to answer questions based on retrieved context.
#
# Design notes:
# - Rules 1-5 enforce strict grounding in the provided context to prevent
#   hallucination.  The LLM must never supplement answers with its own
#   parametric knowledge.
# - Rule 6 (language matching) ensures the answer language follows the user's
#   language while keeping code identifiers verbatim.
# - The "Context Format" section documents the structured metadata fields that
#   _format_metadata_header() injects before each chunk.  This teaches the LLM
#   what metadata is available and how to cite it (files, symbols, call_chains,
#   etc.), so answers include precise code references rather than vague claims.
# - The "confidence" field enables conflict resolution: when two context sources
#   contradict each other, the LLM is instructed to prefer the one with higher
#   confidence, which reflects the knowledge base's self-assessed reliability
#   score (0.0-1.0) assigned during ingestion.
# - Three answer confidence levels (High / Partial / Insufficient) guide the
#   LLM to transparently signal the strength of its evidence rather than
#   presenting uncertain answers as definitive.
# - The priority order at the end reiterates that the KB context is the ONLY
#   allowed information source.
RAG_SYSTEM_PROMPT = """\
You are a source code architecture expert assistant for a code knowledge base. \
Answer user questions based STRICTLY on the retrieved context information below.

CRITICAL Rules:
1. Answer ONLY based on the provided context (KB chunks). Do NOT fabricate, guess, or speculate.
2. Do NOT use your general knowledge to supplement answers. Stick to what's in the context.
3. If the context is insufficient or contains NO relevant information, you MUST explicitly state: \
"The knowledge base does not contain sufficient information to answer this question." \
Do NOT attempt to answer beyond the context.
4. Every claim in your answer must be traceable to the provided context. \
Cite sources: mention relevant function names, file paths, and call chains.
5. If you're uncertain about any detail, state the uncertainty explicitly. \
Never present speculation as fact.
6. Use the same language as the user's question. Keep technical terms (function names, \
file paths, variable names, API names) in their original language.
7. Structure your answer using markdown format for readability.
8. When describing code behavior, reference specific functions and call chains from the context.
9. When discussing cross-module interactions, cite the API exports/imports and IPC mechanisms.

Context Format:
Each context source includes structured metadata headers before the content text. \
Use these metadata fields to provide precise, evidence-based answers:
- files: Source file paths — cite these for precise file references, do NOT invent paths.
- symbols: Function names, classes, variables — reference these when discussing code entities.
- call_chains: Execution flow (e.g., main→init→setup) — use these when describing control flow.
- component: Software component or subsystem name — use to identify module boundaries.
- api_exports / api_imports: Module boundary APIs — cite when discussing interfaces.
- ipc_mechanism: Inter-process communication methods.
- messages_send / messages_receive: Message-based communication endpoints.
- shared_data: Shared data structures between modules.
- confidence: Source reliability (0.0-1.0) — when two sources conflict, prefer higher confidence.
- tags: Categorization labels.

Answer confidence levels (use when appropriate):
- High confidence: The context directly addresses the question with clear evidence (function names, call chains).
- Partial answer: The context provides some relevant information but gaps remain (specify what's missing).
- Insufficient data: The context lacks the information needed to answer.

Priority order:
1. Explicit context from KB chunks (ONLY source)
2. If context is insufficient → state this explicitly, do NOT guess"""

# User prompt template: combines retrieved context with the user question.
# The {context} placeholder is filled by _format_context() with the structured
# metadata headers + chunk content; {question} is the raw user query.
RAG_USER_PROMPT = """\
Context:
{context}

Question: {question}"""

# Single-turn RAG Q&A prompt (no conversation history).
# Used by generate_answer() and generate_answer_stream() when the caller does
# not provide a history string.
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", RAG_USER_PROMPT),
])

# Multi-turn RAG Q&A prompt (includes conversation history for contextual
# continuity).  The history block is placed before the context so the LLM
# can resolve coreferences ("it", "that function", etc.) from earlier turns
# before reading the new context chunks.
RAG_WITH_HISTORY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "Conversation history:\n{history}\n\nContext:\n{context}\n\nQuestion: {question}"),
])

# ── Query Classification Prompt ─────────────────────────────────

# Classify user queries into one of three types to determine the subsequent
# processing flow in the orchestrator:
#
# - "simple"  -> Direct single-retrieval: the query targets a single module,
#   function, or code component.  The orchestrator performs one vector search
#   and feeds the results to generate_answer().
#
# - "complex" -> Decompose-and-synthesize: the query requires cross-module
#   analysis (e.g., tracing a call chain across subsystems, understanding an
#   end-to-end data flow).  The orchestrator uses DECOMPOSE_PROMPT to split
#   the question into sub-questions, retrieves chunks for each, and then uses
#   SYNTHESIZE_PROMPT to merge the sub-answers.
#
# - "compare" -> Comparative analysis: the query asks to compare two or more
#   modules, interfaces, or patterns.  The orchestrator uses COMPARE_PROMPT
#   with side-by-side context from each subject.
#
# The prompt forces the LLM to return exactly one word so parsing is trivial.
CLASSIFY_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Classify the user query into exactly one of the following types for source code analysis. "
        "Return ONLY the type name, nothing else:\n"
        "- simple: A query about a single module, function, or code component that can be answered by direct retrieval.\n"
        "- complex: A query requiring cross-module analysis, tracing call chains across multiple components, or understanding end-to-end data flows.\n"
        "- compare: A query asking to compare two or more modules, interfaces, implementation approaches, or architectural patterns.\n"
        "Return exactly one word: simple, complex, or compare.",
    ),
    ("human", "{query}"),
])

# ── Question Decomposition Prompt ───────────────────────────────

# Break down complex questions into independently retrievable sub-questions.
# The decomposition is guided along four dimensions that reflect common
# source-code analysis patterns:
#   1. Module internals   — how a single module works inside (implementation).
#   2. Module interfaces  — what APIs a module exposes or consumes (contracts).
#   3. Cross-module interactions — how modules talk to each other (IPC, shared
#      data, message passing).
#   4. Data flow          — how data moves through the system end-to-end.
#
# By splitting along these dimensions, each sub-question maps well to a
# focused vector search query, improving retrieval precision compared to
# searching for the full complex question at once.
#
# The output is limited to 2-4 sub-questions: enough to cover the question's
# scope without introducing excessive retrieval latency.  Sub-questions are
# unnumbered (one per line) so the parser can split on newlines simply.
DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Break the complex source code analysis question into 2-4 independent sub-questions "
        "that can each be answered by a single retrieval step.\n\n"
        "Consider decomposing along these dimensions:\n"
        "- Module internals: How does a specific module work internally?\n"
        "- Module interfaces: What APIs does it expose or consume?\n"
        "- Cross-module interactions: How do modules communicate (IPC, shared data, messages)?\n"
        "- Data flow: How does data flow through the system end-to-end?\n\n"
        "Output one sub-question per line. Do NOT number them.",
    ),
    ("human", "{query}"),
])

# ── Answer Synthesis Prompt ──────────────────────────────────────

# Synthesize multiple sub-question answers into a single, complete final answer
SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You have received multiple sub-questions and their individual answers "
        "for a complex question.\n"
        "Synthesize them into a single, coherent, and complete final answer.\n\n"
        "CRITICAL: Only use information from the provided sub-answers. "
        "Do NOT add speculation, assumptions, or external knowledge. "
        "If sub-answers indicate insufficient information, preserve that limitation in synthesis.\n"
        "Use markdown format and cite sources where applicable.",
    ),
    (
        "human",
        "Original question: {question}\n\n"
        "Sub-questions and answers:\n{sub_answers}\n\n"
        "Please provide a synthesized answer:",
    ),
])

# ── Comparative Analysis Prompt ─────────────────────────────────

# Compare and analyze the similarities and differences of multiple concepts or
# modules.  The prompt is metadata-aware: it explicitly tells the LLM which
# structured fields are available in the context headers (component, files,
# symbols, call_chains, api_exports, api_imports, ipc_mechanism, etc.) so the
# model can use them for precise, evidence-based comparison.
#
# Five comparison dimensions are suggested to the model:
#   1. Interface differences — API exports, function signatures, conventions.
#   2. Communication mechanisms — IPC methods, message formats, protocols.
#   3. Data structures — shared data, schemas, configs.
#   4. Initialization and lifecycle — startup, dependencies, shutdown.
#   5. Error handling — failure modes, fallbacks, logging.
#
# These dimensions were chosen because they represent the most common axes
# along which code modules differ in practice.  The LLM is instructed to only
# compare along dimensions that are actually evidenced in the context, and to
# explicitly state when data is missing — preventing speculative comparisons.
# Output format is requested as a markdown table or comparison list with
# per-point citations for traceability.
COMPARE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You need to compare and analyze multiple code modules, components, or implementation approaches. "
        "Based STRICTLY on the retrieved context, list their similarities and differences.\n\n"
        "Each context source includes structured metadata (component, files, symbols, call_chains, "
        "api_exports, api_imports, ipc_mechanism, etc.). Use these fields for precise comparison.\n\n"
        "Comparison dimensions to consider (when applicable):\n"
        "- Interface differences: API exports, function signatures, calling conventions\n"
        "- Communication mechanisms: IPC methods, message formats, protocols\n"
        "- Data structures: Shared data, database schemas, configuration formats\n"
        "- Initialization and lifecycle: Startup sequence, dependencies, shutdown behavior\n"
        "- Error handling: Failure modes, fallback strategies, logging patterns\n\n"
        "CRITICAL: Only compare aspects that are explicitly covered in the context. "
        "Do NOT speculate about similarities/differences not evidenced in the context. "
        "If the context lacks information for certain comparison dimensions, state this explicitly.\n"
        "Use a table or comparison list in markdown format. Cite function names, file paths, and sources for each point.",
    ),
    (
        "human",
        "Comparison question: {question}\n\n"
        "Context:\n{context}\n\n"
        "Please provide a comparative analysis:",
    ),
])

# ── Multi-Angle Query Prompt (R2) ───────────────────────────────

# Generate 4-6 English retrieval queries from multiple angles for a single
# user question.  This is the R2 feature (multi-angle query expansion) which
# improves recall by searching the vector store from several complementary
# perspectives.
#
# Six angles are defined:
#   1. Original keywords     — a direct English translation of the user's query
#      terms.  Captures the most literal match.
#   2. Function/symbol names — extracts or infers concrete code identifiers
#      (function names, class names, macros, globals).  Targets chunks that
#      mention those symbols.
#   3. File paths            — infers likely source file paths (e.g.,
#      module/main.c).  Useful because many KB chunks include file metadata.
#   4. Call chains / data flow — describes execution or data flow patterns
#      (e.g., "caller -> callee -> handler").  Matches chunks annotated with
#      call_chains metadata.
#   5. Component / subsystem — uses broader architectural terms (e.g., "alarm
#      management subsystem").  Improves recall for chunks that describe
#      high-level design rather than specific functions.
#   6. Tags and domain       — generates queries using classification tags and
#      domain identifiers.  Targets the tags/domain metadata fields.
#
# ALL queries must be in English regardless of the user's input language
# because the knowledge base embeddings are English-only.  Generating queries
# in the user's original language would produce poor vector similarity scores.
# Technical terms (function names, file names, error codes) are preserved
# verbatim to ensure exact matching.
MULTI_ANGLE_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a source code retrieval query expansion assistant. Given a user question (in any language), "
        "generate 4-6 retrieval queries in **English** from different angles.\n\n"
        "Cover the following angles:\n"
        "1. Original keywords: Direct English translation of the user's question keywords.\n"
        "2. Function/symbol names: Extract or infer relevant function names, class names, macros, global variables.\n"
        "3. File paths: Infer likely source file paths (e.g., module/main.c, config/parser.py).\n"
        "4. Call chains / data flow: Describe the execution or data flow (e.g., 'caller → callee → handler').\n"
        "5. Component / subsystem: Broader architectural context (e.g., 'alarm management subsystem', 'boot initialization module').\n"
        "6. Tags and domain: Generate a query using classification tags and domain identifiers "
        "(e.g., 'module-internals alarm dedup monitoring', 'end-to-end-flow startup init hardware').\n\n"
        "Rules:\n"
        "- Output one query per line. Do NOT number them.\n"
        "- ALL queries MUST be in English, regardless of the user's language.\n"
        "- Preserve key technical terms exactly (function names, file names, error codes, API names).\n"
        "- Each query should be concise and suitable for vector similarity search.",
    ),
    ("human", "{query}"),
])

# ── Multi-angle query + entity extraction (for Graph-enhanced retrieval) ──
#
# Same multi-angle strategy as MULTI_ANGLE_QUERY_PROMPT, but the LLM also
# extracts structured code entities that can be used directly for knowledge
# graph traversal.  This dramatically improves graph hit rates because:
#   - Natural language questions ("how is a device registered on the bus?")
#     rarely contain exact symbol names for regex extraction.
#   - The LLM can *infer* likely symbol/component/file names from context,
#     bridging the gap between human language and code identifiers.
#
# Output format: JSON with two top-level keys:
#   - "queries": list of 4-6 retrieval query strings (same as before)
#   - "entities": {"symbols": [...], "files": [...], "components": [...]}
MULTI_ANGLE_QUERY_WITH_ENTITIES_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a source code retrieval query expansion assistant. Given a user question (in any language), "
        "produce a JSON object with two keys:\n\n"
        '1. "queries": An array of 4-6 **English** retrieval queries from different angles:\n'
        "   - Original keywords (direct English translation)\n"
        "   - Function/symbol names (extract or infer relevant identifiers)\n"
        "   - File paths (infer likely source files)\n"
        "   - Call chains / data flow\n"
        "   - Component / subsystem (broader architectural context)\n"
        "   - Tags and domain identifiers\n\n"
        '2. "entities": An object with code entities extracted or inferred from the question:\n'
        '   - "symbols": Array of function names, class names, macros, struct names, global variables '
        "(include both explicitly mentioned AND reasonably inferred ones)\n"
        '   - "files": Array of likely source file paths (e.g., "base/bus.c", "include/udrv/driver.h")\n'
        '   - "components": Array of component/subsystem names (e.g., "bus", "driver", "alarm-manager")\n\n'
        "Rules:\n"
        "- Output ONLY valid JSON, no markdown fences, no explanation.\n"
        "- ALL queries MUST be in English.\n"
        "- Preserve technical terms exactly (function names, file names, error codes).\n"
        "- For entities, be generous: include plausible inferred names even if uncertain.\n"
        "- If no entities can be inferred for a category, use an empty array [].\n"
        "- Each query should be concise and suitable for vector similarity search.",
    ),
    ("human", "{query}"),
])

# ── Chunk Evaluation Prompt (R3) ────────────────────────────────

# R3 feature: Chunk-level relevance evaluation.
# Purpose: After retrieval, not all top-K chunks are necessarily relevant to
# the user's question (vector similarity is imperfect).  This prompt asks the
# LLM to perform a quick, coarse-grained triage of the retrieved chunk set
# as a whole, returning one of three verdicts:
#   - "relevant"     — all/most chunks are useful, proceed directly to answer
#                      generation without filtering.
#   - "partial"      — some chunks help but others are noise; triggers the
#                      per-chunk FILTER_CHUNKS_PROMPT to drop irrelevant ones.
#   - "insufficient" — none of the chunks meaningfully address the question;
#                      the system should report that the KB lacks the needed
#                      information rather than fabricating an answer from
#                      tangentially related content.
#
# This two-phase design (evaluate first, filter only when "partial") saves
# LLM calls in the common case where chunks are either clearly good or
# clearly bad.
# Evaluate retrieved chunk relevance
EVALUATE_CHUNKS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a retrieval relevance evaluator. Given a user question and retrieved chunks, "
        "evaluate whether the chunks are relevant to answering the question.\n\n"
        "Return EXACTLY one word:\n"
        "- relevant: Chunks contain information directly useful for answering the question.\n"
        "- partial: Some chunks are useful but significant gaps remain.\n"
        "- insufficient: Chunks are not relevant or too little useful information.\n\n"
        "Return ONLY the verdict word, nothing else.",
    ),
    (
        "human",
        "Question: {question}\n\n"
        "Retrieved chunks:\n{chunks_summary}",
    ),
])

# ── Follow-Up Classification Prompt (R5) ────────────────────────

# R5 feature: Follow-up detection for multi-turn conversations.
# Purpose: In a conversational session, the user may ask a follow-up question
# that refines, clarifies, or dives deeper into the same topic as the previous
# exchange.  In that case, the previously retrieved chunks are likely still
# relevant and can be reused, avoiding a redundant (and potentially less
# accurate) retrieval step.
#
# This prompt classifies the new message as:
#   - "same" — a follow-up/clarification on the current topic.  The caller
#     (classify_follow_up) returns True, signaling the orchestrator to skip
#     retrieval and reuse the cached chunks from the previous turn.
#   - "new"  — a distinct topic requiring fresh retrieval from the vector
#     store.  The caller returns False.
#
# The binary same/new output keeps parsing trivial and the classification
# fast (single LLM call with minimal tokens).

# Determine whether the current question is a follow-up on the same topic
# or a new topic requiring fresh retrieval
FOLLOW_UP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a conversation topic classifier. Given the recent conversation "
        "history and a new user message, determine whether the new message is:\n"
        "- same: A follow-up, clarification, or deeper dive on the SAME topic as the last exchange.\n"
        "- new: A question about a DIFFERENT topic requiring fresh information retrieval.\n\n"
        "Return EXACTLY one word: same or new. Nothing else.",
    ),
    (
        "human",
        "Recent conversation:\n{history}\n\nNew message: {question}",
    ),
])

# ── Chunk Filtering Prompt (for partial verdicts) ──────────────

# Second phase of the R3 feature: per-chunk filtering.
# This prompt is invoked only when EVALUATE_CHUNKS_PROMPT returns "partial",
# meaning the chunk set is a mix of relevant and irrelevant items.  The LLM
# classifies each numbered chunk individually using the compact "idx:label"
# format (e.g., "1:relevant\n2:irrelevant\n3:relevant").
#
# The evaluate_chunks() function in generator.py parses this output, keeps
# only the chunks labeled "relevant", and falls back to the full set if
# parsing fails or all chunks are marked irrelevant.

# When chunks are "partial", filter out irrelevant ones
FILTER_CHUNKS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a chunk relevance classifier. Given a question and numbered chunks, "
        "classify each chunk as relevant or irrelevant.\n\n"
        "Output format: one line per chunk, e.g.:\n"
        "1:relevant\n"
        "2:irrelevant\n"
        "3:relevant\n\n"
        "Only output the classifications, nothing else.",
    ),
    (
        "human",
        "Question: {question}\n\nChunks:\n{chunks_detail}",
    ),
])
