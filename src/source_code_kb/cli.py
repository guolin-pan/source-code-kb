"""CLI entry point — the primary user interface for Source Code Knowledge Base.

Available commands:
- sckb ingest   — Import JSONL data into the knowledge base
- sckb query    — Single-turn Q&A (retrieve, rerank, generate answer)
- sckb chat     — Interactive multi-turn conversation
- sckb serve    — Start the retrieval service API (no LLM dependency)
- sckb stats    — View knowledge base statistics
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from source_code_kb.config import load_config

# ── Theme ───────────────────────────────────────────────────────
# Custom Rich theme that defines semantic style tokens for all console output.
# Using named tokens (e.g. "info", "step") instead of inline styles keeps the
# visual language consistent and lets us change the look in one place.  Each key
# maps to a Rich style string; the console object below resolves them via
# markup tags like [info]...[/info] or [step]...[/step].

_THEME = Theme({
    "info": "dim cyan",           # Informational text (queries, stats)
    "success": "bold green",      # Success confirmations
    "warning": "bold yellow",     # Non-fatal warnings
    "error": "bold red",          # Error messages
    "step": "dim",                # Pipeline progress steps (retrieve, rerank, etc.)
    "prompt.mode": "dim cyan",    # Mode label in the interactive prompt
    "prompt.arrow": "bold cyan",  # Arrow symbol in the interactive prompt
    "header.title": "bold white",  # Banner / panel titles
    "header.sub": "dim white",    # Banner subtitle text
    "answer.border": "bright_cyan",  # Border colour for answer panels
    "source.key": "cyan",        # Key labels in source tables
    "source.val": "dim",         # Value text in source tables
})

# add_completion=False disables Typer's built-in shell completion so that
# our own prompt_toolkit completer handles Tab completions during chat.
app = typer.Typer(
    name="sckb",
    help="Source Code Knowledge Base — RAG knowledge base system for source code architecture analysis with data ingestion, intelligent Q&A, and retrieval services.",
    add_completion=False,
)
# Module-level Console instance shared by every command; the custom theme
# is injected here so all print() calls resolve our semantic tokens.
console = Console(theme=_THEME)

# ── Slash command definitions (for autocomplete) ────────────────
# These lists power the Tab-autocomplete in the interactive chat session.
# _SLASH_COMMANDS are the primary commands; the completer also includes
# common sub-arguments ("simple", "agent", "clear") so that typing
# "/mode <Tab>" or "/filter <Tab>" offers useful completions.
# sentence=True lets the completer match tokens anywhere in the input,
# not just at the very start, which is needed for sub-arguments.

_SLASH_COMMANDS = [
    "/help", "/quit", "/exit",
    "/sources", "/filter", "/mode", "/clear", "/stats",
]
_SLASH_COMPLETER = WordCompleter(
    _SLASH_COMMANDS + ["simple", "agent", "clear"],
    sentence=True,
)


# ── ingest command: data ingestion ──────────────────────────────


@app.command()
def ingest(
    inputs: list[str] = typer.Option(..., "--input", "-i", help="Path to a JSONL file or directory (repeatable)"),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection name"),
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config file"),
):
    """Import JSONL data into the knowledge base (vectorize and store in ChromaDB).

    Supports multiple --input options for batch processing:
      sckb ingest -i file1.jsonl -i file2.jsonl -i /path/to/dir

    When --input points to a directory, all *.jsonl files in that directory are loaded.

    Each input is processed sequentially: load → vectorize → store to database.
    """
    cfg = load_config(config_path)

    # Lazy imports: loading and indexing modules are heavy (they initialise
    # embedding models and database connections), so we import them here
    # rather than at module level to keep CLI startup fast for other commands.
    from source_code_kb.ingest.jsonl_loader import load_jsonl, load_jsonl_directory
    from source_code_kb.ingest.indexer import ingest_documents

    # Accumulators for the final summary printed after all inputs are processed.
    total_indexed = 0
    failed_paths = []

    # Process each --input path one at a time.  Error isolation: a failure in
    # one input does not abort the rest — it is recorded in failed_paths and
    # the loop continues so that as much data as possible is ingested.
    for idx, input_path in enumerate(inputs, 1):
        path = Path(input_path)

        console.print(f"\n[bold cyan]Input {idx}/{len(inputs)}:[/bold cyan] {path}")

        try:
            # Load documents from current input.
            # Directories are expanded to all contained *.jsonl files;
            # single files are loaded directly (strict=False tolerates
            # malformed lines instead of aborting the whole file).
            if path.is_dir():
                console.print(f"  [dim]Loading JSONL files from directory...[/dim]")
                documents = load_jsonl_directory(path)
            elif path.is_file():
                console.print(f"  [dim]Loading JSONL file...[/dim]")
                documents = load_jsonl(path, strict=False)
            else:
                # Path does not exist on disk — record as failed, skip.
                console.print(f"  [red]✗ Path not found[/red]")
                failed_paths.append(str(path))
                continue

            if not documents:
                # File existed but yielded zero usable documents (empty or
                # all lines were malformed).  Not an error, just a warning.
                console.print(f"  [yellow]⚠ No valid documents loaded, skipping[/yellow]")
                continue

            console.print(f"  [dim]Loaded {len(documents)} documents[/dim]")

            # Immediately vectorize and store to database — this embeds
            # each document and upserts it into the ChromaDB collection.
            # ingest_documents returns the count of *newly* added docs
            # (duplicates detected by doc ID are skipped).
            console.print(f"  [dim]Vectorizing and storing to database...[/dim]")
            count = ingest_documents(cfg, documents, collection_name=collection)
            total_indexed += count
            console.print(f"  [green]✓ Indexed {count} new documents[/green]")

        except Exception as e:
            # Catch-all: any unexpected error (e.g. DB connection lost,
            # permission denied) is printed but does not stop other inputs.
            console.print(f"  [red]✗ ERROR: {e}[/red]")
            failed_paths.append(str(path))
            continue

    # ── Final summary ──
    # Always printed so the user can see the aggregate result at a glance,
    # including which inputs (if any) failed.
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  [cyan]Total indexed:[/cyan] {total_indexed} documents")
    console.print(f"  [cyan]Successful inputs:[/cyan] {len(inputs) - len(failed_paths)}/{len(inputs)}")

    if failed_paths:
        console.print(f"  [yellow]Failed inputs:[/yellow] {len(failed_paths)}")
        for fp in failed_paths:
            console.print(f"    - {fp}")

    # Exit with code 1 if nothing was indexed at all — this lets shell
    # scripts detect a complete failure via the exit status.
    if total_indexed == 0:
        console.print(f"\n[red]✗ No documents were indexed.[/red]")
        raise typer.Exit(1)
    else:
        console.print(f"\n[green]✓ Ingestion complete![/green]")


# ── query command: single-turn Q&A ──────────────────────────────


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to query"),
    collection: str = typer.Option("default", "--collection", "-c"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    mode: str = typer.Option("simple", "--mode", "-m", help="Query mode: simple or agent"),
    config_path: Optional[str] = typer.Option(None, "--config"),
):
    """Single-turn Q&A — retrieve, rerank, and generate an answer."""
    cfg = load_config(config_path)

    # Lazy imports for the vector store and retriever — kept out of module
    # scope so that commands that do not need them (e.g. `serve`) start fast.
    from source_code_kb.ingest.indexer import create_vectorstore
    from source_code_kb.retrieval.retriever import HybridRetriever

    # Obtain the ChromaDB collection handle and wrap it in a HybridRetriever
    # which provides both dense (vector) and optional sparse (keyword) search.
    coll_obj = create_vectorstore(cfg, collection_name=collection)
    retriever = HybridRetriever(coll_obj, cfg)

    try:
        # ── Mode dispatch ──
        # "agent" mode delegates to the full LangGraph agent which autonomously
        # classifies the query, rewrites it, retrieves, evaluates, and generates.
        # "simple" mode runs a straightforward retrieve -> (optional) rerank ->
        # generate pipeline without the agent's autonomous decision loop.
        if mode == "agent":
            # Lazy import: the agent graph pulls in LangGraph and other heavy
            # dependencies that are unnecessary for simple-mode queries.
            from source_code_kb.agent.graph import run_agent

            console.print("[info]Running agent mode…[/info]")
            result = run_agent(question, cfg, retriever)
            console.print(
                Panel(
                    Markdown(result["answer"]),
                    title="[bold]Answer[/bold]",
                    title_align="left",
                    border_style="bright_cyan",
                    padding=(1, 2),
                )
            )
            _print_sources(result.get("sources", []))
        else:
            # Simple mode: direct vector search, no query rewriting.
            results = retriever.search(question, top_k=top_k)

            # Optional reranking: a cross-encoder model rescores and filters
            # results for higher precision.  Guarded by config flag so users
            # without a reranker model can still use the system.
            if cfg.retrieval.use_reranker:
                from source_code_kb.retrieval.reranker import rerank

                results = rerank(question, results, cfg, top_n=top_k)

            # Lazy import of the generator — keeps the LLM client out of
            # memory until we actually need to call the language model.
            from source_code_kb.generation.generator import generate_answer

            answer, sources = generate_answer(question, results, cfg)
            console.print(
                Panel(
                    Markdown(answer),
                    title="[bold]Answer[/bold]",
                    title_align="left",
                    border_style="bright_cyan",
                    padding=(1, 2),
                )
            )
            _print_sources(sources)
    except Exception as exc:
        # Catch connection / auth errors so the user gets a hint about config.
        console.print(f"\n[error]  ✗ Error: {exc}[/error]")
        console.print("[dim]  Check LLM service config (llm.base_url in config.yaml).[/dim]")
        raise typer.Exit(1)


# ── chat command: interactive conversation ──────────────────────


def _build_welcome_banner(collection: str, mode: str, model: str) -> Panel:
    """Build a professional welcome banner."""
    # Rich Text is built incrementally via .append() so each segment can
    # carry its own style.  This gives us fine-grained control over colours
    # within a single logical line — something Rich markup tags can do too,
    # but the Text API is cleaner for multi-segment composition.
    lines = Text()
    # Line 1: product name + tagline
    lines.append("SCKB", style="bold bright_cyan")
    lines.append("  Source Code Knowledge Base\n", style="bold white")
    # Line 2: session context — collection, mode, model separated by pipe chars
    lines.append(f"Collection: ", style="dim")
    lines.append(f"{collection}", style="cyan")
    lines.append(f"  │  Mode: ", style="dim")
    lines.append(f"{mode}", style="dim cyan")
    lines.append(f"  │  Model: ", style="dim")
    lines.append(f"{model}", style="cyan")
    # Line 3: quick-start hints — highlights slash commands and Tab key
    lines.append(f"\n\nType ", style="dim")
    lines.append("/help", style="bold cyan")
    lines.append(" for commands, ", style="dim")
    lines.append("/quit", style="bold cyan")
    lines.append(" to exit.  ", style="dim")
    lines.append("Tab", style="bold")
    lines.append(" autocompletes commands.", style="dim")
    # Wrap everything in a Panel for a boxed, polished look.
    return Panel(lines, border_style="bright_cyan", padding=(1, 2))


def _make_prompt_html(mode: str) -> HTML:
    """Build a styled prompt showing current mode."""
    # prompt_toolkit uses its own HTML-like markup, separate from Rich.
    # The prompt shows the current mode in brackets (e.g. [simple]) followed
    # by a bold arrow character, giving a visual cue similar to shell prompts.
    return HTML(f'<style fg="ansicyan">[{mode}]</style> <style fg="ansibrightcyan"><b>❯</b></style> ')


@app.command()
def chat(
    collection: str = typer.Option("default", "--collection", "-c"),
    mode: str = typer.Option("simple", "--mode", "-m", help="Initial mode: simple or agent"),
    config_path: Optional[str] = typer.Option(None, "--config"),
):
    """Interactive multi-turn conversation — supports contextual dialogue with retrieval."""
    cfg = load_config(config_path)

    # Lazy imports — chat depends on the session manager, vector store, and
    # retriever which are only needed for this command.
    from source_code_kb.chat.session import ChatSession
    from source_code_kb.ingest.indexer import create_vectorstore
    from source_code_kb.retrieval.retriever import HybridRetriever, SearchFilter

    # Initialise the ChromaDB collection, wrap it in a retriever, and create
    # a ChatSession that tracks conversation history, mode, and filters.
    coll_obj = create_vectorstore(cfg, collection_name=collection)
    retriever = HybridRetriever(coll_obj, cfg)
    session = ChatSession(mode=mode)

    console.print()
    console.print(_build_welcome_banner(collection, session.mode, cfg.llm.model))
    console.print()

    # prompt_toolkit PromptSession provides:
    #  - InMemoryHistory: up-arrow recalls previous inputs within this session
    #  - _SLASH_COMPLETER: Tab-complete for slash commands and their arguments
    #  - complete_while_typing=False: only show completions on explicit Tab,
    #    avoiding distracting pop-ups while the user is typing a question
    prompt_session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(),
        completer=_SLASH_COMPLETER,
        complete_while_typing=False,
    )

    # ── Main conversation loop ──
    # Each iteration: read input -> classify (slash command vs. question) ->
    # dispatch to simple/agent mode -> display answer + sources.
    while True:
        try:
            # Render the prompt with the current mode label (updates live
            # if the user switches modes with /mode).
            user_input = prompt_session.prompt(
                _make_prompt_html(session.mode),
            ).strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D or Ctrl-C gracefully exits the chat loop.
            console.print("\n[dim]Goodbye![/dim]")
            break

        # Ignore blank lines — just re-prompt.
        if not user_input:
            continue

        # Slash commands are handled separately and never sent to the LLM.
        if user_input.startswith("/"):
            _handle_chat_command(user_input, session, retriever, cfg)
            continue

        # Record the user message in the session before processing so that
        # the history text passed to the LLM includes this turn.
        session.add_user_message(user_input)
        history = session.get_history_text()

        try:
            # Dispatch to the appropriate processing pipeline based on the
            # current session mode.  Both return (answer_text, sources_list).
            if session.mode == "agent":
                answer, sources = _chat_agent_mode(user_input, cfg, retriever, session)
            else:
                answer, sources = _chat_simple_mode(
                    user_input, cfg, retriever, session, history,
                )

            # Persist the assistant's reply in session so it appears in future
            # history context and can be recalled by /sources.
            session.add_assistant_message(answer, sources)

            # Render the final answer as a Rich Markdown panel — Markdown()
            # converts the raw text into styled Rich renderables (headers,
            # code blocks, lists, etc.) before printing.
            if answer.strip():
                console.print()
                console.print(
                    Panel(
                        Markdown(answer),
                        border_style="bright_cyan",
                        title="[bold]Answer[/bold]",
                        title_align="left",
                        padding=(1, 2),
                    )
                )

            # Source report (R8): show a compact table of knowledge-base
            # sources that contributed to the answer, grouped by file.
            if sources:
                _print_source_report(sources)

            console.print()  # spacing between turns

        except Exception as exc:
            # Error is caught per-turn so one bad response does not kill the
            # entire chat session — the user can simply try again.
            console.print(f"\n[error]  ✗ Error: {exc}[/error]")
            console.print("[dim]  Check LLM service config (llm.base_url in config.yaml).[/dim]\n")


def _chat_simple_mode(
    user_input: str,
    cfg,
    retriever,
    session,
    history: str,
) -> tuple[str, list[dict]]:
    """Simple mode chat: multi-angle retrieval → eval → retry → stream answer."""
    import time

    # Lazy imports — generator utilities (extract sources, evaluate, stream)
    # and the query rewriter are only needed in simple mode.
    from source_code_kb.generation.generator import (
        _extract_sources,
        evaluate_chunks,
        generate_answer_stream,
    )
    from source_code_kb.retrieval.query_rewriter import generate_multi_angle_queries
    from source_code_kb.retrieval.retriever import SearchFilter

    t0 = time.time()  # Wall-clock timer for the full pipeline

    # ── R5: Follow-up reuse check ──
    # If the new question is a follow-up on the same topic as the previous
    # turn (detected by session.is_follow_up which uses lightweight LLM or
    # heuristic comparison), skip the entire retrieval pipeline and reuse
    # the chunks that were already fetched.  This saves latency and avoids
    # retrieving near-identical results.
    if session.is_follow_up(user_input, config=cfg) and session.last_chunks:
        console.print("[step]  ↻ Reusing last retrieval (same topic follow-up)[/step]")
        results = session.last_chunks
    else:
        # Build an optional metadata filter from the user's /filter settings.
        # SearchFilter fields (topic, section, language, etc.) are ANDed
        # together in the vector store query.
        search_filter = None
        if session.active_filter:
            search_filter = SearchFilter(
                topic=session.active_filter.get("topic"),
                section=session.active_filter.get("section"),
                language=session.active_filter.get("language"),
                component=session.active_filter.get("component"),
                domain=session.active_filter.get("domain"),
            )

        # ── R2 + R3: Multi-angle retrieval with evaluation retry loop ──
        # The outer loop runs up to max_rounds (3) times.  Each round:
        #   1. R2  — LLM generates multiple reformulated queries from the
        #            user's question (multi-angle) to improve recall.
        #   2. Retrieve — each query is run against the vector store; results
        #            are deduped by doc_id across all queries in the round.
        #   3. (Optional) Rerank — a cross-encoder rescores the union set.
        #   4. R3  — LLM evaluates the retrieved chunks for relevance,
        #            returning a verdict: "relevant", "partial", or
        #            "insufficient".
        #   If the verdict is "relevant" or "partial" with non-empty results,
        #   the loop exits early.  Otherwise it retries with fresh queries.
        results = []
        max_rounds = 3
        all_used_queries: list[str] = []  # Accumulated across all rounds for R5 storage

        for round_num in range(1, max_rounds + 1):
            # ── R2: Generate multi-angle English queries ──
            # The rewriter produces several search queries that approach the
            # topic from different angles (e.g. synonyms, related concepts)
            # to maximise recall in the vector store.
            with Status("[dim]Generating queries…[/dim]", console=console, spinner="dots"):
                t_query = time.time()
                queries = generate_multi_angle_queries(user_input, cfg)
                elapsed_q = time.time() - t_query
            all_used_queries.extend(queries)
            q_preview = "  │  ".join(queries[:3])
            console.print(
                f"[step]  ✓ Rewrite[/step] [dim](round {round_num}, {elapsed_q:.1f}s)[/dim]  "
                f"[cyan]{len(queries)}[/cyan] [dim]queries →[/dim] [info]{q_preview}[/info]"
            )

            # ── Retrieve + dedup by doc_id across multiple queries ──
            # Each query may return overlapping chunks; we track seen doc_ids
            # so the union set contains no duplicates.  The doc_id falls back
            # to the first 50 chars of content when metadata lacks an "id".
            with Status("[dim]Searching knowledge base…[/dim]", console=console, spinner="dots"):
                t_ret = time.time()
                seen_ids: set[str] = set()
                round_results = []
                for q in queries:
                    hits = retriever.search(q, search_filter=search_filter)
                    for r in hits:
                        doc_id = r.metadata.get("id", r.content[:50])
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            round_results.append(r)
                elapsed_r = time.time() - t_ret

            # ── Topic summarization ──
            # Count how many chunks belong to each topic and display the top
            # 5 as a quick diagnostic — helps the user see at a glance
            # whether retrieval targeted the right knowledge areas.
            topics: dict[str, int] = {}
            for r in round_results:
                t = r.metadata.get("topic", "?")
                topics[t] = topics.get(t, 0) + 1
            topic_summary = ", ".join(f"{t}({n})" for t, n in sorted(topics.items(), key=lambda x: -x[1])[:5])
            console.print(
                f"[step]  ✓ Retrieve[/step] [dim]({elapsed_r:.1f}s)[/dim]  "
                f"[cyan]{len(round_results)}[/cyan] [dim]chunks  │  Topics:[/dim] [info]{topic_summary}[/info]"
            )

            # ── Optional reranking ──
            # When enabled, a cross-encoder model (e.g. bge-reranker) scores
            # each chunk against the original user question and drops low-
            # scoring results.  This improves precision at the cost of an
            # extra model call.
            if cfg.retrieval.use_reranker and round_results:
                from source_code_kb.retrieval.reranker import rerank

                with Status("[dim]Reranking…[/dim]", console=console, spinner="dots"):
                    t_rr = time.time()
                    before = len(round_results)
                    round_results = rerank(user_input, round_results, cfg)
                    elapsed_rr = time.time() - t_rr
                console.print(
                    f"[step]  ✓ Rerank[/step] [dim]({elapsed_rr:.1f}s)[/dim]  "
                    f"{before} → [cyan]{len(round_results)}[/cyan] [dim]chunks[/dim]"
                )

            results = round_results

            # ── R3: Chunk evaluation with retry logic ──
            # evaluate_chunks asks the LLM whether the retrieved chunks are
            # sufficient to answer the question.  It returns a verdict string
            # plus a filtered subset of chunks deemed relevant.
            if not results:
                # No results at all — if retries remain, try another round;
                # otherwise give up.
                console.print("[step]  ✓ Evaluate[/step]  [error]no results[/error]")
                if round_num < max_rounds:
                    continue
                break

            with Status("[dim]Evaluating relevance…[/dim]", console=console, spinner="dots"):
                t_eval = time.time()
                verdict, filtered_results = evaluate_chunks(user_input, results, cfg)
                elapsed_e = time.time() - t_eval
            # Colour-code the verdict for visual scanning: green=good,
            # yellow=acceptable, red=needs retry.
            color = {"relevant": "green", "partial": "yellow", "insufficient": "red"}.get(verdict, "dim")
            console.print(
                f"[step]  ✓ Evaluate[/step] [dim]({elapsed_e:.1f}s)[/dim]  "
                f"[{color}]{verdict}[/{color}] [dim]({len(results)} chunks → {len(filtered_results)} kept)[/dim]"
            )
            results = filtered_results  # Use filtered chunks

            # Exit the retry loop if evaluation is acceptable.
            if verdict in ("relevant", "partial") and results:
                break
            # "insufficient" or all chunks filtered out — retry with new
            # queries if rounds remain.
            if round_num < max_rounds:
                console.print("[dim]    → Retrying with refined queries…[/dim]")

        # ── R5: Save retrieval state for follow-up reuse ──
        # Persist the final chunks and all queries used so that the next
        # turn can skip retrieval if it is a same-topic follow-up.
        session.save_last_retrieval(results, all_used_queries)

    # ── Streaming answer generation ──
    # Extract structured source metadata from the result chunks before
    # sending them to the generator.
    sources = _extract_sources(results) if results else []
    elapsed_prep = time.time() - t0
    console.print(
        f"[step]  ✓ Ready[/step] [dim]({elapsed_prep:.1f}s)[/dim]  [dim]Generating answer…[/dim]"
    )

    # Stream tokens from the LLM and collect them into a single string.
    # Tokens are buffered rather than printed because the final answer is
    # rendered later inside a Rich Markdown panel for better formatting.
    answer_parts: list[str] = []
    t_gen = time.time()
    for token in generate_answer_stream(
        user_input, results, cfg, history=history
    ):
        answer_parts.append(token)
    answer = "".join(answer_parts)

    elapsed_gen = time.time() - t_gen
    elapsed_total = time.time() - t0
    console.print(
        f"[step]  ✓ Done[/step]  [dim]Gen: {elapsed_gen:.1f}s  │  "
        f"Total: {elapsed_total:.1f}s  │  {len(answer)} chars[/dim]"
    )
    return answer, sources


def _chat_agent_mode(
    user_input: str,
    cfg,
    retriever,
    session,
) -> tuple[str, list[dict]]:
    """Agent mode chat: full token-level streaming + node progress display.

    Non-generating nodes show progress indicators as they complete.
    Generating nodes (generate, synthesize, compare) stream LLM tokens
    to the console in real-time via a callback.
    """
    from source_code_kb.agent.graph import run_agent_stream

    # _NODE_LABELS maps internal LangGraph node names to short human-readable
    # labels displayed in the progress output.  This decouples the user-facing
    # text from the graph implementation — if the graph renames a node, only
    # this dict needs updating.
    _NODE_LABELS = {
        "classify_query": "Classify",
        "rewrite": "Rewrite",
        "retrieve": "Retrieve",
        "rerank": "Rerank",
        "evaluate": "Evaluate",
        "generate": "Generate",
        "decompose": "Decompose",
        "sub_retrieve": "Sub-retrieve",
        "synthesize": "Synthesize",
        "compare": "Compare",
    }
    # _STREAMING_NODES identifies the nodes whose primary output is generated
    # text (i.e. LLM answer tokens).  These nodes use the _on_token callback
    # to push tokens as they arrive, rather than waiting for the node to
    # finish.  All other nodes only emit a one-line progress indicator.
    _STREAMING_NODES = {"generate", "synthesize", "compare"}
    # _streamed tracks whether _on_token was invoked at least once for the
    # current node so we know to print a trailing newline after the stream.
    _streamed = False
    # _token_buf collects tokens as they stream in, enabling post-processing
    # (e.g. counting characters) even though display is handled by the
    # callback itself.
    _token_buf: list[str] = []

    def _on_token(token: str) -> None:
        """Callback invoked by the agent graph for each LLM token.

        Uses `nonlocal _streamed` to signal to the outer loop that streaming
        output has occurred, so it can emit a newline when the node completes.
        """
        nonlocal _streamed
        _streamed = True
        _token_buf.append(token)

    console.print("[dim]Agent processing…[/dim]")
    answer = ""
    sources: list[dict] = []

    # run_agent_stream yields (node_name, node_output) pairs as each node in
    # the LangGraph completes.  The iteration order follows the graph's
    # topological execution.
    for node_name, node_output in run_agent_stream(
        user_input, cfg, retriever, on_token=_on_token,
    ):
        # Some nodes set _skip_display=True when they are no-ops (e.g. the
        # reranker node when reranking is disabled in config).
        if node_output.get("_skip_display", False):
            continue

        # Resolve a friendly label; fall back to the raw node name if unknown.
        label = _NODE_LABELS.get(node_name, node_name)
        status = node_output.get("_status", "")

        if node_name in _STREAMING_NODES:
            # For streaming nodes, the _on_token callback already printed
            # tokens to the console.  We just need to close the line.
            if _streamed:
                console.print()  # newline after token stream
                _streamed = False
                _token_buf.clear()
        elif status:
            # Non-streaming node with a status message (e.g. "3 queries").
            console.print(f"[step]  ✓ {label}[/step]  {status}")
        else:
            # Non-streaming node with no extra detail — just a checkmark.
            console.print(f"[step]  ✓ {label}[/step]")

        # Capture the latest answer and sources — nodes may overwrite these
        # as the graph progresses (e.g. synthesize replaces generate's output
        # with a merged answer).
        if "answer" in node_output:
            answer = node_output["answer"]
        if "sources" in node_output:
            sources = node_output["sources"]

    return answer, sources


def _handle_chat_command(cmd: str, session, retriever, config) -> None:
    """Handle slash commands during a chat session."""
    # Split into command + optional argument.  maxsplit=1 ensures that
    # arguments containing spaces (e.g. "/filter topic=data flow") are
    # preserved as a single string in parts[1].
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()

    # ── Dispatch: each branch handles one slash command ──

    if command in ("/quit", "/exit"):
        # Raise typer.Exit to break out of the infinite chat loop cleanly.
        raise typer.Exit(0)

    elif command == "/sources":
        # Re-display the sources from the most recent answer (useful if the
        # user scrolled past them).
        sources = session.get_last_sources()
        if sources:
            _print_sources(sources)
        else:
            console.print("[dim]No sources from last response.[/dim]")

    elif command == "/filter":
        # /filter with no argument shows current filters; with "key=value"
        # sets a metadata filter; with "clear" removes all filters.
        if len(parts) < 2:
            if session.active_filter:
                console.print(f"[info]Active filters:[/info] {session.active_filter}")
            else:
                console.print("[dim]No active filters. Usage: /filter topic=XXX[/dim]")
            return

        arg = parts[1]
        if "=" in arg:
            # Parse key=value and store in the session.  Only one value per
            # key is supported; setting the same key again overwrites it.
            key, value = arg.split("=", 1)
            session.set_filter(key.strip(), value.strip())
            console.print(f"[success]Filter set:[/success] {key.strip()} = {value.strip()}")
        elif arg.lower() == "clear":
            session.clear_filter()
            console.print("[success]Filters cleared.[/success]")

    elif command == "/mode":
        # Switch between "simple" and "agent" processing modes at runtime.
        # The new mode takes effect on the next user question.
        if len(parts) < 2:
            console.print(f"[info]Current mode:[/info] {session.mode}")
            return
        new_mode = parts[1].strip().lower()
        if new_mode in ("simple", "agent"):
            session.mode = new_mode
            console.print(f"[success]Mode switched to:[/success] {new_mode}")
        else:
            console.print("[error]Invalid mode. Use: simple or agent[/error]")

    elif command == "/clear":
        # Wipe conversation history so subsequent turns have no prior context.
        session.clear_history()
        console.print("[success]History cleared.[/success]")

    elif command == "/stats":
        # Quick peek at the current collection without leaving the chat.
        from source_code_kb.ingest.indexer import get_collection_stats

        stats = get_collection_stats(config)
        console.print(f"[info]Collection stats:[/info] {stats}")

    elif command == "/help":
        # Render a borderless Rich Table as a concise command reference.
        # box=None removes cell borders for a cleaner look.
        help_table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 2),
        )
        help_table.add_column("Command", style="bold cyan", min_width=22)
        help_table.add_column("Description", style="dim")
        help_table.add_row("/sources", "Show sources cited in the last answer")
        help_table.add_row("/filter topic=X", "Set a metadata filter")
        help_table.add_row("/filter clear", "Clear all active filters")
        help_table.add_row("/mode simple|agent", "Switch query mode")
        help_table.add_row("/clear", "Clear conversation history")
        help_table.add_row("/stats", "Show collection statistics")
        help_table.add_row("/help", "Show this help")
        help_table.add_row("/quit", "Exit chat")
        console.print()
        console.print(help_table)
        console.print()

    else:
        # Unknown command — show an error with a hint to /help.
        console.print(f"[error]Unknown command:[/error] {command}  [dim]Type /help for available commands.[/dim]")


# ── serve command: start retrieval service ──────────────────────


@app.command()
def serve(
    port: int = typer.Option(8765, "--port", "-p", help="Service port"),
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Service bind address"),
    config_path: Optional[str] = typer.Option(None, "--config"),
):
    """Start the retrieval service API — provides retrieval only, no LLM dependency."""
    cfg = load_config(config_path)

    # Lazy import of uvicorn (ASGI server) and the FastAPI app factory.
    # The serve command intentionally does NOT depend on any LLM modules
    # so it can run on machines without GPU or LLM access.
    import uvicorn

    from source_code_kb.server.app import create_app

    # create_app wires up the FastAPI routes for search, health, etc.
    app_instance = create_app(cfg)
    # Print a startup banner with the key endpoints before blocking on serve.
    console.print(
        Panel(
            f"[bold]SCKB Retrieval Service[/bold]\n"
            f"URL: http://{host}:{port}\n"
            f"Docs: http://{host}:{port}/docs\n"
            f"API: http://{host}:{port}/api/v1/search",
            border_style="blue",
        )
    )
    # uvicorn.run blocks the process and serves requests until interrupted.
    # It handles signal handling (SIGINT/SIGTERM) internally.
    uvicorn.run(app_instance, host=host, port=port)


# ── stats command: statistics ───────────────────────────────────


@app.command()
def stats(
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config_path: Optional[str] = typer.Option(None, "--config"),
):
    """View knowledge base statistics (collection list, document count, topic list)."""
    cfg = load_config(config_path)

    from source_code_kb.ingest.indexer import get_collection_stats, get_collection_topics, list_collections

    if collection:
        info = get_collection_stats(cfg, collection)
        topics = get_collection_topics(cfg, collection)

        tbl = Table(title=f"Collection: {collection}")
        tbl.add_column("Property", style="cyan")
        tbl.add_column("Value", style="green")
        tbl.add_row("Exists", str(info.get("exists", False)))
        tbl.add_row("Documents", str(info.get("count", 0)))
        tbl.add_row("Topics", str(len(topics)))
        tbl.add_row("Metadata Keys", ", ".join(info.get("sample_metadata_keys", [])))
        console.print(tbl)

        if topics:
            console.print("\n[cyan]Topics:[/cyan]")
            for t in topics:
                console.print(f"  • {t}")
    else:
        names = list_collections(cfg)
        if not names:
            console.print("[yellow]No collections found.[/yellow]")
            return

        overview = Table(title="Collections")
        overview.add_column("Name", style="cyan")
        overview.add_column("Documents", style="green")
        for name in names:
            info = get_collection_stats(cfg, name)
            overview.add_row(name, str(info.get("count", 0)))
        console.print(overview)


# ── helper functions ────────────────────────────────────────────


def _print_sources(sources: list[dict]) -> None:
    """Print cited sources in a table format."""
    if not sources:
        return

    # Detailed source table used by the `query` command and `/sources` slash
    # command.  Each source chunk is shown as its own row with topic, section,
    # references, and relevance score — giving full transparency into what
    # the retriever returned.  show_lines=True draws horizontal rules between
    # rows for readability when there are many sources.
    table = Table(title="Sources", show_lines=True, border_style="dim")
    table.add_column("Topic", style="cyan", max_width=30)
    table.add_column("Section", style="green", max_width=20)
    table.add_column("References", style="dim", max_width=40)
    table.add_column("Score", style="yellow", width=8)

    for src in sources:
        # Score is formatted to 4 decimal places only when present;
        # absent scores render as an empty cell.
        table.add_row(
            src.get("topic", ""),
            src.get("section", ""),
            src.get("references", ""),
            f"{src.get('score', 0):.4f}" if src.get("score") else "",
        )

    console.print(table)


def _print_source_report(sources: list[dict]) -> None:
    """Print a source report table after chat answers (R8).

    Groups sources by source file and shows tags + chunk counts.
    """
    if not sources:
        return

    # ── Group by source file ──
    # Multiple chunks may come from the same source file.  We aggregate them
    # into a single row showing the source path, any domain tags collected
    # across its chunks, and the total chunk count.  This keeps the report
    # compact — the user sees *which files* contributed, not every chunk.
    source_map: dict[str, dict] = {}
    for s in sources:
        # Use the "source" field as the grouping key; fall back to
        # "topic/section" if the source field is missing or empty.
        key = s.get("source", "") or f"{s.get('topic', '')}/{s.get('section', '')}"
        if key not in source_map:
            source_map[key] = {"tags": set(), "count": 0, "domain": s.get("domain", "")}
        source_map[key]["count"] += 1
        # Collect unique domain tags across all chunks for this source.
        if s.get("domain"):
            source_map[key]["tags"].add(s["domain"])

    # Render a minimal borderless table (box=None) aligned left with the
    # "KB Sources" title, matching the understated style of pipeline
    # progress output rather than the heavier bordered tables.
    table = Table(
        show_header=True,
        header_style="dim bold",
        box=None,
        padding=(0, 1),
        title="[dim]KB Sources[/dim]",
        title_style="dim",
        title_justify="left",
    )
    table.add_column("Source", style="cyan")
    table.add_column("Tags", style="dim")
    table.add_column("Chunks", style="yellow", justify="right")

    for source, info in source_map.items():
        # Tags are sorted alphabetically for stable, scannable output.
        table.add_row(
            source,
            ", ".join(sorted(info["tags"])) if info["tags"] else "",
            str(info["count"]),
        )

    console.print(table)


if __name__ == "__main__":
    app()
