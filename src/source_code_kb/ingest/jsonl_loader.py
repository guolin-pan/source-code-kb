"""JSONL data loader — parses JSONL files into LangChain Document objects.

Schema v4 field definitions:

Required fields (13):
- id: str — Unique document identifier, format "{domain}:{topic}:{section}"
- content: str — Knowledge chunk body text (mapped to Document.page_content)
- domain: str — Knowledge domain classification
- topic: str — Parent topic name
- section: str — Section heading
- scope: list[str] — Applicability scope (boards/products/versions)
- tags: list[str] — Tag list (minimum 2)
- confidence: float — Confidence score (0.0-1.0)
- source: str — Source identifier
- updated_at: str — Last update timestamp (ISO 8601)
- files: list[str] — Source file paths relevant to this knowledge chunk
- symbols: list[str] — Code symbols (functions, classes, variables) covered
- language: str — Primary programming language

Optional fields (9):
- meta: dict — Extension metadata (stored as JSON string)
- component: str — Logical component or subsystem name
- call_chains: list[str] — Function/method call chains
- api_exports: list[str] — Exported API identifiers
- api_imports: list[str] — Imported API identifiers
- ipc_mechanism: list[str] — Inter-process communication mechanisms
- messages_send: list[str] — Outgoing message types
- messages_receive: list[str] — Incoming message types
- shared_data: list[str] — Shared data structures or stores
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from rich.console import Console

console = Console()

# ── Schema validation constants ──────────────────────────────

VALID_DOMAINS = {
    "module-internals",
    "module-interface",
    "end-to-end-flow",
    "system-constraints",
    "data-model",
    "build-deploy",
}


def _validate_record(record: dict, line_no: int) -> list[str]:
    """Validate a single JSONL record against the JSONL schema.

    Returns a list of error messages (empty if valid).
    """
    # Validation strategy: collect ALL errors in a single pass rather than
    # failing on the first problem.  This lets callers (especially in non-strict
    # mode) log every issue in one go, making JSONL authoring much easier to
    # debug.  The checks are ordered to mirror the schema definition: required
    # fields first, then optional fields.  Each check follows a three-tier
    # pattern: (1) presence, (2) type, (3) domain/range constraint.
    errors: list[str] = []

    # --- Required string fields ---
    # These six fields must all be present, be strings, and be non-blank.
    # They form the minimal textual skeleton of every knowledge chunk.
    for field in ("content", "domain", "topic", "section", "source", "updated_at"):
        val = record.get(field)
        if val is None:
            errors.append(f"L{line_no} {field}: missing")
        elif not isinstance(val, str):
            errors.append(f"L{line_no} {field}: expected string, got {type(val).__name__}")
        elif not val.strip():
            errors.append(f"L{line_no} {field}: empty")

    # Content minimum-length guard: chunks shorter than 50 characters are almost
    # certainly truncated or placeholder text, so reject them early.
    content = record.get("content", "")
    if isinstance(content, str) and content.strip() and len(content) < 50:
        errors.append(f"L{line_no} content: too short ({len(content)} chars, min 50)")

    # Domain is a closed enum — only the values in VALID_DOMAINS are accepted.
    # This prevents typos from silently creating new categories.
    domain = record.get("domain", "")
    if isinstance(domain, str) and domain and domain not in VALID_DOMAINS:
        errors.append(f"L{line_no} domain: '{domain}' not in {sorted(VALID_DOMAINS)}")

    # --- Required list fields ---

    # scope must be a non-empty list; it tells downstream consumers which
    # boards/products/versions the knowledge chunk applies to.
    scope = record.get("scope")
    if scope is None:
        errors.append(f"L{line_no} scope: missing")
    elif not isinstance(scope, list) or len(scope) == 0:
        errors.append(f"L{line_no} scope: must be non-empty list")

    # tags must contain at least 2 items to ensure meaningful categorization.
    tags = record.get("tags")
    if tags is None:
        errors.append(f"L{line_no} tags: missing")
    elif not isinstance(tags, list) or len(tags) < 2:
        errors.append(f"L{line_no} tags: must be list with >= 2 items")

    # --- Required numeric field ---

    # confidence is a float in [0.0, 1.0].  Accept int as well (JSON parsers
    # may decode "1" as int) and cast to float for the range check.
    confidence = record.get("confidence")
    if confidence is None:
        errors.append(f"L{line_no} confidence: missing")
    elif not isinstance(confidence, (int, float)):
        errors.append(f"L{line_no} confidence: expected number")
    elif not 0.0 <= float(confidence) <= 1.0:
        errors.append(f"L{line_no} confidence: {confidence} out of [0.0, 1.0]")

    # --- Timestamp format check ---

    # updated_at must conform to ISO 8601.  The regex is intentionally lenient
    # about optional time/timezone parts (date-only "2024-01-15" is accepted)
    # to accommodate varying LLM output formats.
    updated_at = record.get("updated_at", "")
    if isinstance(updated_at, str) and updated_at.strip():
        iso_re = r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$"
        if not re.match(iso_re, updated_at):
            errors.append(f"L{line_no} updated_at: not valid ISO 8601")

    # --- Optional dict field ---

    # meta is an optional extension point.  When present it must be a dict so
    # that it can be safely serialized to a JSON string downstream.
    meta = record.get("meta")
    if meta is not None and not isinstance(meta, dict):
        errors.append(f"L{line_no} meta: expected object, got {type(meta).__name__}")

    # ── New required fields (v4) ──────────────────────────────────
    # These fields were added in schema v4 to carry first-class code metadata
    # (file paths, symbol names, programming language).

    # files: non-empty list of strings — source file paths for traceability.
    files = record.get("files")
    if files is None:
        errors.append(f"L{line_no} files: missing")
    elif not isinstance(files, list) or len(files) == 0:
        errors.append(f"L{line_no} files: must be non-empty list")
    elif not all(isinstance(f, str) for f in files):
        errors.append(f"L{line_no} files: all items must be strings")

    # symbols: non-empty list of strings — function/class/variable names that
    # this knowledge chunk covers.
    symbols = record.get("symbols")
    if symbols is None:
        errors.append(f"L{line_no} symbols: missing")
    elif not isinstance(symbols, list) or len(symbols) == 0:
        errors.append(f"L{line_no} symbols: must be non-empty list")
    elif not all(isinstance(s, str) for s in symbols):
        errors.append(f"L{line_no} symbols: all items must be strings")

    # language: the primary programming language, required so retrieval can
    # filter or boost results by language context.
    language = record.get("language")
    if language is None:
        errors.append(f"L{line_no} language: missing")
    elif not isinstance(language, str):
        errors.append(f"L{line_no} language: expected string, got {type(language).__name__}")
    elif not language.strip():
        errors.append(f"L{line_no} language: empty")

    # ── Optional fields type checking (v4) ────────────────────────
    # These fields enrich the knowledge chunk with inter-component relationships
    # (call chains, APIs, IPC, messages, shared data).  They are optional
    # because not every chunk describes cross-cutting concerns.

    # component: optional string identifying the logical subsystem.
    component = record.get("component")
    if component is not None and not isinstance(component, str):
        errors.append(f"L{line_no} component: expected string, got {type(component).__name__}")

    # Batch-validate all optional list-of-strings fields with a single loop
    # to avoid repetitive per-field boilerplate.
    for field_name in (
        "call_chains", "api_exports", "api_imports",
        "ipc_mechanism", "messages_send", "messages_receive", "shared_data",
    ):
        val = record.get(field_name)
        if val is not None:
            if not isinstance(val, list):
                errors.append(f"L{line_no} {field_name}: expected list, got {type(val).__name__}")
            elif not all(isinstance(v, str) for v in val):
                errors.append(f"L{line_no} {field_name}: all items must be strings")

    return errors


def _make_doc_id(domain: str, topic: str, section: str) -> str:
    """Generate a deterministic document ID from domain + topic + section.

    Args:
        domain: Knowledge domain.
        topic: Topic name.
        section: Section heading.

    Returns:
        A formatted document ID string.
    """
    # Build a composite key from the three fields that together uniquely
    # identify a knowledge chunk within the corpus.
    key = f"{domain}:{topic}:{section}"

    # Use SHA-256 to produce a deterministic, fixed-length hash suffix.
    # Determinism is critical: re-ingesting the same JSONL file must yield the
    # same IDs so that ChromaDB's dedup logic (which keys on ID) can detect
    # duplicates without content comparison.  Only the first 8 hex chars are
    # kept — 32 bits is sufficient to avoid collisions within a single
    # knowledge base while keeping the ID human-readable.
    short_hash = hashlib.sha256(key.encode()).hexdigest()[:8]

    # The final ID preserves the human-readable key and appends the hash for
    # uniqueness, e.g. "module-internals:scheduler:init:a1b2c3d4".
    return f"{domain}:{topic}:{section}:{short_hash}"


def load_jsonl(file_path: str | Path, strict: bool = False) -> list[Document]:
    """Load a single JSONL file and return a list of LangChain Documents.

    Each JSON line is parsed and validated against the JSONL schema:
    - content → page_content
    - remaining fields → metadata (list types preserved as-is)

    When strict=True, invalid records cause a ValueError.
    When strict=False (default), invalid records are skipped with a warning.

    Args:
        file_path: Path to the JSONL file.
        strict: If True, raise on validation errors; if False, skip bad records.

    Returns:
        A list of Document objects.

    Raises:
        FileNotFoundError: File does not exist.
        ValueError: JSON parsing error or schema validation failure (strict mode only).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    documents: list[Document] = []
    skipped = 0

    # Parse the file line-by-line rather than loading it entirely into memory.
    # JSONL format has one JSON object per line, so streaming is both memory-
    # efficient and allows precise per-line error reporting.
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            # Skip blank lines (common at end-of-file or between sections).
            line = line.strip()
            if not line:
                continue

            # --- JSON parsing with strict/non-strict error handling ---
            # In strict mode, a single malformed line aborts the whole load so
            # that CI pipelines can catch data quality issues.  In non-strict
            # mode (the default), bad lines are logged and skipped so that one
            # corrupt record does not block the rest of the file.
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON at line {line_no} in {path.name}: {e}"
                if strict:
                    raise ValueError(error_msg) from e
                console.print(f"[red]ERROR {path.name}:{line_no}:[/red] Invalid JSON - {e}")
                skipped += 1
                continue

            # Schema validation — delegates to _validate_record which returns
            # a list of every problem found (not just the first).  Strict mode
            # raises; non-strict mode warns and moves on.
            errors = _validate_record(record, line_no)
            if errors:
                if strict:
                    msg = f"Schema validation failed in {path.name}:\n  " + "\n  ".join(errors)
                    raise ValueError(msg)
                console.print(f"[yellow]WARN {path.name}:{line_no}:[/yellow] {errors[0]}")
                skipped += 1
                continue

            # --- Extract core fields from the validated record ---

            content = record.get("content", "")

            domain = record.get("domain", "")
            topic = record.get("topic", "")
            section = record.get("section", "")

            # Prefer the record's explicit "id" field when present; fall back to
            # a deterministic hash-based ID so every document has a stable
            # identifier for ChromaDB dedup.
            doc_id = record.get("id") or _make_doc_id(domain, topic, section)

            # --- Type coercion for list fields: str -> single-element list ---
            # Some LLM-generated JSONL emits a bare string instead of a
            # single-item list (e.g. "scope": "global" instead of ["global"]).
            # Rather than rejecting these records, silently wrap the string in a
            # list to be resilient to minor formatting variance.
            scope = record.get("scope", [])
            if isinstance(scope, str):
                scope = [scope]
            tags = record.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]

            confidence = float(record.get("confidence", 0.5))
            source = record.get("source", str(path.name))
            updated_at = record.get("updated_at", "")

            # Serialize the optional meta dict to a JSON string so downstream
            # code can store it as a simple string value in ChromaDB metadata.
            meta = record.get("meta")
            meta_str = json.dumps(meta, ensure_ascii=False) if meta else ""

            # --- First-class code metadata (v4 schema) ---
            # Same str->list coercion applied to every list-typed code field.
            files = record.get("files", [])
            if isinstance(files, str):
                files = [files]
            symbols = record.get("symbols", [])
            if isinstance(symbols, str):
                symbols = [symbols]
            language = record.get("language", "")

            # --- Optional code metadata fields ---
            # All follow the same coercion pattern.  Default to empty
            # list/string so that downstream code never encounters None.
            component = record.get("component", "")
            call_chains = record.get("call_chains", [])
            if isinstance(call_chains, str):
                call_chains = [call_chains]
            api_exports = record.get("api_exports", [])
            if isinstance(api_exports, str):
                api_exports = [api_exports]
            api_imports = record.get("api_imports", [])
            if isinstance(api_imports, str):
                api_imports = [api_imports]
            ipc_mechanism = record.get("ipc_mechanism", [])
            if isinstance(ipc_mechanism, str):
                ipc_mechanism = [ipc_mechanism]
            messages_send = record.get("messages_send", [])
            if isinstance(messages_send, str):
                messages_send = [messages_send]
            messages_receive = record.get("messages_receive", [])
            if isinstance(messages_receive, str):
                messages_receive = [messages_receive]
            shared_data = record.get("shared_data", [])
            if isinstance(shared_data, str):
                shared_data = [shared_data]

            # --- Assemble the metadata dict ---
            # All fields except "content" go into metadata.  Lists are kept in
            # native Python list form here; flattening to ChromaDB-compatible
            # scalar types happens later in the indexer's _flatten_metadata().
            metadata: dict[str, Any] = {
                "id": doc_id,
                "domain": domain,
                "topic": topic,
                "section": section,
                "scope": scope,
                "tags": tags,
                "confidence": confidence,
                "source": source,
                "updated_at": updated_at,
                "meta": meta_str,
                # New first-class code metadata
                "files": files,
                "symbols": symbols,
                "language": language,
                "component": component,
                "call_chains": call_chains,
                "api_exports": api_exports,
                "api_imports": api_imports,
                "ipc_mechanism": ipc_mechanism,
                "messages_send": messages_send,
                "messages_receive": messages_receive,
                "shared_data": shared_data,
            }

            # page_content holds the human-readable chunk text that will also be
            # stored as the ChromaDB "document"; metadata rides alongside it.
            documents.append(Document(page_content=content, metadata=metadata))

    if skipped:
        console.print(f"[yellow]Warning: skipped {skipped} invalid record(s) in {path.name}[/yellow]")

    return documents


def load_jsonl_directory(dir_path: str | Path) -> list[Document]:
    """Load all .jsonl files from a directory.

    Files are loaded in sorted order to ensure deterministic results.
    If individual files fail to load, errors are reported but processing continues.

    Args:
        dir_path: Path to the directory containing JSONL files.

    Returns:
        A merged list of Document objects from all successfully loaded files.

    Raises:
        NotADirectoryError: Path is not a directory.
    """
    path = Path(dir_path)
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    documents: list[Document] = []
    failed_files = []

    # Sort the glob results so that file processing order is deterministic
    # across platforms and runs.  This makes logs reproducible and ensures that
    # if two files contribute conflicting IDs, the winner is always the one
    # that comes first alphabetically.
    jsonl_files = sorted(path.glob("*.jsonl"))
    if not jsonl_files:
        console.print(f"[yellow]Warning: No .jsonl files found in {path}[/yellow]")
        return documents

    # --- Error isolation ---
    # Each file is loaded independently inside its own try/except.  A
    # catastrophic failure in one file (e.g. file-level encoding issue) does
    # not prevent the remaining files from being loaded.  Individual bad
    # *records* within a file are already handled by load_jsonl's non-strict
    # mode (strict=False), so this outer try/except only catches file-level
    # errors such as permission or I/O problems.
    for jsonl_file in jsonl_files:
        try:
            file_docs = load_jsonl(jsonl_file, strict=False)
            documents.extend(file_docs)
        except Exception as e:
            console.print(f"[red]ERROR loading {jsonl_file.name}:[/red] {e}")
            failed_files.append(jsonl_file.name)
            continue

    if failed_files:
        console.print(f"[yellow]⚠ Failed to load {len(failed_files)} file(s) from directory:[/yellow]")
        for fn in failed_files:
            console.print(f"  - {fn}")

    return documents
